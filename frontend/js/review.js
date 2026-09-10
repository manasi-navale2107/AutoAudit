/**
 * Planner clarification and Data Engineering review controls.
 *
 * app.js creates one controller after initI18n():
 *   createReviewController({ onRunUpdate, onBusyChange })
 *
 * render(record) accepts a complete GET /runs/{id} response.
 * setEnabled(true) enables decisions after the application is ready.
 * refresh() fetches the selected run and checks uncertain submissions.
 * clear() hides the panels; drafts remain in memory until destroy().
 *
 * app.js must discard older polling requests when selection changes or
 * onBusyChange(true) fires. A false render() result means an older response.
 *
 * No request or DOM change happens when this module is imported.
 */

import {
  api,
  ApiError,
  isAbortError,
} from "./api.js";

import {
  t,
  getLanguage,
  formatNumber,
  onLanguageChange,
} from "./i18n.js";


const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const MAX_FEEDBACK = 4_000;
const MAX_SAVED_REVIEWS = 20;

const controllers = new WeakMap();

// Optional catalogue key with multilingual fallbacks.
const RETRY_HINT = Object.freeze({
  en: "The latest status still shows this review. You can resend only your previous decision.",
  mr: "नवीनतम स्थितीत हाच प्रस्ताव किंवा प्रश्न प्रलंबित आहे. फक्त आधीचा निर्णय पुन्हा पाठवता येईल.",
  hi: "ताज़ा स्थिति में यही प्रस्ताव या प्रश्न बाकी है. आप केवल अपना पिछला निर्णय फिर भेज सकते हैं.",
});


function isRecord(value) {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
  );
}


function invalidResponse() {
  return new ApiError(
    "The review response has an invalid structure.",
    {
      code: "invalid_response",
    },
  );
}


function canonical(value) {
  return JSON.stringify(value, (_, item) => {
    if (!isRecord(item)) {
      return item;
    }

    return Object.fromEntries(
      Object.entries(item).sort(([a], [b]) => {
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
      }),
    );
  });
}


function questionText(value) {
  if (
    typeof value === "string"
    && value.trim()
  ) {
    return value;
  }

  // Preserve structured questions, including their answer choices.
  if (
    isRecord(value)
    && Object.keys(value).length
  ) {
    return JSON.stringify(value, null, 2);
  }

  throw invalidResponse();
}


function readRecord(record) {
  if (
    !isRecord(record)
    || typeof record.run_id !== "string"
    || !UUID_PATTERN.test(record.run_id)
    || typeof record.status !== "string"
    || typeof record.updated_at !== "string"
    || !Array.isArray(record.interrupts)
  ) {
    throw invalidResponse();
  }

  const updatedAt = Date.parse(record.updated_at);

  if (!Number.isFinite(updatedAt)) {
    throw invalidResponse();
  }

  const runId = record.run_id.toLowerCase();

  const result = {
    runId,
    updatedAt,
    review: null,
  };

  const expectedKind =
    record.status === "awaiting_review"
      ? "cleaning_review"
      : record.status === "awaiting_clarification"
        ? "planner_clarification"
        : null;

  // Queued/running responses may still contain an old checkpoint interrupt.
  if (!expectedKind) {
    return result;
  }

  if (record.interrupts.length !== 1) {
    throw invalidResponse();
  }

  const interrupt = record.interrupts[0];
  const payload = interrupt?.value;

  if (
    typeof interrupt?.id !== "string"
    || !interrupt.id.trim()
    || !isRecord(payload)
    || payload.kind !== expectedKind
    || !Number.isSafeInteger(payload.revision)
    || payload.revision < 1
  ) {
    throw invalidResponse();
  }

  if (expectedKind === "planner_clarification") {
    if (
      !Array.isArray(payload.questions)
      || !payload.questions.length
    ) {
      throw invalidResponse();
    }

    payload.questions.forEach(questionText);
  } else {
    const proposal = payload.proposal;

    if (
      !isRecord(proposal)
      || typeof proposal.summary !== "string"
      || !proposal.summary.trim()
      || !Array.isArray(proposal.policies)
      || !proposal.policies.length
      || !Array.isArray(proposal.unresolved_questions)
      || proposal.unresolved_questions.some(
        (value) => (
          typeof value !== "string"
          || !value.trim()
        ),
      )
    ) {
      throw invalidResponse();
    }

    const fields = [
      "column",
      "action",
      "feature_kind",
      "missing_strategy",
      "scaling",
      "encoding",
      "reason",
    ];

    const columns = new Set();

    for (const policy of proposal.policies) {
      if (
        !isRecord(policy)
        || fields.some(
          (field) => (
            typeof policy[field] !== "string"
            || !policy[field].trim()
          ),
        )
        || columns.has(policy.column)
        || !["keep", "exclude"].includes(policy.action)
      ) {
        throw invalidResponse();
      }

      columns.add(policy.column);

      if (
        policy.fill_value != null
        && !["string", "number", "boolean"].includes(
          typeof policy.fill_value,
        )
      ) {
        throw invalidResponse();
      }
    }
  }

  const snapshot = JSON.parse(
    JSON.stringify(payload),
  );

  const key = canonical([
    runId,
    interrupt.id,
    snapshot,
  ]);

  result.review = {
    runId,
    key,
    kind: expectedKind,
    revision: payload.revision,
    payload: snapshot,
  };

  return result;
}


function remember(map, key, value) {
  map.delete(key);
  map.set(key, value);

  while (map.size > MAX_SAVED_REVIEWS) {
    map.delete(
      map.keys().next().value,
    );
  }
}


function notify(callback, value) {
  try {
    const result = callback(value);

    if (
      result
      && typeof result.catch === "function"
    ) {
      result.catch((error) => {
        console.error(
          "AutoAudit review callback failed.",
          error,
        );
      });
    }
  } catch (error) {
    console.error(
      "AutoAudit review callback failed.",
      error,
    );
  }
}


export function createReviewController({
  onRunUpdate,
  onBusyChange = () => {},
} = {}) {
  if (
    typeof onRunUpdate !== "function"
    || typeof onBusyChange !== "function"
  ) {
    throw new TypeError(
      "Provide onRunUpdate and an optional onBusyChange callback.",
    );
  }

  const ids = {
    reviewPanel: "review-panel",
    reviewForm: "review-form",
    reviewFields: "review-fields",
    feedback: "review-feedback",
    reviewError: "review-error",
    reviewStatus: "review-submit-status",
    revision: "review-revision",
    summary: "proposal-summary",
    policies: "policy-table-body",
    policyTemplate: "policy-row-template",
    questionsPanel: "cleaning-questions-panel",
    questions: "cleaning-questions",
    approve: "approve-review-button",
    changes: "change-review-button",
    reject: "reject-review-button",
    clarificationPanel: "clarification-panel",
    clarificationForm: "clarification-form",
    clarificationFields: "clarification-fields",
    answers: "clarification-answer",
    clarificationError: "clarification-error",
    clarificationQuestions: "clarification-questions",
    answer: "submit-clarification-button",
    rejectClarification: "reject-clarification-button",
  };

  const elements = Object.fromEntries(
    Object.entries(ids).map(([name, id]) => {
      const element = document.getElementById(id);

      if (!element) {
        throw new Error(
          `Missing review element: #${id}`,
        );
      }

      return [name, element];
    }),
  );

  if (controllers.has(elements.reviewForm)) {
    throw new Error(
      "Review controls are already initialized.",
    );
  }

  if (!elements.policyTemplate.content) {
    throw new Error(
      "The policy row template is invalid.",
    );
  }

  const actionButtons = [
    [
      elements.approve,
      "approve",
      "cleaning_review",
    ],
    [
      elements.changes,
      "changes",
      "cleaning_review",
    ],
    [
      elements.reject,
      "reject",
      "cleaning_review",
    ],
    [
      elements.answer,
      "answer",
      "planner_clarification",
    ],
    [
      elements.rejectClarification,
      "reject",
      "planner_clarification",
    ],
  ];

  const clarificationStatus = document.createElement("p");

  clarificationStatus.className = "field-help";
  clarificationStatus.setAttribute("role", "status");
  clarificationStatus.setAttribute("aria-live", "polite");

  elements.clarificationForm.append(
    clarificationStatus,
  );

  const bindings = new AbortController();
  const drafts = new Map();
  const submissions = new Map();

  const previousNoValidate =
    elements.clarificationForm.noValidate;

  elements.clarificationForm.noValidate = true;

  let runId = null;
  let updatedAt = -Infinity;
  let current = null;
  let viewVersion = 0;
  let enabled = false;
  let destroyed = false;
  let operation = null;
  let errorMessage = null;
  let statusKey = null;
  let retryAt = 0;
  let cooldownTimer = null;

  const inputFor = (review) => (
    review?.kind === "planner_clarification"
      ? elements.answers
      : elements.feedback
  );

  const staleKey = (review) => (
    review?.kind === "planner_clarification"
      ? "clarification.stale"
      : "review.stale"
  );


  function saveDraft() {
    if (current) {
      remember(
        drafts,
        current.key,
        inputFor(current).value,
      );
    }
  }


  function messageForError(error) {
    const fallback = typeof error?.message === "string"
      ? error.message
      : t("errors.generic");

    const key = typeof error?.code === "string"
      ? `errors.${error.code}`
      : "errors.generic";

    let message = t(key, {}, fallback);

    if (Array.isArray(error?.issues)) {
      for (const issue of error.issues.slice(0, 4)) {
        if (typeof issue?.message === "string") {
          const field = issue.field
            ? `${issue.field}: `
            : "";

          message += `\n${field}${issue.message}`;
        }
      }
    }

    return message;
  }


  function showMessages() {
    const saved = current
      ? submissions.get(current.key)
      : null;

    let message = errorMessage?.key
      ? t(errorMessage.key)
      : errorMessage?.error
        ? messageForError(errorMessage.error)
        : "";

    if (saved?.state === "unknown") {
      message = saved.rechecked
        ? t(
            "review.retrySameDecision",
            {},
            RETRY_HINT[getLanguage()] ?? RETRY_HINT.en,
          )
        : t("review.outcomeUnknown");
    }

    const remaining = Math.max(
      0,
      retryAt - Date.now(),
    );

    if (remaining > 0) {
      const waitMessage = t(
        "common.retryAfter",
        {
          seconds: formatNumber(
            Math.ceil(remaining / 1000),
            {
              maximumFractionDigits: 0,
            },
          ),
        },
      );

      message += `${message ? "\n" : ""}${waitMessage}`;
    }

    const clarification =
      current?.kind === "planner_clarification";

    elements.reviewError.textContent =
      clarification ? "" : message;

    elements.reviewError.hidden =
      clarification || !message;

    elements.clarificationError.textContent =
      clarification ? message : "";

    elements.clarificationError.hidden =
      !clarification || !message;

    elements.reviewStatus.textContent =
      !clarification && statusKey
        ? t(statusKey)
        : "";

    clarificationStatus.textContent =
      clarification && statusKey
        ? t(statusKey)
        : "";
  }


  function updateControls() {
    const saved = current
      ? submissions.get(current.key)
      : null;

    const coolingDown = retryAt > Date.now();

    const canAct = (
      enabled
      && !destroyed
      && !operation
      && !coolingDown
      && current
      && (
        !saved
        || (
          saved.state === "unknown"
          && saved.rechecked
        )
      )
    );

    const clarification =
      current?.kind === "planner_clarification";

    elements.reviewFields.disabled =
      !canAct || clarification;

    elements.clarificationFields.disabled =
      !canAct || !clarification;

    elements.feedback.readOnly = Boolean(saved);
    elements.answers.readOnly = Boolean(saved);

    for (const [button, action, kind] of actionButtons) {
      button.disabled = (
        !canAct
        || current.kind !== kind
        || Boolean(
          saved
          && saved.decision.action !== action,
        )
      );

      const busy = Boolean(
        operation?.action === action
        && current?.kind === kind
        && current?.key === operation.key,
      );

      button.setAttribute(
        "aria-busy",
        String(busy),
      );
    }

    if (
      current?.kind === "cleaning_review"
      && current.payload.proposal.unresolved_questions.length
    ) {
      elements.approve.disabled = true;
    }

    elements.reviewForm.setAttribute(
      "aria-busy",
      String(
        Boolean(operation && !clarification),
      ),
    );

    elements.clarificationForm.setAttribute(
      "aria-busy",
      String(
        Boolean(operation && clarification),
      ),
    );

    clearTimeout(cooldownTimer);
    cooldownTimer = null;

    if (coolingDown && !destroyed) {
      cooldownTimer = setTimeout(
        () => {
          showMessages();
          updateControls();
        },
        Math.min(
          retryAt - Date.now(),
          60_000,
        ),
      );
    }
  }


  function fillQuestions(container, values) {
    const fragment = document.createDocumentFragment();

    for (const value of values) {
      const item = document.createElement("li");

      item.className = "preserve-lines";
      item.textContent = questionText(value);

      fragment.append(item);
    }

    container.replaceChildren(fragment);
  }


  function paintContent() {
    elements.reviewPanel.hidden =
      current?.kind !== "cleaning_review";

    elements.clarificationPanel.hidden =
      current?.kind !== "planner_clarification";

    if (!current) {
      return;
    }

    if (current.kind === "planner_clarification") {
      fillQuestions(
        elements.clarificationQuestions,
        current.payload.questions,
      );

      return;
    }

    const proposal = current.payload.proposal;

    elements.revision.textContent = formatNumber(
      current.revision,
      {
        maximumFractionDigits: 0,
      },
    );

    elements.summary.textContent = proposal.summary;

    const fragment = document.createDocumentFragment();

    const fields = [
      "column",
      "action",
      "feature_kind",
      "missing_strategy",
      "scaling",
      "encoding",
      "reason",
    ];

    for (const policy of proposal.policies) {
      const row = elements.policyTemplate.content.cloneNode(
        true,
      );

      for (const field of fields) {
        const cell = row.querySelector(
          `[data-field="${field}"]`,
        );

        if (!cell) {
          throw new Error(
            `Missing policy cell: ${field}`,
          );
        }

        cell.textContent = policy[field];

        if (
          field === "missing_strategy"
          && policy.fill_value != null
        ) {
          const detail = document.createElement("div");

          detail.className = "field-help";

          // Distinguish the string "0" from the numeric constant 0.
          detail.textContent = t(
            "review.fillValue",
            {
              value: JSON.stringify(
                policy.fill_value,
              ),
            },
          );

          cell.append(detail);
        }
      }

      fragment.append(row);
    }

    elements.policies.replaceChildren(fragment);

    fillQuestions(
      elements.questions,
      proposal.unresolved_questions,
    );

    elements.questionsPanel.hidden =
      proposal.unresolved_questions.length === 0;
  }


  function resetFields() {
    elements.feedback.value = "";
    elements.answers.value = "";

    for (const input of [
      elements.feedback,
      elements.answers,
    ]) {
      input.removeAttribute("aria-invalid");
    }

    if (current) {
      inputFor(current).value =
        drafts.get(current.key) ?? "";
    }
  }


  function render(record) {
    if (destroyed) {
      return false;
    }

    let parsed;

    try {
      parsed = readRecord(record);
    } catch (error) {
      clear();
      throw error;
    }

    if (
      parsed.runId === runId
      && parsed.updatedAt < updatedAt
    ) {
      return false;
    }

    saveDraft();

    const changed = (
      parsed.runId !== runId
      || parsed.review?.key !== current?.key
    );

    if (changed) {
      viewVersion += 1;
      errorMessage = null;
      statusKey = null;
    }

    runId = parsed.runId;
    updatedAt = parsed.updatedAt;
    current = parsed.review;

    if (changed) {
      resetFields();
      paintContent();
    }

    showMessages();
    updateControls();

    return true;
  }


  function clear() {
    saveDraft();

    viewVersion += 1;

    if (operation?.phase === "checking") {
      operation.controller.abort();
    }

    runId = null;
    updatedAt = -Infinity;
    current = null;
    errorMessage = null;
    statusKey = null;

    resetFields();
    paintContent();
    showMessages();
    updateControls();
  }


  function sameView(task) {
    return (
      !destroyed
      && viewVersion === task.viewVersion
      && runId === task.runId
    );
  }


  function publish(record, task) {
    if (!sameView(task)) {
      return false;
    }

    if (
      record?.run_id?.toLowerCase() !== task.runId
    ) {
      throw invalidResponse();
    }

    if (!render(record)) {
      return false;
    }

    notify(onRunUpdate, record);

    return true;
  }


  function begin(action = null) {
    const task = {
      runId,
      key: current?.key,
      viewVersion,
      action,
      phase: "checking",
      controller: new AbortController(),
    };

    operation = task;

    updateControls();
    notify(onBusyChange, true);

    return task;
  }


  function finish(task) {
    if (operation !== task) {
      return;
    }

    operation = null;

    if (!destroyed) {
      statusKey = null;

      showMessages();
      updateControls();
      notify(onBusyChange, false);
    }
  }


  async function checkLatest(task) {
    const record = await api.getRun(
      task.runId,
      {
        signal: task.controller.signal,
      },
    );

    if (!sameView(task)) {
      return null;
    }

    const parsed = readRecord(record);

    if (parsed.runId !== task.runId) {
      throw invalidResponse();
    }

    if (!publish(record, task)) {
      return null;
    }

    const saved = parsed.review
      ? submissions.get(parsed.review.key)
      : null;

    if (saved?.state === "unknown") {
      saved.rechecked = true;
    }

    return parsed;
  }


  async function refresh() {
    if (
      destroyed
      || operation
      || !runId
    ) {
      return false;
    }

    const task = begin();

    try {
      return Boolean(
        await checkLatest(task),
      );
    } catch (error) {
      if (
        sameView(task)
        && !isAbortError(error)
      ) {
        errorMessage = { error };
        throw error;
      }

      return false;
    } finally {
      finish(task);
    }
  }


  function validateDecision(action) {
    const clarification =
      current.kind === "planner_clarification";

    const allowed = clarification
      ? ["answer", "reject"]
      : ["approve", "changes", "reject"];

    if (!allowed.includes(action)) {
      return null;
    }

    const saved = submissions.get(current.key);

    if (saved) {
      const canRetry = (
        saved.state === "unknown"
        && saved.rechecked
        && action === saved.decision.action
      );

      return canRetry
        ? { ...saved.decision }
        : null;
    }

    const input = inputFor(current);

    const feedback =
      clarification && action === "reject"
        ? ""
        : input.value.trim();

    let key = null;

    if (
      action === "approve"
      && current.payload.proposal.unresolved_questions.length
    ) {
      key = "review.unresolvedHelp";
    } else if (
      action === "approve"
      && feedback
    ) {
      key = "review.approveWithFeedback";
    } else if (
      action === "answer"
      && !feedback
    ) {
      key = "validation.answersRequired";
    } else if (
      action === "changes"
      && !feedback
    ) {
      key = "validation.feedbackRequired";
    } else if (
      [...feedback].length > MAX_FEEDBACK
    ) {
      key = "validation.feedbackLength";
    }

    if (key) {
      errorMessage = { key };

      input.setAttribute(
        "aria-invalid",
        "true",
      );

      showMessages();
      input.focus();

      return null;
    }

    input.removeAttribute("aria-invalid");

    const decision = {
      revision: current.revision,
      action,
    };

    if (
      feedback
      && action !== "approve"
    ) {
      decision.feedback = feedback;
    }

    return decision;
  }


  async function submit(action) {
    if (
      destroyed
      || !enabled
      || operation
      || !current
      || retryAt > Date.now()
    ) {
      return;
    }

    const decision = validateDecision(action);

    if (!decision) {
      return;
    }

    saveDraft();

    const reviewed = current;

    const wasUnknown =
      submissions.get(reviewed.key)?.state === "unknown";

    const task = begin(action);

    statusKey = {
      approve: "review.approving",
      changes: "review.requestingChanges",
      reject: "review.rejecting",
      answer: "clarification.submitting",
    }[action];

    errorMessage = null;
    showMessages();

    let posted = false;

    try {
      const latest = await checkLatest(task);

      if (!latest) {
        return;
      }

      if (latest.review?.key !== reviewed.key) {
        if (current) {
          errorMessage = {
            key: staleKey(reviewed),
          };
        }

        return;
      }

      if (
        !sameView(task)
        || !enabled
      ) {
        return;
      }

      // Preserve the exact command before sending it.
      // The API supports retries of an identical decision.
      remember(
        submissions,
        reviewed.key,
        {
          state: "posting",
          decision: { ...decision },
          rechecked: false,
        },
      );

      task.phase = "posting";
      posted = true;

      const result = await api.submitReview(
        task.runId,
        decision,
      );

      if (
        result?.run_id?.toLowerCase() !== task.runId
      ) {
        throw invalidResponse();
      }

      readRecord(result);

      remember(
        submissions,
        reviewed.key,
        {
          state: "accepted",
          decision,
          rechecked: false,
        },
      );

      publish(result, task);
      drafts.delete(reviewed.key);
    } catch (error) {
      const unknown = (
        posted
        && (
          error.outcomeUnknown === true
          || error.code === "invalid_response"
          || !(error instanceof ApiError)
          || (
            wasUnknown
            && ![400, 422].includes(error.status)
          )
        )
      );

      if (unknown) {
        remember(
          submissions,
          reviewed.key,
          {
            state: "unknown",
            decision: { ...decision },
            rechecked: false,
          },
        );
      } else if (posted) {
        submissions.delete(reviewed.key);
      }

      if (!sameView(task)) {
        return;
      }

      if (
        Number.isFinite(error.retryAfterMs)
        && error.retryAfterMs > 0
      ) {
        retryAt = Date.now() + error.retryAfterMs;
      } else if (error.status === 429) {
        retryAt = Date.now() + 30_000;
      }

      if (!isAbortError(error)) {
        errorMessage = { error };
      }

      if (
        unknown
        || error.status === 409
      ) {
        // Only GET is automatic.
        // Another POST requires an explicit user action.
        task.phase = "checking";

        try {
          const latest = await checkLatest(task);

          if (
            latest?.review
            && latest.review.key !== reviewed.key
            && current?.key === latest.review.key
          ) {
            errorMessage = {
              key: staleKey(reviewed),
            };
          }
        } catch {
          // Keep the decision locked until an explicit refresh succeeds.
        }
      }
    } finally {
      finish(task);
    }
  }


  function bind(element, event, listener) {
    element.addEventListener(
      event,
      listener,
      {
        signal: bindings.signal,
      },
    );
  }


  bind(
    elements.clarificationForm,
    "submit",
    (event) => {
      event.preventDefault();
      void submit("answer");
    },
  );

  // Form submission must never imply approval.
  bind(
    elements.reviewForm,
    "submit",
    (event) => {
      event.preventDefault();
    },
  );

  for (const [button, action] of actionButtons) {
    if (action !== "answer") {
      bind(
        button,
        "click",
        () => {
          if (!button.disabled) {
            void submit(action);
          }
        },
      );
    }
  }

  for (const input of [
    elements.feedback,
    elements.answers,
  ]) {
    bind(
      input,
      "input",
      () => {
        saveDraft();
        input.removeAttribute("aria-invalid");

        errorMessage = null;

        showMessages();
      },
    );
  }

  const unsubscribe = onLanguageChange(() => {
    if (!destroyed) {
      paintContent();
      showMessages();
      updateControls();
    }
  });


  const controller = Object.freeze({
    render,

    refresh,

    clear,

    setEnabled(value) {
      if (typeof value !== "boolean") {
        throw new TypeError(
          "enabled must be a boolean.",
        );
      }

      if (destroyed) {
        return;
      }

      enabled = value;

      updateControls();
    },

    get busy() {
      return operation !== null;
    },

    get runId() {
      return runId;
    },

    destroy() {
      if (destroyed) {
        return;
      }

      clear();

      destroyed = true;
      enabled = false;

      clearTimeout(cooldownTimer);

      bindings.abort();
      unsubscribe();

      clarificationStatus.remove();

      elements.clarificationForm.noValidate =
        previousNoValidate;

      drafts.clear();
      submissions.clear();

      controllers.delete(
        elements.reviewForm,
      );

      updateControls();
    },
  });

  controllers.set(
    elements.reviewForm,
    controller,
  );

  paintContent();
  showMessages();
  updateControls();

  return controller;
}