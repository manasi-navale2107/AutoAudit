/**
 * AutoAudit browser entry point.
 * All HTTP requests go through api.js.
 */

import {
  api,
  ApiError,
  createRequestId,
  isAbortError,
  MAX_CSV_BYTES,
} from "./api.js";

import {
  initI18n,
  setLanguage,
  getLanguage,
  t,
  formatNumber,
  formatDateTime,
  onLanguageChange,
} from "./i18n.js";

import { createReviewController } from "./review.js";
import { createVoiceController } from "./voice.js";

const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const ACTIVE = new Set(["queued", "running"]);

const WAITING = new Set([
  "awaiting_review",
  "awaiting_clarification",
]);

const RETRYABLE = new Set(["failed", "interrupted"]);

const EXTRA = {
  checkRequest: {
    en: "Check request status",
    mr: "Request ची स्थिती तपासा",
    hi: "Request की स्थिति देखें",
  },
  resendRequest: {
    en: "Retry the same request",
    mr: "तोच request पुन्हा पाठवा",
    hi: "वही request फिर भेजें",
  },
};

const object = (value) =>
  value !== null &&
  typeof value === "object" &&
  !Array.isArray(value);

const invalid = () =>
  new ApiError("The API response has an invalid structure.", {
    code: "invalid_response",
  });

const text = (value) =>
  value == null
    ? "—"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);

const pretty = (value) => JSON.stringify(value, null, 2);

const number = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? formatNumber(value)
    : "—";

const statusText = (value) =>
  t(
    `status.${value}`,
    {},
    t("status.unknown", { status: text(value) }),
  );

const extra = (key) =>
  t(
    `run.${key}`,
    {},
    EXTRA[key][getLanguage()] ?? EXTRA[key].en,
  );

function validateSummary(value) {
  if (
    !object(value) ||
    !UUID.test(value.run_id ?? "") ||
    typeof value.status !== "string" ||
    typeof value.updated_at !== "string" ||
    !Number.isFinite(Date.parse(value.updated_at))
  ) {
    throw invalid();
  }

  return value;
}

function validateRun(value, expectedId) {
  validateSummary(value);

  if (
    value.run_id.toLowerCase() !== expectedId ||
    !object(value.state) ||
    !Array.isArray(value.interrupts)
  ) {
    throw invalid();
  }

  for (const key of ["training_report", "artifact"]) {
    const item = value.state[key];

    if (
      item != null &&
      (
        !object(item) ||
        item.run_id !== value.run_id ||
        item.test_evaluated !== false ||
        item.audit_status !== "not_performed"
      )
    ) {
      throw invalid();
    }
  }

  const report = value.state.training_report;
  const artifact = value.state.artifact;

  if (report && !Array.isArray(report.candidates)) {
    throw invalid();
  }

  if (artifact && !object(artifact.files)) {
    throw invalid();
  }

  if (
    report?.candidates.some(
      (item) =>
        !object(item) ||
        typeof item.model !== "string" ||
        typeof item.eligible_for_selection !== "boolean" ||
        typeof item.validation_score !== "number" ||
        !Number.isFinite(item.validation_score) ||
        !Array.isArray(item.warning_types),
    )
  ) {
    throw invalid();
  }

  if (
    value.status === "ready_for_audit" &&
    (!report || !artifact)
  ) {
    throw invalid();
  }

  return value;
}

function validateDataset(value, expectedId = null) {
  if (
    !object(value) ||
    !UUID.test(value.dataset_id ?? "") ||
    typeof value.filename !== "string" ||
    !Array.isArray(value.columns) ||
    !value.columns.length ||
    value.columns.some(
      (column) =>
        typeof column !== "string" ||
        !column.trim(),
    ) ||
    new Set(value.columns).size !== value.columns.length ||
    !Number.isSafeInteger(value.row_count) ||
    value.row_count < 1 ||
    value.column_count !== value.columns.length ||
    !Number.isSafeInteger(value.size_bytes) ||
    value.size_bytes < 1 ||
    (
      expectedId &&
      value.dataset_id.toLowerCase() !== expectedId
    )
  ) {
    throw invalid();
  }

  return value;
}

function errorText(error) {
  let result = t(
    `errors.${error?.code ?? "generic"}`,
    {},
    error?.message || t("errors.generic"),
  );

  const issues = Array.isArray(error?.issues)
    ? error.issues
    : [];

  for (const issue of issues.slice(0, 4)) {
    if (typeof issue?.message === "string") {
      result += `\n${issue.field || ""}: ${issue.message}`;
    }
  }

  return result;
}

export function createApp() {
  const $ = (id) => {
    const element = document.getElementById(id);

    if (!element) {
      throw new Error(`Missing application element: #${id}`);
    }

    return element;
  };

  const bindings = new AbortController();

  const reads = new Map();
  const messages = new Map();
  const datasets = new Map();
  const runs = new Map();
  const uncertainRetries = new Set();
  const downloads = new Map();

  const state = {
    alive: true,
    started: false,
    ready: false,
    connecting: false,
    connection: "unknown",

    datasetId: "",
    dataset: null,
    profile: null,

    activeId: null,
    record: null,
    selection: 0,

    uploading: false,
    creating: false,
    pending: null,
    checkingCreate: false,
    retrying: false,

    retryAt: 0,
    pollAt: 0,
    pollFailures: 0,
    missingRun: false,
  };

  let pollTimer = null;
  let cooldownTimer = null;

  let review = null;
  let voice = null;

  let unsubscribe = () => {};

  // Recovery controls for a create request whose response was lost.
  const recovery = document.createElement("div");
  recovery.className = "action-row";
  recovery.hidden = true;

  const checkCreate = document.createElement("button");
  const resendCreate = document.createElement("button");

  checkCreate.id = "check-create-button";
  resendCreate.id = "resend-create-button";

  for (const button of [checkCreate, resendCreate]) {
    button.type = "button";
    button.className = "button button--secondary";
    recovery.append(button);
  }

  $("run-form").append(recovery);

  const originalValidation = [
    $("upload-form").noValidate,
    $("run-form").noValidate,
  ];

  $("upload-form").noValidate = true;
  $("run-form").noValidate = true;

  function paintMessage(id) {
    const item = messages.get(id);

    const value = item?.error
      ? errorText(item.error)
      : item
        ? t(item.key, item.params)
        : "";

    const element = $(id);

    element.removeAttribute("data-i18n");

    if (element.textContent !== value) {
      element.textContent = value;
    }

    element.hidden = !value;

    if (id === "app-error-message") {
      $("app-error").hidden = !value;
    }
  }

  function message(id, value = null, params = {}) {
    if (!state.alive) return;

    if (value === null) {
      messages.delete(id);
    } else {
      messages.set(
        id,
        typeof value === "string"
          ? { key: value, params }
          : { error: value },
      );
    }

    paintMessage(id);
  }

  function appError(error) {
    if (!isAbortError(error)) {
      message("app-error-message", error);
    }
  }

  function bind(target, event, handler) {
    target.addEventListener(
      event,
      (eventObject) => {
        try {
          Promise.resolve(handler(eventObject)).catch(appError);
        } catch (error) {
          appError(error);
        }
      },
      { signal: bindings.signal },
    );
  }

  function cancelRead(name) {
    reads.get(name)?.abort();
    reads.delete(name);
  }

  async function read(
    name,
    operation,
    accept,
    reject = appError,
  ) {
    cancelRead(name);

    const controller = new AbortController();
    reads.set(name, controller);

    syncControls();

    try {
      const value = await operation(controller.signal);

      if (
        !state.alive ||
        reads.get(name) !== controller
      ) {
        return false;
      }

      accept(value);
      return true;
    } catch (error) {
      if (
        state.alive &&
        reads.get(name) === controller &&
        !isAbortError(error)
      ) {
        reject(error);
      }

      return false;
    } finally {
      if (reads.get(name) === controller) {
        reads.delete(name);
      }

      if (state.alive) {
        syncControls();
        schedulePoll();
      }
    }
  }

  function cooldown(error) {
    const delay = Number.isFinite(error?.retryAfterMs)
      ? Math.max(0, error.retryAfterMs)
      : error?.status === 429
        ? 30_000
        : 0;

    state.retryAt = Math.max(
      state.retryAt,
      Date.now() + delay,
    );

    clearTimeout(cooldownTimer);

    function update() {
      if (!state.alive) return;

      syncControls();

      if (state.retryAt > Date.now()) {
        cooldownTimer = setTimeout(
          update,
          Math.min(state.retryAt - Date.now(), 60_000),
        );
      }
    }

    update();

    if (delay > 0) {
      message("app-message", "common.retryAfter", {
        seconds: Math.ceil(delay / 1000),
      });
    }
  }

  function syncControls() {
    if (!state.alive) return;

    const cooling = Date.now() < state.retryAt;

    const formBusy =
      state.uploading ||
      state.creating ||
      state.pending !== null;

    const canEdit = state.ready && !formBusy;

    const canRun =
      canEdit &&
      state.dataset !== null &&
      !reads.has("dataset");

    $("interface-language").disabled = !state.started;
    $("new-run-button").disabled = !state.started;

    $("refresh-runs-button").disabled =
      !state.ready || reads.has("history");

    $("retry-connection-button").disabled =
      state.connecting;

    $("upload-fields").disabled =
      !canEdit || cooling;

    $("dataset-select").disabled =
      !canEdit || reads.has("datasets");

    $("run-fields").disabled = !canRun;

    $("objective").readOnly =
      Boolean(state.pending);

    $("start-run-button").disabled =
      !canRun ||
      cooling ||
      Boolean(voice?.busy);

    $("start-run-button").setAttribute(
      "aria-busy",
      String(state.creating),
    );

    $("upload-button").setAttribute(
      "aria-busy",
      String(state.uploading),
    );

    $("upload-progress").hidden = !state.uploading;

    $("refresh-run-button").disabled =
      !state.ready ||
      !state.activeId ||
      reads.has("run") ||
      Boolean(review?.busy) ||
      state.retrying;

    const retryable = RETRYABLE.has(
      state.record?.status,
    );

    $("retry-run-button").hidden = !retryable;

    $("retry-run-button").disabled =
      !state.ready ||
      !retryable ||
      state.retrying ||
      Boolean(review?.busy) ||
      reads.has("run") ||
      uncertainRetries.has(state.activeId) ||
      cooling;

    $("retry-run-button").setAttribute(
      "aria-busy",
      String(state.retrying),
    );

    $("download-report-button").disabled =
      !state.record?.state?.training_report;

    $("download-manifest-button").disabled =
      !state.record?.state?.artifact;

    recovery.hidden =
      !state.pending || state.creating;

    checkCreate.textContent = extra("checkRequest");
    resendCreate.textContent = extra("resendRequest");

    checkCreate.disabled =
      !state.ready ||
      state.checkingCreate ||
      cooling;

    resendCreate.hidden =
      state.pending?.mode !== "missing";

    resendCreate.disabled =
      !state.ready ||
      state.checkingCreate ||
      cooling;

    review?.setEnabled(
      state.ready && !state.retrying,
    );

    // Keep the fieldset enabled while speaking,
    // so the Stop dictation button remains usable.
    voice?.setEnabled(
      canRun && !state.activeId,
    );

    const badge = $("connection-status");

    const key =
      state.connection === "unknown"
        ? "connection.unchecked"
        : `connection.${state.connection}`;

    badge.dataset.state = state.connection;
    badge.setAttribute("data-i18n", key);
    badge.textContent = t(key);

    document.body.dataset.appState =
      state.ready
        ? "ready"
        : state.connecting
          ? "booting"
          : "offline";
  }

  function definitions(id, entries) {
    const fragment = document.createDocumentFragment();

    for (const [label, value] of entries) {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");

      term.textContent = t(label);
      detail.textContent = text(value);
      detail.className = "preserve-lines";

      item.append(term, detail);
      fragment.append(item);
    }

    $(id).replaceChildren(fragment);
  }

  function paintDatasetOptions() {
    const fragment = document.createDocumentFragment();
    const placeholder = document.createElement("option");

    placeholder.value = "";
    placeholder.textContent = t(
      datasets.size
        ? "dataset.chooseSaved"
        : "dataset.noneSaved",
    );

    fragment.append(placeholder);

    for (const item of datasets.values()) {
      const option = document.createElement("option");

      option.value = item.dataset_id.toLowerCase();

      option.textContent =
        t("dataset.optionLabel", {
          filename: item.filename,
          rows: number(item.row_count),
        }) +
        ` · ${option.value.slice(0, 8)}`;

      fragment.append(option);
    }

    $("dataset-select").replaceChildren(fragment);
    $("dataset-select").value = state.datasetId;
  }

  function paintDataset() {
    const item = state.dataset;

    $("dataset-summary").hidden = !item;

    $("dataset-filename").textContent =
      item?.filename ?? "";

    $("dataset-row-count").textContent =
      number(item?.row_count);

    $("dataset-column-count").textContent =
      number(item?.column_count);

    $("dataset-size").textContent = item
      ? `${number(item.size_bytes / (1024 * 1024))} MiB`
      : "—";

    $("dataset-profile-json").textContent =
      state.profile ? pretty(state.profile) : "";

    const target = $("target-column");
    const previous = target.value;

    const fragment = document.createDocumentFragment();
    const infer = document.createElement("option");

    infer.value = "";
    infer.textContent = t("run.inferTarget");

    fragment.append(infer);

    for (const column of item?.columns ?? []) {
      const option = document.createElement("option");

      // Preserve exact CSV column names, including spaces.
      option.value = column;
      option.textContent = column;

      fragment.append(option);
    }

    target.replaceChildren(fragment);

    target.value =
      item?.columns.includes(previous)
        ? previous
        : "";
  }

  function mergeRuns(items) {
    items.forEach(validateSummary);

    for (const item of items) {
      const id = item.run_id.toLowerCase();
      const existing = runs.get(id);

      if (
        !existing ||
        Date.parse(item.updated_at) >=
          Date.parse(existing.updated_at)
      ) {
        runs.set(id, {
          run_id: id,
          status: item.status,
          updated_at: item.updated_at,
        });
      }
    }

    const sorted = [...runs.values()].sort(
      (a, b) =>
        Date.parse(b.updated_at) -
        Date.parse(a.updated_at),
    );

    runs.clear();

    sorted.slice(0, 100).forEach((item) => {
      runs.set(item.run_id, item);
    });

    paintHistory();
  }

  function paintHistory() {
    const focusedId =
      $("run-list").contains(document.activeElement)
        ? document.activeElement?.dataset?.runId
        : null;

    const fragment = document.createDocumentFragment();

    for (const item of runs.values()) {
      const row =
        $("run-item-template").content.cloneNode(true);

      const button = row.querySelector("button");

      button.dataset.runId = item.run_id;

      button.setAttribute(
        "aria-label",
        t("history.openRun", { id: item.run_id }),
      );

      if (item.run_id === state.activeId) {
        button.setAttribute("aria-current", "true");
      }

      row.querySelector(
        '[data-field="title"]',
      ).textContent = t("history.runLabel", {
        id: item.run_id.slice(0, 8),
      });

      row.querySelector(
        '[data-field="status"]',
      ).textContent = statusText(item.status);

      const time = row.querySelector(
        '[data-field="updated_at"]',
      );

      time.textContent = formatDateTime(item.updated_at);
      time.dateTime = item.updated_at;

      fragment.append(row);
    }

    $("run-list").replaceChildren(fragment);

    $("run-list-empty").hidden =
      runs.size !== 0 || reads.has("history");

    if (focusedId) {
      const button = [
        ...$("run-list").querySelectorAll("button"),
      ].find(
        (item) => item.dataset.runId === focusedId,
      );

      button?.focus({ preventScroll: true });
    }
  }

  function paintRun() {
    const record = state.record;
    const current = record?.state ?? {};

    $("new-run-panel").hidden =
      Boolean(state.activeId);

    $("run-panel").hidden = !state.activeId;

    $("active-run-id").textContent =
      state.activeId ?? "";

    $("active-run-objective").textContent =
      current.request?.objective ?? "";

    $("run-status").textContent = record
      ? statusText(record.status)
      : t("common.loading");

    $("run-status").dataset.status =
      record?.status ?? "unknown";

    $("run-progress").hidden =
      !record || !ACTIVE.has(record.status);

    $("run-updated-at").textContent = record
      ? formatDateTime(record.updated_at)
      : "—";

    if (record) {
      $("run-updated-at").dateTime = record.updated_at;
    } else {
      $("run-updated-at").removeAttribute("datetime");
    }

    definitions("active-run-settings", [
      ["settings.provider", current.request?.provider],
      ["settings.model", current.request?.model],
      [
        "settings.responseLanguage",
        current.request?.language,
      ],
      ["dataset.id", current.request?.dataset_id],
    ]);

    const plan = current.execution_plan;

    $("plan-panel").hidden = !object(plan);
    $("plan-json").textContent = plan
      ? pretty(plan)
      : "";

    if (object(plan)) {
      const split = plan.split ?? {};

      const percent = (value) =>
        typeof value === "number"
          ? formatNumber(value, {
              style: "percent",
              maximumFractionDigits: 2,
            })
          : "—";

      definitions("plan-summary", [
        ["plan.target", plan.target_column],
        ["plan.metric", plan.primary_metric],
        ["plan.split", split.strategy],
        [
          "plan.trainFraction",
          percent(split.train_fraction),
        ],
        [
          "plan.validationFraction",
          percent(split.validation_fraction),
        ],
        [
          "plan.testFraction",
          percent(split.test_fraction),
        ],
        [
          "plan.candidateModels",
          plan.candidate_models?.join(", "),
        ],
        ["plan.rationale", split.rationale],
      ]);
    }

    const report = current.training_report;

    let stage = plan ? 1 : 0;

    if (
      current.training_decision ||
      [
        "approved",
        "training",
        "trained",
        "ready_for_audit",
      ].includes(current.status)
    ) {
      stage = 2;
    }

    if (record?.status === "awaiting_clarification") {
      stage = 0;
    }

    if (record?.status === "awaiting_review") {
      stage = 1;
    }

    for (const [index, name] of [
      "planner",
      "engineering",
      "training",
    ].entries()) {
      let value =
        index < stage || report
          ? "completed"
          : "waiting";

      if (index === stage && !report && record) {
        value =
          record.status === "queued"
            ? "waiting"
            : record.status;
      }

      $(`${name}-step`).dataset.status = value;

      $(`${name}-status`).setAttribute(
        "data-i18n",
        `status.${value}`,
      );

      $(`${name}-status`).textContent =
        statusText(value);
    }

    const helpKey = {
      failed: "run.failedHelp",
      interrupted: "run.interruptedHelp",
      needs_review: "run.needsReviewHelp",
      rejected: "run.rejectedHelp",
    }[record?.status];

    $("run-error-panel").hidden = !helpKey;

    const details =
      record?.error ??
      (
        record?.status === "needs_review"
          ? current.planner_decision
          : null
      );

    $("run-error-message").textContent = helpKey
      ? [
          t(helpKey),
          details == null ? "" : text(details),
          uncertainRetries.has(state.activeId)
            ? t("run.retryUnknown")
            : "",
        ]
          .filter(Boolean)
          .join("\n")
      : "";

    $("training-results-panel").hidden = !report;

    $("training-report-json").textContent = report
      ? pretty(report)
      : "";

    if (!report) return;

    definitions("training-summary", [
      ["training.selectedModel", report.selected_model],
      ["training.primaryMetric", report.primary_metric],
      [
        "training.selectedScore",
        number(report.selected_validation_score),
      ],
      [
        "training.approvedRevision",
        number(report.approved_revision),
      ],
      [
        "training.trainRows",
        number(report.split_counts?.train),
      ],
      [
        "training.validationRows",
        number(report.split_counts?.validation),
      ],
      [
        "training.testRows",
        number(report.split_counts?.test),
      ],
      ["training.testEvaluated", t("common.no")],
      ["training.auditStatus", t("training.notPerformed")],
      ["training.selectionRule", report.selection_rule],
    ]);

    const fragment = document.createDocumentFragment();

    for (const candidate of report.candidates) {
      const row =
        $("model-row-template").content.cloneNode(true);

      const selected =
        candidate.model === report.selected_model;

      row.querySelector("tr").dataset.selected =
        String(selected);

      row.querySelector(
        '[data-field="model"]',
      ).textContent = text(candidate.model);

      row.querySelector(
        '[data-field="status"]',
      ).textContent = t(
        selected
          ? "training.selected"
          : candidate.eligible_for_selection === true
            ? "training.eligible"
            : "training.ineligible",
      );

      row.querySelector(
        '[data-field="primary_score"]',
      ).textContent = number(candidate.validation_score);

      const notes = [
        t("training.elapsed", {
          seconds: number(candidate.elapsed_seconds),
        }),
      ];

      if (candidate.warning_types?.length) {
        notes.push(
          t("training.warnings", {
            warnings: candidate.warning_types.join(", "),
          }),
        );
      }

      row.querySelector(
        '[data-field="notes"]',
      ).textContent = notes.join("\n");

      fragment.append(row);
    }

    $("model-table-body").replaceChildren(fragment);

    const manifest = current.artifact;

    definitions(
      "artifact-summary",
      manifest
        ? [
            ["artifact.id", manifest.artifact_id],
            [
              "artifact.status",
              statusText(manifest.status),
            ],
            [
              "artifact.created",
              formatDateTime(manifest.created_at),
            ],
            [
              "artifact.fileCount",
              number(Object.keys(manifest.files).length),
            ],
            [
              "artifact.files",
              Object.keys(manifest.files).join("\n"),
            ],
          ]
        : [
            [
              "artifact.status",
              t("artifact.manifestUnavailable"),
            ],
          ],
    );
  }

  function acceptRun(value, reviewed = false) {
    if (!state.activeId) return false;

    validateRun(value, state.activeId);

    if (
      state.record &&
      Date.parse(value.updated_at) <
        Date.parse(state.record.updated_at)
    ) {
      return false;
    }

    if (!reviewed && !review.render(value)) {
      return false;
    }

    state.record = JSON.parse(JSON.stringify(value));
    state.pollFailures = 0;
    state.missingRun = false;

    mergeRuns([value]);
    paintRun();
    syncControls();

    return true;
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    pollTimer = null;

    if (
      !state.alive ||
      !state.ready ||
      document.hidden ||
      !state.activeId ||
      state.missingRun ||
      reads.has("run") ||
      review?.busy ||
      state.retrying
    ) {
      return;
    }

    if (
      state.record &&
      !ACTIVE.has(state.record.status) &&
      !WAITING.has(state.record.status)
    ) {
      return;
    }

    const base = WAITING.has(state.record?.status)
      ? 10_000
      : 2_500;

    const delay = Math.max(
      state.pollAt - Date.now(),
      Math.min(
        30_000,
        base * (2 ** Math.min(state.pollFailures, 4)),
      ),
    );

    pollTimer = setTimeout(() => {
      if (Date.now() < state.pollAt) {
        schedulePoll();
      } else {
        void loadRun();
      }
    }, Math.min(60_000, delay));
  }

  function loadRun() {
    if (
      !state.alive ||
      !state.ready ||
      !state.activeId ||
      state.retrying ||
      (
        review?.busy &&
        review.runId === state.activeId
      )
    ) {
      return Promise.resolve(false);
    }

    const id = state.activeId;
    const selection = state.selection;

    return read(
      "run",
      (signal) => api.getRun(id, { signal }),
      (value) => {
        if (selection !== state.selection) return;

        acceptRun(value);
        message("app-message");
      },
      (error) => {
        state.pollFailures += 1;
        state.missingRun = error.status === 404;

        const delay = Number.isFinite(error.retryAfterMs)
          ? Math.max(0, error.retryAfterMs)
          : error.status === 429
            ? 30_000
            : 0;

        state.pollAt = Date.now() + delay;

        message("app-message", "run.statusStale");
        appError(error);
      },
    );
  }

  function selectRun(id, record = null) {
    if (!UUID.test(id)) throw invalid();

    voice.clear();

    cancelRead("run");
    clearTimeout(pollTimer);

    state.selection += 1;
    state.activeId = id.toLowerCase();
    state.record = null;
    state.missingRun = false;

    review.clear();

    message(
      "app-message",
      record ? "run.created" : "run.loading",
    );

    paintRun();
    paintHistory();
    syncControls();

    $("run-panel").focus({ preventScroll: true });

    if (record) {
      acceptRun(record);
      schedulePoll();
    } else {
      void loadRun();
    }
  }

  function newRun() {
    voice.clear();

    cancelRead("run");
    clearTimeout(pollTimer);

    state.selection += 1;
    state.activeId = null;
    state.record = null;

    review.clear();

    // Preserve an uncertain create request until it is resolved.
    if (!state.pending) {
      $("run-form").reset();

      message("run-form-error");
      $("objective").removeAttribute("aria-invalid");
    }

    message("app-message");

    paintRun();
    paintHistory();
    paintDataset();
    countObjective();
    syncControls();
  }

  async function refreshRun() {
    if (
      !state.ready ||
      !state.activeId ||
      state.retrying ||
      review.busy
    ) {
      return;
    }

    const id = state.activeId;
    const selection = state.selection;

    cancelRead("run");

    try {
      // review.refresh() also checks uncertain review submissions.
      const ok =
        review.runId === id
          ? await review.refresh()
          : await loadRun();

      if (ok && selection === state.selection) {
        uncertainRetries.delete(id);

        message("app-message");
        message("app-error-message");

        paintRun();
      }
    } catch (error) {
      appError(error);
    } finally {
      syncControls();
      schedulePoll();
    }
  }

  function loadHistory() {
    message("run-list-status", "history.loading");

    return read(
      "history",
      (signal) =>
        api.listRuns({
          limit: 100,
          signal,
        }),
      (items) => {
        mergeRuns(items);

        message("run-list-status", "history.loaded", {
          count: number(runs.size),
        });

        $("run-list-empty").hidden = runs.size !== 0;
      },
      (error) => {
        message("run-list-status", "history.loadFailed");
        appError(error);
      },
    );
  }

  function loadDatasets() {
    return read(
      "datasets",
      (signal) =>
        api.listDatasets({
          limit: 100,
          signal,
        }),
      (items) => {
        items.forEach((item) => validateDataset(item));

        for (const item of items) {
          datasets.set(
            item.dataset_id.toLowerCase(),
            item,
          );
        }

        paintDatasetOptions();
      },
      (error) => {
        message("upload-error", "dataset.listFailed");
        appError(error);
      },
    );
  }

  function chooseDataset(id) {
    voice.clear();

    cancelRead("dataset");
    cancelRead("profile");

    state.datasetId = id;
    state.dataset = null;
    state.profile = null;

    $("dataset-profile-panel").open = false;
    $("target-column").value = "";

    message("dataset-profile-status");
    message("upload-error");

    paintDataset();
    syncControls();

    if (!id) {
      message("upload-status");
      return Promise.resolve(false);
    }

    message("upload-status", "dataset.loading");

    return read(
      "dataset",
      (signal) => api.getDataset(id, { signal }),
      (value) => {
        state.dataset = validateDataset(value, id);

        datasets.set(id, value);

        paintDatasetOptions();
        paintDataset();

        message("upload-status", "dataset.selected", {
          filename: value.filename,
        });
      },
      (error) => {
        message("upload-status");
        message("upload-error", error);
      },
    );
  }

  function loadProfile() {
    if (
      !state.dataset ||
      state.profile ||
      reads.has("profile") ||
      !$("dataset-profile-panel").open
    ) {
      return;
    }

    const id = state.datasetId;

    message(
      "dataset-profile-status",
      "dataset.profileLoading",
    );

    void read(
      "profile",
      (signal) =>
        api.getDatasetProfile(id, { signal }),
      (value) => {
        state.profile = value;

        $("dataset-profile-json").textContent =
          pretty(value);

        message(
          "dataset-profile-status",
          "dataset.profileLoaded",
        );
      },
      (error) => {
        message(
          "dataset-profile-status",
          "dataset.profileFailed",
        );

        appError(error);
      },
    );
  }

  async function upload(event) {
    event.preventDefault();

    if (
      !state.ready ||
      state.uploading ||
      state.pending ||
      state.creating ||
      Date.now() < state.retryAt
    ) {
      return;
    }

    const file = $("dataset-file").files?.[0];

    const key = !file
      ? "validation.fileRequired"
      : !file.name.toLowerCase().endsWith(".csv")
        ? "validation.csvRequired"
        : file.size === 0
          ? "validation.emptyFile"
          : file.size > MAX_CSV_BYTES
            ? "validation.fileTooLarge"
            : null;

    if (key) {
      message("upload-error", key);
      $("dataset-file").focus();
      return;
    }

    state.uploading = true;

    voice.cancel();
    cancelRead("dataset");

    message("upload-error");
    message("upload-status", "dataset.uploading");

    syncControls();

    try {
      const value = validateDataset(
        await api.uploadDataset(file),
      );

      if (!state.alive) return;

      cancelRead("datasets");

      datasets.set(
        value.dataset_id.toLowerCase(),
        value,
      );

      $("dataset-file").value = "";

      await chooseDataset(
        value.dataset_id.toLowerCase(),
      );
    } catch (error) {
      if (!state.alive) return;

      cooldown(error);
      message("upload-status");

      const unknown =
        error.outcomeUnknown ||
        error.code === "invalid_response";

      message(
        "upload-error",
        unknown ? "dataset.uploadUnknown" : error,
      );

      if (unknown) {
        await loadDatasets();
      }
    } finally {
      state.uploading = false;

      if (state.alive) {
        syncControls();
      }
    }
  }

  function countObjective() {
    $("objective-count").textContent = formatNumber(
      $("objective").value.length,
      { maximumFractionDigits: 0 },
    );
  }

  function payload() {
    const objective = $("objective").value.trim();

    let key = !state.dataset
      ? "validation.datasetRequired"
      : [...objective].length < 5 ||
          $("objective").value.length > 5000
        ? "validation.objectiveLength"
        : null;

    const target = $("target-column").value;
    const provider = $("llm-provider").value;
    const language = $("response-language").value;

    if (
      !key &&
      target &&
      !state.dataset.columns.includes(target)
    ) {
      key = "validation.targetColumn";
    }

    if (
      !key &&
      !["", "groq", "ollama"].includes(provider)
    ) {
      key = "validation.provider";
    }

    if (
      !key &&
      !["", "en", "mr", "hi"].includes(language)
    ) {
      key = "validation.language";
    }

    if (key) {
      message("run-form-error", key);

      if (key === "validation.objectiveLength") {
        $("objective").setAttribute(
          "aria-invalid",
          "true",
        );

        $("objective").focus();
      }

      return null;
    }

    return Object.freeze({
      request_id: createRequestId(),
      dataset_id: state.datasetId,
      objective,

      ...(target ? { target_column: target } : {}),
      ...(provider ? { provider } : {}),
      ...(language ? { language } : {}),
    });
  }

  async function postCreate() {
    if (
      !state.ready ||
      state.creating ||
      !state.pending ||
      Date.now() < state.retryAt
    ) {
      return;
    }

    const pending = state.pending;
    const selection = state.selection;

    state.creating = true;
    pending.mode = "posting";

    message("run-form-error");
    message("app-message", "run.creating");

    syncControls();

    try {
      const value = validateRun(
        await api.createRun(pending.payload),
        pending.payload.request_id,
      );

      if (!state.alive) return;

      state.pending = null;

      mergeRuns([value]);
      message("run-form-error");

      if (selection === state.selection) {
        selectRun(value.run_id, value);
      }
    } catch (error) {
      if (!state.alive) return;

      cooldown(error);

      if (
        error.outcomeUnknown ||
        error.code === "invalid_response" ||
        !(error instanceof ApiError)
      ) {
        pending.mode = "unknown";

        message("run-form-error", "run.creationUnknown", {
          id: pending.payload.request_id,
        });
      } else {
        state.pending = null;
        message("run-form-error", error);
      }

      if (
        selection === state.selection &&
        Date.now() >= state.retryAt
      ) {
        message("app-message");
      }
    } finally {
      state.creating = false;

      if (state.alive) {
        syncControls();
      }
    }
  }

  async function submitRun(event) {
    event.preventDefault();

    if (
      !state.ready ||
      state.pending ||
      state.creating ||
      state.uploading ||
      reads.has("dataset") ||
      voice.busy ||
      Date.now() < state.retryAt
    ) {
      return;
    }

    const request = payload();

    if (!request) return;

    voice.clear();

    state.pending = {
      payload: request,
      mode: "posting",
    };

    await postCreate();
  }

  async function checkPending() {
    if (
      !state.pending ||
      state.creating ||
      state.checkingCreate ||
      !state.ready ||
      Date.now() < state.retryAt
    ) {
      return;
    }

    const pending = state.pending;
    const selection = state.selection;

    state.checkingCreate = true;
    pending.mode = "unknown";

    syncControls();

    try {
      const value = validateRun(
        await api.getRun(pending.payload.request_id),
        pending.payload.request_id,
      );

      if (
        !state.alive ||
        state.pending !== pending
      ) {
        return;
      }

      state.pending = null;

      message("run-form-error");
      mergeRuns([value]);

      if (selection === state.selection) {
        selectRun(value.run_id, value);
      }
    } catch (error) {
      if (!state.alive) return;

      cooldown(error);

      if (error.status === 404) {
        // A user may explicitly resend the same immutable payload.
        pending.mode = "missing";
      } else {
        appError(error);
      }

      message("run-form-error", "run.creationUnknown", {
        id: pending.payload.request_id,
      });
    } finally {
      state.checkingCreate = false;

      if (state.alive) {
        syncControls();
      }
    }
  }

  async function retryRun() {
    if (
      !state.ready ||
      !state.record ||
      !RETRYABLE.has(state.record.status) ||
      state.retrying ||
      review.busy ||
      uncertainRetries.has(state.activeId) ||
      Date.now() < state.retryAt
    ) {
      return;
    }

    const id = state.activeId;
    const selection = state.selection;
    const previous = pretty(state.record);

    state.retrying = true;

    cancelRead("run");
    clearTimeout(pollTimer);

    syncControls();

    let posted = false;

    try {
      const latest = validateRun(
        await api.getRun(id),
        id,
      );

      if (
        !state.alive ||
        selection !== state.selection
      ) {
        return;
      }

      acceptRun(latest);

      // Recheck the exact displayed state before retrying.
      if (
        !RETRYABLE.has(latest.status) ||
        pretty(latest) !== previous
      ) {
        message("app-message", "run.statusStale");
        return;
      }

      posted = true;

      message("app-message", "run.retrying");

      const value = validateRun(
        await api.retryRun(id),
        id,
      );

      if (!state.alive) return;

      mergeRuns([value]);

      if (selection === state.selection) {
        acceptRun(value);
        message("app-message", "run.retryAccepted");
      }
    } catch (error) {
      if (!state.alive) return;

      cooldown(error);

      if (
        posted &&
        (
          error.outcomeUnknown ||
          error.code === "invalid_response" ||
          !(error instanceof ApiError)
        )
      ) {
        uncertainRetries.add(id);

        if (selection === state.selection) {
          message("app-message", "run.retryUnknown");
          paintRun();
        }
      } else if (selection === state.selection) {
        appError(error);
      }
    } finally {
      state.retrying = false;

      if (state.alive) {
        syncControls();
        schedulePoll();
      }
    }
  }

  function download(kind) {
    const value =
      state.record?.state?.[
        kind === "report"
          ? "training_report"
          : "artifact"
      ];

    if (!value || !state.activeId) return;

    let url;
    const anchor = document.createElement("a");

    try {
      url = URL.createObjectURL(
        new Blob(
          [pretty(value) + "\n"],
          { type: "application/json;charset=utf-8" },
        ),
      );

      anchor.href = url;
      anchor.download =
        `autoaudit-${state.activeId}-${kind}.json`;

      anchor.hidden = true;

      document.body.append(anchor);
      anchor.click();

      downloads.set(
        url,
        setTimeout(() => {
          URL.revokeObjectURL(url);
          downloads.delete(url);
        }, 60_000),
      );
    } catch {
      if (url) {
        URL.revokeObjectURL(url);
      }

      message(
        "app-error-message",
        "errors.download_failed",
      );
    } finally {
      anchor.remove();
    }
  }

  async function connect() {
    if (!state.alive || state.connecting) return;

    state.connecting = true;
    state.connection = "checking";

    message("app-error-message");
    syncControls();

    try {
      const result = await api.getReadiness();

      if (!state.alive) return;

      if (result.status !== "ready") {
        throw new ApiError("Services are not ready.", {
          code: "not_ready",
        });
      }

      state.ready = true;
      state.connection = "ready";

      syncControls();

      await Promise.allSettled([
        loadHistory(),
        loadDatasets(),
        state.activeId
          ? refreshRun()
          : Promise.resolve(),
      ]);
    } catch (error) {
      if (!state.alive) return;

      state.ready = false;

      state.connection =
        error.code === "not_ready"
          ? "not_ready"
          : "offline";

      appError(error);
    } finally {
      state.connecting = false;

      if (state.alive) {
        syncControls();
        schedulePoll();
      }
    }
  }

  function repaint() {
    if (!state.alive) return;

    messages.forEach((_, id) => {
      paintMessage(id);
    });

    paintDatasetOptions();
    paintDataset();
    paintHistory();
    paintRun();
    countObjective();
    syncControls();
  }

  async function start() {
    if (state.started || !state.alive) return;

    const locale = await initI18n();

    if (!state.alive) return;

    state.started = true;

    review = createReviewController({
      onRunUpdate(value) {
        if (
          state.alive &&
          value.run_id.toLowerCase() === state.activeId
        ) {
          acceptRun(value, true);
        }
      },

      onBusyChange(busy) {
        if (!state.alive) return;

        if (busy) {
          cancelRead("run");
          clearTimeout(pollTimer);
        }

        syncControls();

        if (!busy) {
          schedulePoll();
        }
      },
    });

    voice = createVoiceController({
      onBusyChange: syncControls,
    });

    bind(
      $("new-run-button"),
      "click",
      newRun,
    );

    bind(
      $("refresh-runs-button"),
      "click",
      loadHistory,
    );

    bind(
      $("retry-connection-button"),
      "click",
      connect,
    );

    bind(
      $("upload-form"),
      "submit",
      upload,
    );

    bind($("dataset-select"), "change", () => {
      if (
        !state.pending &&
        !state.uploading &&
        !state.creating
      ) {
        return chooseDataset(
          $("dataset-select").value,
        );
      }
    });

    bind(
      $("dataset-profile-panel"),
      "toggle",
      loadProfile,
    );

    bind(
      $("run-form"),
      "submit",
      submitRun,
    );

    bind($("objective"), "input", () => {
      countObjective();

      $("objective").removeAttribute("aria-invalid");

      if (!state.pending) {
        message("run-form-error");
      }
    });

    bind($("interface-language"), "change", async () => {
      const requested = $("interface-language").value;

      try {
        await setLanguage(requested);
      } catch (error) {
        $("interface-language").value = getLanguage();
        appError(error);
      }
    });

    bind($("run-list"), "click", (event) => {
      const button = event.target.closest(
        "button[data-run-id]",
      );

      if (
        button &&
        $("run-list").contains(button)
      ) {
        selectRun(button.dataset.runId);
      }
    });

    bind(
      $("refresh-run-button"),
      "click",
      refreshRun,
    );

    bind(
      $("retry-run-button"),
      "click",
      retryRun,
    );

    bind(
      checkCreate,
      "click",
      checkPending,
    );

    bind(resendCreate, "click", () => {
      if (state.pending?.mode === "missing") {
        return postCreate();
      }
    });

    bind($("download-report-button"), "click", () => {
      download("report");
    });

    bind($("download-manifest-button"), "click", () => {
      download("manifest");
    });

    bind(document, "visibilitychange", () => {
      if (document.hidden) {
        clearTimeout(pollTimer);
        cancelRead("run");
      } else if (state.ready) {
        return loadRun();
      }
    });

    bind(globalThis, "online", connect);

    bind(globalThis, "pagehide", (event) => {
      clearTimeout(pollTimer);

      for (const name of [...reads.keys()]) {
        cancelRead(name);
      }

      if (!event.persisted) {
        destroy();
      }
    });

    bind(globalThis, "pageshow", (event) => {
      if (event.persisted) {
        return connect();
      }
    });

    unsubscribe = onLanguageChange(repaint);

    repaint();
    await connect();

    if (locale.error && state.alive) {
      appError(locale.error);
    }
  }

  function destroy() {
    if (!state.alive) return;

    state.alive = false;

    clearTimeout(pollTimer);
    clearTimeout(cooldownTimer);

    for (const name of [...reads.keys()]) {
      cancelRead(name);
    }

    bindings.abort();

    voice?.destroy();
    review?.destroy();

    unsubscribe();
    recovery.remove();

    $("upload-form").noValidate =
      originalValidation[0];

    $("run-form").noValidate =
      originalValidation[1];

    for (const [url, timer] of downloads) {
      clearTimeout(timer);
      URL.revokeObjectURL(url);
    }

    downloads.clear();

    // Submitted POST requests may still finish on the server.
  }

  return Object.freeze({
    start,
    destroy,
  });
}

function boot() {
  let application;

  try {
    application = createApp();
    application.start().catch(fail);
  } catch (error) {
    fail(error);
  }

  function fail(error) {
    application?.destroy();

    const panel = document.getElementById("app-error");

    const message = document.getElementById(
      "app-error-message",
    );

    if (panel && message) {
      panel.hidden = false;
      message.hidden = false;

      message.textContent =
        "AutoAudit could not start. Check the browser console and saved frontend files, then reload.";
    }

    console.error(
      "AutoAudit frontend initialization failed.",
      error,
    );
  }
}

if (document.readyState === "loading") {
  document.addEventListener(
    "DOMContentLoaded",
    boot,
    { once: true },
  );
} else {
  boot();
}