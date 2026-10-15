
/**
 * AutoAudit: One Unified ML Project Workflow
 *
 * Goal -> Blueprint -> Dataset -> Data Engineering
 * -> Human Approval -> Training -> Results
 */

const $ = (id) => document.getElementById(id);

const asText = (x) =>
  x == null
    ? "—"
    : typeof x === "object"
      ? JSON.stringify(x)
      : String(x);

const pretty = (x) => JSON.stringify(x, null, 2);

const validId = (x) =>
  /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(x || "");

const isObject = (x) =>
  x !== null && typeof x === "object" && !Array.isArray(x);

const ACTIVE = new Set(["queued", "running"]);

const WAITING = new Set([
  "awaiting_review",
  "awaiting_clarification",
]);

const RETRYABLE = new Set(["failed", "interrupted"]);

// ============================================================
// GLOBAL UI HELPERS
// ============================================================

function notice(id, value = "") {
  const element = $(id);
  if (!element) return;

  element.textContent = value;
  element.hidden = !value;

  if (id === "app-error-message") {
    $("app-error").hidden = !value;
  }
}

// Prevent native form submissions to "/".
// This also protects forms if a module fails to initialize.

function protectForms() {
  const ids = [
    "project-plan-form",
    "upload-form",
    "run-form",
    "review-form",
    "clarification-form",
  ];

  for (const id of ids) {
    $(id)?.addEventListener(
      "submit",
      (event) => event.preventDefault(),
      { capture: true }
    );
  }
}

// ============================================================
// APPLICATION BOOT
// ============================================================

async function boot() {
  protectForms();

  let application;

  try {
    const [http, language, reviewModule] =
      await Promise.all([
        import("./api.js"),
        import("./i18n.js"),
        import("./review.js"),
      ]);

    // Voice is optional.
    const voiceModule =
      await import("./voice.js").catch((error) => {
        console.warn(
          "Voice module unavailable:",
          error
        );
        return null;
      });

    application = createApplication(
      http,
      language,
      reviewModule,
      voiceModule
    );

    await application.start();

  } catch (error) {
    application?.destroy();

    console.error(
      "AutoAudit startup error:",
      error
    );

    notice(
      "app-error-message",
      `Frontend could not initialize: ${error.message}`
    );
  }
}

// ============================================================
// MAIN APPLICATION
// ============================================================

function createApplication(
  http,
  i18n,
  reviewModule,
  voiceModule
) {
  const {
    api,
    createRequestId,
    MAX_CSV_BYTES,
  } = http;

  const {
    initI18n,
    setLanguage,
    getLanguage,
    t,
    formatNumber,
    formatDateTime,
    onLanguageChange,
  } = i18n;

  const state = {
    alive: true,
    ready: false,
    connecting: false,

    datasets: new Map(),
    selectedDataset: "",
    dataset: null,
    profile: null,

    runs: new Map(),
    activeRun: "",
    run: null,

    uploading: false,
    creatingRun: false,
    retrying: false,
    reviewing: false,

    pendingRun: null,

    project: {
      id: "",
      request: null,
      result: null,
      executionId: "",
      progress: null,
      busy: false,
      refreshing: false,
      uncertain: false,
    },
  };

  let review = null;
  let voice = null;
  let pollTimer = null;
  let pollBusy = false;

  let removeLanguageListener = () => {};

  const signals = new AbortController();

  // ==========================================================
  // EVENT HELPERS
  // ==========================================================

  function listen(id, eventName, callback) {
    const node =
      typeof id === "string" ? $(id) : id;

    if (!node) {
      throw new Error(`Missing element: ${id}`);
    }

    node.addEventListener(
      eventName,
      (event) => {
        try {
          Promise.resolve(
            callback(event)
          ).catch(showError);
        } catch (error) {
          showError(error);
        }
      },
      {
        signal: signals.signal,
      }
    );
  }

  function showError(error) {
    if (
      !state.alive ||
      error?.name === "AbortError"
    ) {
      return;
    }

    notice(
      "app-error-message",
      error?.message ||
        "Unexpected error. Check server logs."
    );

    console.error(error);
  }

  const translated = (key, fallback) =>
    t(key, {}, fallback);

  const number = (x) =>
    typeof x === "number" &&
    Number.isFinite(x)
      ? formatNumber(x)
      : "—";

  const date = (x) =>
    x ? formatDateTime(x) : "—";

  // ==========================================================
  // KEY-VALUE DISPLAY
  // ==========================================================

  function details(containerId, entries) {
    const fragment =
      document.createDocumentFragment();

    for (const [key, value] of entries) {
      const div =
        document.createElement("div");

      const dt =
        document.createElement("dt");

      const dd =
        document.createElement("dd");

      dt.textContent = translated(key, key);
      dd.textContent = asText(value);

      div.append(dt, dd);
      fragment.append(div);
    }

    $(containerId).replaceChildren(fragment);
  }

  // ==========================================================
  // BUTTON AND FORM STATES
  // ==========================================================

  function sync() {
    if (!state.alive) return;

    const blocked =
      state.uploading ||
      state.creatingRun ||
      Boolean(state.pendingRun);

    const p = state.project;

    const pBusy =
      p.busy || p.refreshing;

    $("new-run-button").disabled = false;
    $("interface-language").disabled = false;

    $("retry-connection-button").disabled =
      state.connecting;

    $("refresh-runs-button").disabled =
      !state.ready;

    $("upload-fields").disabled =
      !state.ready || blocked;

    $("dataset-select").disabled =
      !state.ready || blocked;

    $("upload-progress").hidden =
      !state.uploading;

    // Hidden legacy form:
    // Actual runs begin through Continue Training.

    $("run-fields").disabled = true;
    $("start-run-button").disabled = true;

    // Project planning

    $("project-plan-fields").disabled =
      !state.ready ||
      pBusy ||
      p.uncertain;

    $("refresh-project-plan-button").disabled =
      !state.ready ||
      !p.id ||
      pBusy;

    $("refresh-project-progress-button").disabled =
      !state.ready ||
      !p.executionId ||
      pBusy;

    // Continue becomes available after
    // blueprint + actual dataset.

    $("continue-training-button").disabled =
      !state.ready ||
      !p.result?.blueprint ||
      !state.dataset ||
      blocked ||
      pBusy;

    $("continue-training-button").setAttribute(
      "aria-busy",
      String(state.creatingRun)
    );

    const hint = $("project-training-hint");

    hint.textContent = state.creatingRun
      ? "Starting Data Engineering and Model Training workflow…"
      : !p.result?.blueprint
        ? "First generate a project blueprint."
        : !state.dataset
          ? "Now upload or select a dataset to continue."
          : p.executionId
            ? "Execution run created. See Agent Progress & Results below."
            : "Ready. Continue starts the existing dataset-backed agents. The blueprint is not yet persisted into that run.";

    $("refresh-run-button").disabled =
      !state.ready ||
      !state.activeRun ||
      state.reviewing;

    const retry = RETRYABLE.has(
      state.run?.status
    );

    $("retry-run-button").hidden = !retry;

    $("retry-run-button").disabled =
      !state.ready ||
      !retry ||
      state.retrying ||
      state.reviewing;

    $("download-report-button").disabled =
      !state.run?.state?.training_report;

    $("download-manifest-button").disabled =
      !state.run?.state?.artifact;

    review?.setEnabled(
      state.ready && !state.retrying
    );

    voice?.setEnabled?.(
      state.ready &&
      !blocked &&
      Boolean(state.dataset)
    );

    $("connection-status").dataset.state =
      state.ready
        ? "ready"
        : state.connecting
          ? "checking"
          : "offline";

    $("connection-status").textContent =
      state.ready
        ? "Workspace ready"
        : state.connecting
          ? "Connecting…"
          : "Offline";

    document.body.dataset.appState =
      state.ready ? "ready" : "offline";
  }

  // ==========================================================
  // DATASET UI
  // ==========================================================

  function renderDatasets() {
    const select = $("dataset-select");
    const previous = state.selectedDataset;

    const options =
      document.createDocumentFragment();

    options.append(
      new Option("Choose a saved dataset", "")
    );

    for (const data of state.datasets.values()) {
      options.append(
        new Option(
          `${data.filename} (${number(data.row_count)} rows)`,
          data.dataset_id
        )
      );
    }

    select.replaceChildren(options);
    select.value = previous;

    const data = state.dataset;

    $("dataset-summary").hidden = !data;

    $("dataset-filename").textContent =
      data?.filename || "";

    $("dataset-row-count").textContent =
      number(data?.row_count);

    $("dataset-column-count").textContent =
      number(
        data?.column_count ??
        data?.columns?.length
      );

    $("dataset-size").textContent =
      Number.isFinite(data?.size_bytes)
        ? `${(data.size_bytes / 1048576).toFixed(2)} MiB`
        : "—";

    $("dataset-profile-json").textContent =
      state.profile ? pretty(state.profile) : "";

    const target = $("target-column");
    const old = target.value;

    target.replaceChildren(
      new Option(
        "Let the planner identify it",
        ""
      )
    );

    for (const column of data?.columns || []) {
      target.append(
        new Option(column, column)
      );
    }

    if (data?.columns?.includes(old)) {
      target.value = old;
    }

    sync();
  }

  // ==========================================================
  // DATASET API
  // ==========================================================

  async function loadDatasets() {
    const list = await api.listDatasets({
      limit: 100,
    });

    if (!Array.isArray(list)) {
      throw new Error("Invalid dataset list.");
    }

    if (!state.alive) return;

    state.datasets.clear();

    for (const data of list) {
      if (validId(data.dataset_id)) {
        state.datasets.set(
          data.dataset_id.toLowerCase(),
          data
        );
      }
    }

    renderDatasets();
  }

  async function chooseDataset(id) {
    state.selectedDataset = id;
    state.dataset = null;
    state.profile = null;

    renderDatasets();

    if (!id) return;

    try {
      const data = await api.getDataset(id);

      if (
        !state.alive ||
        state.selectedDataset !== id
      ) {
        return;
      }

      if (
        !validId(data?.dataset_id) ||
        !Array.isArray(data.columns)
      ) {
        throw new Error(
          "Invalid dataset metadata."
        );
      }

      state.dataset = data;
      state.datasets.set(id, data);

      notice(
        "upload-status",
        `Selected ${data.filename}`
      );

      notice("upload-error");

      renderDatasets();

    } catch (error) {
      notice(
        "upload-error",
        error.message
      );
    }
  }

  // ==========================================================
  // UPLOAD DATASET
  // ==========================================================

  async function uploadDataset(event) {
    event.preventDefault();

    if (
      !state.ready ||
      state.uploading
    ) {
      return;
    }

    const file =
      $("dataset-file").files?.[0];

    if (
      !file ||
      !file.name.toLowerCase().endsWith(".csv") ||
      file.size < 1 ||
      file.size > MAX_CSV_BYTES
    ) {
      notice(
        "upload-error",
        "Select a nonempty CSV file under 25 MB."
      );
      return;
    }

    state.uploading = true;

    notice("upload-error");

    notice(
      "upload-status",
      "Uploading CSV…"
    );

    sync();

    try {
      const data =
        await api.uploadDataset(file);

      if (!validId(data?.dataset_id)) {
        throw new Error(
          "Invalid upload response."
        );
      }

      const id =
        data.dataset_id.toLowerCase();

      state.datasets.set(id, data);

      $("dataset-file").value = "";

      await chooseDataset(id);

    } catch (error) {
      notice(
        "upload-error",
        error.message
      );
    } finally {
      state.uploading = false;
      sync();
    }
  }

  // ==========================================================
  // DATASET PROFILE
  // ==========================================================

  async function loadProfile() {
    if (
      !state.selectedDataset ||
      !$("dataset-profile-panel").open ||
      state.profile
    ) {
      return;
    }

    const id = state.selectedDataset;

    notice(
      "dataset-profile-status",
      "Loading profile…"
    );

    try {
      const profile =
        await api.getDatasetProfile(id);

      if (
        !state.alive ||
        id !== state.selectedDataset
      ) {
        return;
      }

      state.profile = profile;

      $("dataset-profile-json").textContent =
        pretty(profile);

      notice(
        "dataset-profile-status",
        "Profile loaded."
      );

    } catch (error) {
      notice(
        "dataset-profile-status",
        error.message
      );
    }
  }

  // ==========================================================
  // RUN HISTORY
  // ==========================================================

  function renderHistory() {
    const fragment =
      document.createDocumentFragment();

    const runs = [
      ...state.runs.values(),
    ].sort(
      (a, b) =>
        Date.parse(b.updated_at || 0) -
        Date.parse(a.updated_at || 0)
    );

    for (const item of runs.slice(0, 100)) {
      const clone =
        $("run-item-template")
          .content.cloneNode(true);

      const button =
        clone.querySelector("button");

      button.dataset.runId =
        item.run_id;

      button.setAttribute(
        "aria-current",
        String(item.run_id === state.activeRun)
      );

      clone.querySelector(
        '[data-field="title"]'
      ).textContent =
        `Run ${item.run_id.slice(0, 8)}`;

      clone.querySelector(
        '[data-field="status"]'
      ).textContent =
        translated(
          `status.${item.status}`,
          item.status || "Unknown"
        );

      const time =
        clone.querySelector(
          '[data-field="updated_at"]'
        );

      time.textContent =
        date(item.updated_at);

      if (item.updated_at) {
        time.dateTime = item.updated_at;
      }

      fragment.append(clone);
    }

    $("run-list").replaceChildren(fragment);

    $("run-list-empty").hidden =
      runs.length > 0;
  }

  function rememberRun(run) {
    if (!validId(run?.run_id)) return;

    state.runs.set(
      run.run_id.toLowerCase(),
      {
        run_id: run.run_id.toLowerCase(),
        status: run.status,
        updated_at:
          run.updated_at ||
          new Date().toISOString(),
      }
    );

    renderHistory();
  }

  async function loadHistory() {
    notice(
      "run-list-status",
      "Loading runs…"
    );

    try {
      const list = await api.listRuns({
        limit: 100,
      });

      if (!Array.isArray(list)) {
        throw new Error(
          "Invalid run list."
        );
      }

      if (!state.alive) return;

      state.runs.clear();

      for (const item of list) {
        rememberRun(item);
      }

      notice(
        "run-list-status",
        `${list.length} run(s) loaded.`
      );

    } catch (error) {
      notice(
        "run-list-status",
        error.message
      );
    }
  }

  // ==========================================================
  // MASTER PROJECT BLUEPRINT UI
  // ==========================================================

  function renderProject() {
    const p = state.project;

    $("project-blueprint-panel").hidden =
      !p.id;

    $("project-plan-id").textContent =
      p.id;

    const blueprint =
      p.result?.blueprint;

    $("project-blueprint-json").textContent =
      blueprint ? pretty(blueprint) : "";

    $("project-dataset-status").textContent =
      p.busy
        ? "Creating project blueprint…"
        : p.result?.status === "awaiting_dataset"
          ? "Blueprint ready; upload a real dataset before training."
          : p.result?.status === "planned"
            ? "Blueprint ready; continue with a dataset-backed run."
            : p.result?.status === "failed"
              ? asText(
                  p.result.error ||
                  "Planning failed"
                )
              : p.id
                ? "Waiting for Planner response."
                : "";

    const files =
      Array.isArray(blueprint?.files)
        ? blueprint.files
        : [];

    const receipts =
      p.progress?.files || {};

    let done = 0;

    const fragment =
      document.createDocumentFragment();

    for (const file of files) {
      if (!file?.path) continue;

      const status =
        receipts[file.path]?.status ||
        "planned";

      if (
        ["created", "revised"].includes(status)
      ) {
        done++;
      }

      const li =
        document.createElement("li");

      li.className = "run-list-item";

      li.textContent =
        `${file.path} — ` +
        `${file.producer || "Agent"} (${status})`;

      fragment.append(li);
    }

    $("project-planned-file-list")
      .replaceChildren(fragment);

    $("project-file-progress").hidden =
      files.length === 0;

    $("project-file-progress").value =
      files.length
        ? 100 * done / files.length
        : 0;

    $("project-file-status").textContent =
      files.length
        ? `${done} of ${files.length} files confirmed generated. ` +
          (
            p.executionId
              ? "Execution progress is tracked separately."
              : "Blueprint is only a plan; files are not generated yet."
          )
        : "No planned files yet.";

    sync();
  }

  // ==========================================================
  // VALIDATE PROJECT RESPONSE
  // ==========================================================

  function validateProject(data, id) {
    const allowed = [
      "planning",
      "planned",
      "awaiting_dataset",
      "needs_clarification",
      "unsupported",
      "failed",
    ];

    if (
      !isObject(data) ||
      data.run_id?.toLowerCase() !== id ||
      !allowed.includes(data.status) ||
      (
        data.blueprint != null &&
        !Array.isArray(data.blueprint.files)
      )
    ) {
      throw new Error(
        "Unexpected project-planning response."
      );
    }

    return data;
  }

  // ==========================================================
  // STEP 1: CREATE PROJECT BLUEPRINT
  // ==========================================================

  async function createProject(event) {
    event.preventDefault();

    if (
      !state.ready ||
      state.project.busy
    ) {
      return;
    }

    const objective =
      $("project-goal").value.trim();

    if (
      objective.length < 5 ||
      objective.length > 5000
    ) {
      notice(
        "project-plan-error",
        "Describe your ML goal in 5–5000 characters."
      );
      return;
    }

    const useDataset =
      $("project-use-dataset").checked;

    if (
      useDataset &&
      !state.dataset
    ) {
      notice(
        "project-plan-error",
        "Select a dataset first, or uncheck the optional dataset box."
      );
      return;
    }

    const request = {
      request_id: createRequestId(),
      objective,
      dataset_id:
        useDataset
          ? state.selectedDataset
          : null,
    };

    if ($("project-provider").value) {
      request.provider =
        $("project-provider").value;
    }

    if ($("project-language").value) {
      request.language =
        $("project-language").value;
    }

    state.project = {
      id: request.request_id,
      request,
      result: null,
      executionId: "",
      progress: null,
      busy: true,
      refreshing: false,
      uncertain: false,
    };

    notice("project-plan-error");

    notice(
      "project-plan-status",
      "Generating master project blueprint…"
    );

    renderProject();

    try {
      const result =
        await api.createProjectPlan(request);

      if (
        !state.alive ||
        state.project.id !== request.request_id
      ) {
        return;
      }

      state.project.result =
        validateProject(
          result,
          request.request_id
        );

      notice(
        "project-plan-status",
        "Blueprint response received. Upload/select dataset to continue."
      );

    } catch (error) {
      if (
        !state.alive ||
        state.project.id !== request.request_id
      ) {
        return;
      }

      state.project.uncertain =
        Boolean(error.outcomeUnknown);

      notice(
        "project-plan-error",
        error.message
      );

      notice(
        "project-plan-status",
        error.outcomeUnknown
          ? "Request outcome uncertain. Use Refresh Blueprint rather than submitting again."
          : "Planner request failed; check the error above."
      );

    } finally {
      if (
        state.project.id === request.request_id
      ) {
        state.project.busy = false;
        renderProject();
      }
    }
  }

  // ==========================================================
  // REFRESH BLUEPRINT
  // ==========================================================

  async function refreshProject() {
    const p = state.project;

    if (
      !state.ready ||
      !p.id ||
      p.busy ||
      p.refreshing
    ) {
      return;
    }

    p.refreshing = true;
    sync();

    try {
      p.result = validateProject(
        await api.getProjectPlan(p.id),
        p.id
      );

      p.uncertain = false;

      notice("project-plan-error");

      notice(
        "project-plan-status",
        "Blueprint refreshed."
      );

    } catch (error) {
      notice(
        "project-plan-error",
        error.message
      );
    } finally {
      p.refreshing = false;
      renderProject();
    }
  }

  // ==========================================================
  // REFRESH PROJECT FILE PROGRESS
  // ==========================================================

  async function refreshProjectProgress() {
    const p = state.project;

    if (
      !state.ready ||
      !p.executionId ||
      p.busy ||
      p.refreshing
    ) {
      return;
    }

    p.refreshing = true;
    sync();

    try {
      const value =
        await api.getProjectProgress(
          p.executionId
        );

      if (!isObject(value?.files)) {
        throw new Error(
          "Invalid generated-project progress response."
        );
      }

      p.progress = value;

      notice(
        "project-plan-status",
        "Execution file progress refreshed."
      );

    } catch (error) {
      notice(
        "project-plan-status",
        `Progress unavailable: ${error.message}`
      );
    } finally {
      p.refreshing = false;
      renderProject();
    }
  }

  // ==========================================================
  // RUN DETAILS AND AGENT PROGRESS
  // ==========================================================

  function renderRun() {
    const r = state.run;
    const data = r?.state || {};
    const report = data.training_report;

    // Keep the main project form visible.
    $("run-panel").hidden =
      !state.activeRun;

    $("active-run-id").textContent =
      state.activeRun;

    $("active-run-objective").textContent =
      data.request?.objective ||
      state.project.request?.objective ||
      "";

    $("run-status").textContent =
      translated(
        `status.${r?.status}`,
        r?.status || "Loading…"
      );

    $("run-status").dataset.status =
      r?.status || "unknown";

    $("run-progress").hidden =
      !ACTIVE.has(r?.status);

    $("run-updated-at").textContent =
      date(r?.updated_at);

    details(
      "active-run-settings",
      [
        ["settings.provider", data.request?.provider],
        ["settings.model", data.request?.model],
        ["settings.responseLanguage", data.request?.language],
        ["dataset.id", data.request?.dataset_id],
      ]
    );

    const plan = data.execution_plan;

    $("plan-panel").hidden = !plan;

    $("plan-json").textContent =
      plan ? pretty(plan) : "";

    details(
      "plan-summary",
      plan
        ? [
            ["plan.target", plan.target_column],
            ["plan.metric", plan.primary_metric],
            ["plan.split", plan.split?.strategy],
            [
              "plan.candidateModels",
              plan.candidate_models?.join(", "),
            ],
          ]
        : []
    );

    const completedStages = report
      ? 3
      : data.training_decision
        ? 2
        : plan
          ? 1
          : 0;

    const agents = [
      "planner",
      "engineering",
      "training",
    ];

    for (
      const [i, agent] of agents.entries()
    ) {
      const status =
        i < completedStages
          ? "completed"
          : i === completedStages
            ? r?.status || "waiting"
            : "waiting";

      $(`${agent}-step`).dataset.status =
        status;

      $(`${agent}-status`).textContent =
        translated(
          `status.${status}`,
          status
        );
    }

    $("run-error-panel").hidden =
      ![
        "failed",
        "interrupted",
        "rejected",
        "needs_review",
      ].includes(r?.status);

    $("run-error-message").textContent =
      asText(r?.error || r?.status);

    $("training-results-panel").hidden =
      !report;

    $("training-report-json").textContent =
      report ? pretty(report) : "";

    details(
      "training-summary",
      report
        ? [
            [
              "training.selectedModel",
              report.selected_model,
            ],
            [
              "training.primaryMetric",
              report.primary_metric,
            ],
            [
              "training.selectedScore",
              report.selected_validation_score,
            ],
            [
              "training.trainRows",
              report.split_counts?.train,
            ],
            [
              "training.validationRows",
              report.split_counts?.validation,
            ],
            [
              "training.testRows",
              report.split_counts?.test,
            ],
          ]
        : []
    );

    const fragment =
      document.createDocumentFragment();

    for (
      const candidate of report?.candidates || []
    ) {
      const clone =
        $("model-row-template")
          .content.cloneNode(true);

      clone.querySelector(
        '[data-field="model"]'
      ).textContent =
        asText(candidate.model);

      clone.querySelector(
        '[data-field="status"]'
      ).textContent =
        candidate.eligible_for_selection
          ? "Eligible"
          : "Ineligible";

      clone.querySelector(
        '[data-field="primary_score"]'
      ).textContent =
        number(candidate.validation_score);

      clone.querySelector(
        '[data-field="notes"]'
      ).textContent =
        (
          candidate.warning_types || []
        ).join(", ");

      fragment.append(clone);
    }

    $("model-table-body").replaceChildren(
      fragment
    );

    details(
      "artifact-summary",
      data.artifact
        ? [
            [
              "artifact.id",
              data.artifact.artifact_id,
            ],
            [
              "artifact.status",
              data.artifact.status,
            ],
            [
              "artifact.files",
              Object.keys(
                data.artifact.files || {}
              ).join("\n"),
            ],
          ]
        : []
    );

    sync();
  }

  // ==========================================================
  // HANDLE RUN RESPONSE
  // ==========================================================

  function acceptRun(
    value,
    skipReview = false
  ) {
    if (
      !validId(value?.run_id) ||
      value.run_id.toLowerCase() !== state.activeRun ||
      !isObject(value.state) ||
      !Array.isArray(value.interrupts)
    ) {
      throw new Error(
        "Invalid run response."
      );
    }

    if (
      !skipReview &&
      review?.render(value) === false
    ) {
      return;
    }

    state.run = value;

    rememberRun(value);
    renderRun();
    schedulePoll();
  }

  async function selectRun(
    id,
    initial = null
  ) {
    if (!validId(id)) return;

    state.activeRun =
      id.toLowerCase();

    state.run = null;

    clearTimeout(pollTimer);

    review?.clear();

    renderRun();

    if (initial) {
      acceptRun(initial);
    } else {
      await loadRun();
    }

    $("run-panel").focus({
      preventScroll: true,
    });
  }

  async function loadRun() {
    if (
      !state.ready ||
      !state.activeRun ||
      pollBusy ||
      state.reviewing
    ) {
      return;
    }

    const id = state.activeRun;

    pollBusy = true;

    try {
      const result =
        await api.getRun(id);

      if (
        state.alive &&
        state.activeRun === id
      ) {
        acceptRun(result);
      }

    } catch (error) {
      showError(error);

    } finally {
      pollBusy = false;
      schedulePoll();
    }
  }

  // ==========================================================
  // RUN POLLING
  // ==========================================================

  function schedulePoll() {
    clearTimeout(pollTimer);

    if (
      !state.alive ||
      !state.ready ||
      !state.activeRun ||
      document.hidden
    ) {
      return;
    }

    const status =
      state.run?.status;

    if (
      !ACTIVE.has(status) &&
      !WAITING.has(status)
    ) {
      return;
    }

    pollTimer = setTimeout(
      () => void loadRun(),
      WAITING.has(status) ? 10000 : 3000
    );
  }

  // ==========================================================
  // STEP 3: CONTINUE TO DATA ENGINEERING AND TRAINING
  // ==========================================================

  async function continueTraining() {
    const p = state.project;

    if (
      !state.ready ||
      !p.result?.blueprint ||
      !state.dataset ||
      state.creatingRun ||
      state.pendingRun
    ) {
      return;
    }

    /*
     * Current backend integration:
     * Start a dataset-backed run with the same ML objective.
     *
     * The separate blueprint has NOT yet been attached
     * to the dataset-backed execution record.
     */

    const request = {
      request_id: createRequestId(),
      dataset_id: state.selectedDataset,
      objective: p.request.objective,
    };

    if (p.request.provider) {
      request.provider = p.request.provider;
    }

    if (p.request.language) {
      request.language = p.request.language;
    }

    state.pendingRun = request;
    state.creatingRun = true;

    notice("project-plan-error");

    notice(
      "project-plan-status",
      "Starting dataset-backed Data Engineering and Training run…"
    );

    sync();

    try {
      const result =
        await api.createRun(request);

      if (!state.alive) return;

      state.pendingRun = null;

      p.executionId =
        request.request_id;

      await selectRun(
        request.request_id,
        result
      );

      notice(
        "project-plan-status",
        "Execution started. Follow agent progress and approve cleaning before training."
      );

    } catch (error) {
      if (!error.outcomeUnknown) {
        state.pendingRun = null;
      }

      notice(
        "project-plan-error",
        error.message
      );

      notice(
        "project-plan-status",
        error.outcomeUnknown
          ? `Execution request may have succeeded. Do not click again. Run ID: ${request.request_id}. Refresh the run list.`
          : "Could not start execution. Check the error."
      );

    } finally {
      state.creatingRun = false;
      sync();
    }
  }

  // ==========================================================
  // RETRY FAILED RUN
  // ==========================================================

  async function retryRun() {
    if (
      !state.activeRun ||
      !RETRYABLE.has(state.run?.status) ||
      state.retrying
    ) {
      return;
    }

    const id = state.activeRun;

    state.retrying = true;
    sync();

    try {
      const current =
        await api.getRun(id);

      if (
        id !== state.activeRun ||
        !RETRYABLE.has(current.status)
      ) {
        return;
      }

      acceptRun(
        await api.retryRun(id)
      );

    } catch (error) {
      showError(error);

    } finally {
      state.retrying = false;
      sync();
    }
  }

  // ==========================================================
  // DOWNLOAD RESULTS
  // ==========================================================

  function downloadJSON(kind) {
    const data =
      kind === "report"
        ? state.run?.state?.training_report
        : state.run?.state?.artifact;

    if (
      !data ||
      !state.activeRun
    ) {
      return;
    }

    const url =
      URL.createObjectURL(
        new Blob(
          [pretty(data)],
          {
            type: "application/json;charset=utf-8",
          }
        )
      );

    const a =
      document.createElement("a");

    a.href = url;

    a.download =
      `autoaudit-${state.activeRun}-${kind}.json`;

    document.body.append(a);

    a.click();
    a.remove();

    setTimeout(
      () => URL.revokeObjectURL(url),
      10000
    );
  }

  // ==========================================================
  // NEW PROJECT
  // ==========================================================

  function resetProject() {
    if (
      state.creatingRun ||
      state.project.busy
    ) {
      return;
    }

    state.project = {
      id: "",
      request: null,
      result: null,
      executionId: "",
      progress: null,
      busy: false,
      refreshing: false,
      uncertain: false,
    };

    state.activeRun = "";
    state.run = null;
    state.pendingRun = null;

    review?.clear();

    clearTimeout(pollTimer);

    $("project-plan-form").reset();

    $("project-plan-status").textContent = "";

    notice("project-plan-error");
    notice("app-error-message");

    renderProject();
    renderRun();
    renderHistory();

    $("project-goal").focus();
  }

  // ==========================================================
  // BACKEND CONNECTION
  // ==========================================================

  async function connect() {
    if (
      !state.alive ||
      state.connecting
    ) {
      return;
    }

    state.connecting = true;
    sync();

    try {
      const response =
        await api.getReadiness();

      if (response?.status !== "ready") {
        throw new Error(
          "Backend is not ready."
        );
      }

      state.ready = true;

      notice("app-error-message");

      sync();

      await Promise.allSettled([
        loadHistory(),
        loadDatasets(),
      ]);

      if (state.activeRun) {
        await loadRun();
      }

    } catch (error) {
      state.ready = false;
      showError(error);

    } finally {
      state.connecting = false;
      sync();
      schedulePoll();
    }
  }

  // ==========================================================
  // REPAINT UI
  // ==========================================================

  function repaint() {
    renderDatasets();
    renderHistory();
    renderProject();
    renderRun();
    sync();
  }

  // ==========================================================
  // START APPLICATION
  // ==========================================================

  async function start() {
    // Main workflow

    listen(
      "project-plan-form",
      "submit",
      createProject
    );

    listen(
      "refresh-project-plan-button",
      "click",
      refreshProject
    );

    listen(
      "refresh-project-progress-button",
      "click",
      refreshProjectProgress
    );

    listen(
      "continue-training-button",
      "click",
      continueTraining
    );

    listen(
      "new-run-button",
      "click",
      resetProject
    );

    // Dataset

    listen(
      "upload-form",
      "submit",
      uploadDataset
    );

    listen(
      "dataset-select",
      "change",
      (event) =>
        chooseDataset(event.target.value)
    );

    listen(
      "dataset-profile-panel",
      "toggle",
      loadProfile
    );

    // Run history

    listen(
      "refresh-runs-button",
      "click",
      loadHistory
    );

    listen(
      "refresh-run-button",
      "click",
      loadRun
    );

    listen(
      "retry-run-button",
      "click",
      retryRun
    );

    listen(
      "retry-connection-button",
      "click",
      connect
    );

    // Downloads

    listen(
      "download-report-button",
      "click",
      () => downloadJSON("report")
    );

    listen(
      "download-manifest-button",
      "click",
      () => downloadJSON("manifest")
    );

    // Select previous run

    listen(
      "run-list",
      "click",
      (event) => {
        const button =
          event.target.closest(
            "button[data-run-id]"
          );

        if (button) {
          selectRun(
            button.dataset.runId
          );
        }
      }
    );

    // Hidden compatibility form

    listen(
      "run-form",
      "submit",
      (event) => event.preventDefault()
    );

    // Language

    listen(
      "interface-language",
      "change",
      async (event) => {
        try {
          await setLanguage(
            event.target.value
          );

        } catch (error) {
          event.target.value =
            getLanguage();

          showError(error);
        }
      }
    );

    // Browser events

    listen(
      document,
      "visibilitychange",
      () => {
        if (document.hidden) {
          clearTimeout(pollTimer);
        } else {
          void loadRun();
        }
      }
    );

    listen(
      window,
      "online",
      connect
    );

    listen(
      window,
      "pagehide",
      (event) => {
        if (!event.persisted) {
          destroy();
        }
      }
    );

    // Initialize interface translations

    const locale = await initI18n();

    if (!state.alive) return;

    // Human-in-the-loop controller

    review = reviewModule.createReviewController({
      onRunUpdate(value) {
        if (
          state.alive &&
          value?.run_id?.toLowerCase() ===
            state.activeRun
        ) {
          acceptRun(value, true);
        }
      },

      onBusyChange(value) {
        state.reviewing = value;

        if (value) {
          clearTimeout(pollTimer);
        } else {
          schedulePoll();
        }

        sync();
      },
    });

    // Optional voice controller

    if (voiceModule?.createVoiceController) {
      try {
        voice =
          voiceModule.createVoiceController({
            onBusyChange: sync,
          });

      } catch (error) {
        console.warn(
          "Voice unavailable:",
          error
        );
      }
    }

    removeLanguageListener =
      onLanguageChange(repaint);

    repaint();

    await connect();

    if (locale?.error) {
      console.warn(
        "Language pack fallback:",
        locale.error
      );
    }
  }

  // ==========================================================
  // CLEANUP
  // ==========================================================

  function destroy() {
    if (!state.alive) return;

    state.alive = false;

    clearTimeout(pollTimer);

    signals.abort();

    review?.destroy?.();
    voice?.destroy?.();

    removeLanguageListener();
  }

  return {
    start,
    destroy,
  };
}

// ============================================================
// BOOTSTRAP
// ============================================================

if (document.readyState === "loading") {
  document.addEventListener(
    "DOMContentLoaded",
    () => void boot(),
    { once: true }
  );
} else {
  void boot();
}
