
/**
 * AutoAudit HTTP API Client
 *
 * Supports:
 * 1. Existing dataset upload and profiling
 * 2. Existing three-agent ML training workflow
 * 3. Human-in-the-loop approval and retries
 * 4. Goal-only project blueprint generation
 * 5. Project planning status and generated file progress
 *
 * The final project ZIP is NOT generated or exposed here.
 */

export const MAX_CSV_BYTES = 25 * 1024 * 1024;

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 30_000;
const PLANNING_TIMEOUT_MS = 180_000;

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;


// ============================================================
// API ERROR
// ============================================================

export class ApiError extends Error {
  constructor(
    message,
    {
      code = "api_error",
      status = 0,
      issues = [],
      retryAfterMs = null,
      outcomeUnknown = false,
      cause,
    } = {},
  ) {
    super(message, { cause });

    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.issues = issues;
    this.retryAfterMs = retryAfterMs;
    this.outcomeUnknown = outcomeUnknown;
  }
}


// ============================================================
// REQUEST HELPERS
// ============================================================

export function isAbortError(error) {
  return (
    error?.code === "request_cancelled" ||
    error?.name === "AbortError"
  );
}


export function createRequestId() {
  if (typeof globalThis.crypto?.randomUUID !== "function") {
    throw new ApiError(
      "Use localhost or HTTPS to create runs.",
      { code: "secure_context_required" },
    );
  }

  return globalThis.crypto.randomUUID();
}


function isRecord(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}


function inputError(message, field = "") {
  return new ApiError(message, {
    code: "invalid_input",
    issues: field ? [{ field, message }] : [],
  });
}


function uuid(value, field) {
  if (
    typeof value !== "string" ||
    !UUID_PATTERN.test(value)
  ) {
    throw inputError(
      `${field} must be a valid UUID.`,
      field,
    );
  }

  return value.toLowerCase();
}


function textField(
  value,
  field,
  min,
  max,
  trim = true,
) {
  if (typeof value !== "string") {
    throw inputError(
      `${field} must be text.`,
      field,
    );
  }

  const result = trim ? value.trim() : value;

  const length =
    result.length > max * 2
      ? Infinity
      : [...result].length;

  if (
    !result.trim() ||
    length < min ||
    length > max
  ) {
    throw inputError(
      `${field} must contain ${min}-${max} characters.`,
      field,
    );
  }

  return result;
}


function checkFields(value, allowed) {
  if (!isRecord(value)) {
    throw inputError(
      "Provide a valid request object.",
    );
  }

  const allowedFields = new Set(allowed);

  for (const key of Object.keys(value)) {
    if (!allowedFields.has(key)) {
      throw inputError(
        `Unexpected request field: ${key}`,
        key,
      );
    }
  }
}


function providerField(value) {
  if (
    value !== "groq" &&
    value !== "ollama"
  ) {
    throw inputError(
      "Unsupported provider.",
      "provider",
    );
  }

  return value;
}


function languageField(value) {
  if (!["en", "mr", "hi"].includes(value)) {
    throw inputError(
      "Unsupported language.",
      "language",
    );
  }

  return value;
}


function optionalFields(value, body) {
  for (const key of [
    "target_column",
    "provider",
    "language",
    "model",
  ]) {
    const item = value[key];

    if (
      item === undefined ||
      item === null ||
      item === ""
    ) {
      continue;
    }

    if (key === "provider") {
      body.provider = providerField(item);
    } else if (key === "language") {
      body.language = languageField(item);
    } else {
      body[key] = textField(
        item,
        key,
        1,
        200,
        key !== "target_column",
      );
    }
  }

  if (body.model && !body.provider) {
    throw inputError(
      "Select a provider when specifying a model.",
      "provider",
    );
  }

  return body;
}


// ============================================================
// EXISTING DATASET-BACKED RUN REQUEST
// ============================================================

function runPayload(value) {
  checkFields(value, [
    "request_id",
    "dataset_id",
    "objective",
    "target_column",
    "provider",
    "language",
    "model",
  ]);

  const body = {
    request_id: uuid(
      value.request_id,
      "request_id",
    ),

    dataset_id: uuid(
      value.dataset_id,
      "dataset_id",
    ),

    objective: textField(
      value.objective,
      "objective",
      5,
      5000,
    ),
  };

  return optionalFields(value, body);
}


// ============================================================
// NEW: GOAL-ONLY PROJECT PLANNING REQUEST
// ============================================================

function projectPlanPayload(value) {
  checkFields(value, [
    "request_id",
    "objective",
    "dataset_id",
    "target_column",
    "provider",
    "language",
    "model",
  ]);

  const body = {
    request_id: uuid(
      value.request_id,
      "request_id",
    ),

    objective: textField(
      value.objective,
      "objective",
      5,
      5000,
    ),

    dataset_id: (
      value.dataset_id === undefined ||
      value.dataset_id === null ||
      value.dataset_id === ""
    )
      ? null
      : uuid(
          value.dataset_id,
          "dataset_id",
        ),
  };

  return optionalFields(value, body);
}


// ============================================================
// HUMAN REVIEW REQUEST
// ============================================================

function reviewPayload(value) {
  checkFields(value, [
    "revision",
    "action",
    "feedback",
  ]);

  if (
    !Number.isSafeInteger(value.revision) ||
    value.revision < 1
  ) {
    throw inputError(
      "Review revision must be a positive integer.",
      "revision",
    );
  }

  if (
    ![
      "approve",
      "changes",
      "reject",
      "answer",
    ].includes(value.action)
  ) {
    throw inputError(
      "Unsupported review action.",
      "action",
    );
  }

  const body = {
    revision: value.revision,
    action: value.action,
  };

  if (
    value.feedback !== undefined &&
    value.feedback !== null
  ) {
    body.feedback = textField(
      value.feedback,
      "feedback",
      1,
      4000,
    );
  }

  if (
    ["changes", "answer"].includes(body.action) &&
    !body.feedback
  ) {
    throw inputError(
      "Feedback is required.",
      "feedback",
    );
  }

  if (
    body.action === "approve" &&
    body.feedback !== undefined
  ) {
    throw inputError(
      "Use Request Changes to modify a proposal.",
      "feedback",
    );
  }

  return body;
}


function listLimit(value) {
  if (
    !Number.isInteger(value) ||
    value < 1 ||
    value > 100
  ) {
    throw inputError(
      "Limit must be between 1 and 100.",
      "limit",
    );
  }

  return value;
}


// ============================================================
// HTTP OPTIONS
// ============================================================

function requestOptions(
  options,
  defaultTimeout = DEFAULT_TIMEOUT_MS,
) {
  return {
    signal: options?.signal,
    timeoutMs: options?.timeoutMs ?? defaultTimeout,
  };
}


// ============================================================
// API URL
// ============================================================

function apiURL(path) {
  if (
    !["http:", "https:"].includes(
      globalThis.location?.protocol,
    )
  ) {
    throw new ApiError(
      "Open AutoAudit through its running backend.",
      { code: "invalid_app_origin" },
    );
  }

  const base =
    globalThis.document?.body?.dataset?.apiBase ??
    "/api";

  if (
    typeof base !== "string" ||
    !/^\/[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*\/?$/.test(base)
  ) {
    throw new ApiError(
      "The API base path is invalid.",
      { code: "invalid_api_base" },
    );
  }

  const cleanBase = base.replace(/\/$/, "");

  return new URL(
    `${cleanBase}/${path}`,
    globalThis.location.origin,
  ).href;
}


// ============================================================
// ERROR RESPONSE HELPERS
// ============================================================

function shortText(
  value,
  fallback,
  max = 600,
) {
  return (
    typeof value === "string" &&
    value.trim()
  )
    ? value.trim().slice(0, max)
    : fallback;
}


function retryDelay(response) {
  const value = response.headers
    .get("retry-after")
    ?.trim();

  if (!value) {
    return null;
  }

  const milliseconds = /^\d+$/.test(value)
    ? Number(value) * 1000
    : Date.parse(value) - Date.now();

  if (!Number.isFinite(milliseconds)) {
    return null;
  }

  return Math.max(
    0,
    Math.ceil(milliseconds),
  );
}


function httpError(response, data) {
  let code = `http_${response.status}`;

  let message =
    `The server returned HTTP ${response.status}.`;

  let issues = [];

  const detail = isRecord(data)
    ? data.detail
    : null;

  if (isRecord(detail)) {
    code = shortText(
      detail.code,
      code,
      80,
    );

    message = shortText(
      detail.message,
      message,
    );
  } else if (Array.isArray(detail)) {
    code = "validation_error";

    message =
      "Check the request fields and try again.";

    issues = detail
      .filter(isRecord)
      .slice(0, 12)
      .map((item) => ({
        field: (
          Array.isArray(item.loc)
            ? item.loc
            : []
        )
          .filter(
            (part) =>
              typeof part === "string" ||
              Number.isInteger(part),
          )
          .filter(
            (part) => part !== "body",
          )
          .join(".")
          .slice(0, 160),

        message: shortText(
          item.msg,
          "Invalid value.",
          400,
        ),
      }));
  } else if (typeof detail === "string") {
    message = shortText(
      detail,
      message,
    );
  } else if (data?.status === "not_ready") {
    code = "not_ready";

    message =
      "Application services are not ready yet.";
  }

  return new ApiError(message, {
    code,
    status: response.status,
    issues,
    retryAfterMs: retryDelay(response),
  });
}


// ============================================================
// BOUNDED JSON RESPONSE READER
// ============================================================

async function readJSON(response) {
  const mediaType = (
    response.headers.get("content-type") ?? ""
  )
    .split(";", 1)[0]
    .trim()
    .toLowerCase();

  const isJSON =
    mediaType === "application/json" ||
    mediaType.endsWith("+json");

  const declaredLength = response.headers.get(
    "content-length",
  );

  if (
    declaredLength !== null &&
    Number(declaredLength) > MAX_RESPONSE_BYTES
  ) {
    await response.body?.cancel().catch(() => {});

    throw new ApiError(
      "The API response is too large.",
      { code: "response_too_large" },
    );
  }

  if (!isJSON) {
    await response.body?.cancel().catch(() => {});

    if (!response.ok) {
      return null;
    }

    throw new ApiError(
      "Expected a JSON response.",
      { code: "invalid_response" },
    );
  }

  if (!response.body) {
    return null;
  }

  const reader = response.body.getReader();

  const decoder = new TextDecoder("utf-8", {
    fatal: true,
  });

  const chunks = [];
  let received = 0;
  let complete = false;

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        complete = true;
        break;
      }

      received += value.byteLength;

      if (received > MAX_RESPONSE_BYTES) {
        throw new ApiError(
          "The API response is too large.",
          { code: "response_too_large" },
        );
      }

      try {
        chunks.push(
          decoder.decode(value, {
            stream: true,
          }),
        );
      } catch {
        throw new ApiError(
          "Invalid UTF-8 in API response.",
          { code: "invalid_response" },
        );
      }
    }

    try {
      chunks.push(decoder.decode());

      return JSON.parse(
        chunks.join(""),
      );
    } catch {
      if (!response.ok) {
        return null;
      }

      throw new ApiError(
        "The server returned invalid JSON.",
        { code: "invalid_response" },
      );
    }
  } finally {
    if (!complete) {
      await reader.cancel().catch(() => {});
    }

    reader.releaseLock();
  }
}


// ============================================================
// MAIN HTTP REQUEST FUNCTION
// ============================================================

async function request(
  path,
  {
    method = "GET",
    json,
    form,
    expected = "object",
    signal,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = {},
) {
  if (
    !Number.isFinite(timeoutMs) ||
    timeoutMs < 1 ||
    timeoutMs > 300_000
  ) {
    throw inputError(
      "timeoutMs must be between 1 and 300000.",
    );
  }

  if (
    signal != null &&
    !(signal instanceof AbortSignal)
  ) {
    throw inputError(
      "signal must be an AbortSignal.",
    );
  }

  if (signal?.aborted) {
    throw new ApiError(
      "Request cancelled.",
      { code: "request_cancelled" },
    );
  }

  const url = apiURL(path);

  const headers = {
    Accept: "application/json",
  };

  let body = form;

  if (json !== undefined) {
    headers["Content-Type"] =
      "application/json";

    body = JSON.stringify(json);
  }

  const controller = new AbortController();

  let timedOut = false;
  let started = false;
  let receivedStatus = 0;

  const cancel = () => controller.abort();

  signal?.addEventListener(
    "abort",
    cancel,
    { once: true },
  );

  const timer = setTimeout(() => {
    if (!controller.signal.aborted) {
      timedOut = true;
      controller.abort();
    }
  }, timeoutMs);

  try {
    started = true;

    const response = await fetch(url, {
      method,
      headers,
      body,
      signal: controller.signal,
      credentials: "same-origin",
      mode: "same-origin",
      redirect: "error",
      cache: "no-store",
      referrerPolicy: "same-origin",
    });

    receivedStatus = response.status;

    const data = await readJSON(
      response,
    );

    if (!response.ok) {
      throw httpError(
        response,
        data,
      );
    }

    const invalidShape =
      expected === "array"
        ? !Array.isArray(data)
        : !isRecord(data);

    if (invalidShape) {
      throw new ApiError(
        "Unexpected API response shape.",
        { code: "invalid_response" },
      );
    }

    return data;
  } catch (cause) {
    let error;

    if (controller.signal.aborted) {
      error = new ApiError(
        timedOut
          ? "The request timed out."
          : "Request cancelled.",
        {
          code: timedOut
            ? "request_timeout"
            : "request_cancelled",
        },
      );
    } else if (cause instanceof ApiError) {
      error = cause;
    } else {
      error = new ApiError(
        "The connection to AutoAudit was interrupted.",
        {
          code: "network_error",
          cause,
        },
      );
    }

    error.status =
      error.status || receivedStatus;

    // A lost POST response does not prove failure.
    // The browser should check the saved request ID
    // before asking the user to submit it again.

    error.outcomeUnknown =
      method !== "GET" &&
      started &&
      (
        receivedStatus === 0 ||
        receivedStatus >= 500 ||
        (
          receivedStatus >= 200 &&
          receivedStatus < 300
        )
      );

    throw error;
  } finally {
    clearTimeout(timer);

    signal?.removeEventListener(
      "abort",
      cancel,
    );
  }
}


// ============================================================
// PUBLIC API METHODS
// ============================================================

export const api = Object.freeze({

  // ----------------------------------------------------------
  // HEALTH AND READINESS
  // ----------------------------------------------------------

  async getHealth(options) {
    return request(
      "health",
      requestOptions(options, 8000),
    );
  },

  async getReadiness(options) {
    return request(
      "ready",
      requestOptions(options, 8000),
    );
  },


  // ----------------------------------------------------------
  // DATASET UPLOAD
  // ----------------------------------------------------------

  async uploadDataset(file, options) {
    if (!(file instanceof File)) {
      throw inputError(
        "Choose a CSV file.",
        "file",
      );
    }

    if (
      !file.name.trim()
        .toLowerCase()
        .endsWith(".csv")
    ) {
      throw inputError(
        "Choose a .csv file.",
        "file",
      );
    }

    if (
      file.size === 0 ||
      file.size > MAX_CSV_BYTES
    ) {
      throw inputError(
        "CSV must be nonempty and at most 25 MB.",
        "file",
      );
    }

    const form = new FormData();

    form.append(
      "file",
      file,
      file.name,
    );

    return request(
      "datasets",
      {
        ...requestOptions(options, 120_000),
        method: "POST",
        form,
      },
    );
  },


  // ----------------------------------------------------------
  // DATASET HISTORY AND PROFILE
  // ----------------------------------------------------------

  async listDatasets({
    limit = 50,
    ...options
  } = {}) {
    return request(
      `datasets?limit=${listLimit(limit)}`,
      {
        ...requestOptions(options),
        expected: "array",
      },
    );
  },

  async getDataset(datasetId, options) {
    return request(
      `datasets/${uuid(datasetId, "dataset_id")}`,
      requestOptions(options),
    );
  },

  async getDatasetProfile(datasetId, options) {
    return request(
      `datasets/${uuid(datasetId, "dataset_id")}/profile`,
      requestOptions(options, 120_000),
    );
  },


  // ----------------------------------------------------------
  // NEW: GOAL-ONLY PROJECT BLUEPRINT GENERATION
  // ----------------------------------------------------------

  async createProjectPlan(payload, options) {
    return request(
      "projects/plan",
      {
        ...requestOptions(
          options,
          PLANNING_TIMEOUT_MS,
        ),

        method: "POST",

        json: projectPlanPayload(
          payload,
        ),
      },
    );
  },


  // ----------------------------------------------------------
  // NEW: FETCH SAVED PROJECT BLUEPRINT
  // ----------------------------------------------------------

  async getProjectPlan(runId, options) {
    return request(
      `projects/plan/${uuid(runId, "run_id")}`,
      requestOptions(options),
    );
  },


  // ----------------------------------------------------------
  // EXISTING: DATASET-BACKED ML WORKFLOW
  // ----------------------------------------------------------

  async createRun(payload, options) {
    return request(
      "runs",
      {
        ...requestOptions(options),
        method: "POST",
        json: runPayload(payload),
      },
    );
  },


  // ----------------------------------------------------------
  // RUN HISTORY
  // ----------------------------------------------------------

  async listRuns({
    limit = 50,
    ...options
  } = {}) {
    return request(
      `runs?limit=${listLimit(limit)}`,
      {
        ...requestOptions(options),
        expected: "array",
      },
    );
  },

  async getRun(runId, options) {
    return request(
      `runs/${uuid(runId, "run_id")}`,
      requestOptions(options),
    );
  },


  // ----------------------------------------------------------
  // NEW: GENERATED PROJECT FILE PROGRESS
  // ----------------------------------------------------------

  async getProjectProgress(runId, options) {
    return request(
      `runs/${uuid(runId, "run_id")}/project`,
      requestOptions(options),
    );
  },


  // ----------------------------------------------------------
  // HUMAN-IN-THE-LOOP REVIEW
  // ----------------------------------------------------------

  async submitReview(runId, decision, options) {
    return request(
      `runs/${uuid(runId, "run_id")}/reviews`,
      {
        ...requestOptions(options),
        method: "POST",
        json: reviewPayload(decision),
      },
    );
  },


  // ----------------------------------------------------------
  // RETRY FAILED / INTERRUPTED RUN
  // ----------------------------------------------------------

  async retryRun(runId, options) {
    return request(
      `runs/${uuid(runId, "run_id")}/retry`,
      {
        ...requestOptions(options),
        method: "POST",
      },
    );
  },

});
