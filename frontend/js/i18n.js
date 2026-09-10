/**
 * AutoAudit interface translations.
 *
 * Locale files:
 *   /static/locales/en.json
 *   /static/locales/mr.json
 *   /static/locales/hi.json
 *
 * JSON uses nested objects with string values.
 * Placeholders use the {name} format.
 *
 * app.js calls initI18n() and binds the language selector.
 * Dynamic renderers use t() and onLanguageChange().
 *
 * Update a status element's data-i18n key when its status changes.
 * Agent output and response-language are managed separately.
 */

export const DEFAULT_LANGUAGE = "en";

export const SUPPORTED_LANGUAGES = Object.freeze([
  "en",
  "mr",
  "hi",
]);

const STORAGE_KEY = "autoaudit.interfaceLanguage";
const LOAD_TIMEOUT_MS = 10_000;
const CHANGE_EVENT = "autoaudit:languagechange";

const INTL_LOCALES = Object.freeze({
  en: "en-IN",
  mr: "mr-IN",
  hi: "hi-IN",
});

const BINDINGS = Object.freeze([
  ["data-i18n", null],
  ["data-i18n-aria-label", "aria-label"],
]);

const SELECTOR = BINDINGS
  .map(([source]) => `[${source}]`)
  .join(",");

const NO_TEXT_TAGS = new Set([
  "INPUT",
  "TEXTAREA",
  "SELECT",
  "SCRIPT",
  "STYLE",
  "TEMPLATE",
  "CODE",
  "PRE",
]);

const catalogues = new Map();
const pendingLoads = new Map();
const htmlFallbacks = new Map();

let language = DEFAULT_LANGUAGE;
let changeVersion = 0;
let fallbacksCaptured = false;
let initialization = null;


export class I18nError extends Error {
  constructor(
    message,
    {
      code = "locale_unavailable",
      language: requestedLanguage = null,
      status = 0,
      cause,
    } = {},
  ) {
    super(message, { cause });

    this.name = "I18nError";
    this.code = code;
    this.language = requestedLanguage;
    this.status = status;
  }
}


function isRecord(value) {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
  );
}


function visitBindings(root, visit) {
  const elements = [
    ...root.querySelectorAll(SELECTOR),
  ];

  if (root.nodeType === 1 && root.matches(SELECTOR)) {
    elements.unshift(root);
  }

  for (const element of elements) {
    if (element.isContentEditable) {
      continue;
    }

    for (const [source, attribute] of BINDINGS) {
      const key = element.getAttribute(source)?.trim();

      if (!key) {
        continue;
      }

      if (attribute === null) {
        // Preserve form values, JSON, nested controls, and icons.
        if (
          NO_TEXT_TAGS.has(element.tagName)
          || element.childElementCount > 0
        ) {
          continue;
        }

        // Without an explicit value, an option uses its text as its value.
        if (
          element.tagName === "OPTION"
          && !element.hasAttribute("value")
        ) {
          continue;
        }
      }

      visit(element, attribute, key);
    }
  }
}


function captureEnglishFallbacks() {
  if (fallbacksCaptured) {
    return;
  }

  visitBindings(document, (element, attribute, key) => {
    const value = attribute === null
      ? element.textContent
      : element.getAttribute(attribute);

    if (value?.trim() && !htmlFallbacks.has(key)) {
      htmlFallbacks.set(key, value);
    }
  });

  htmlFallbacks.set(
    "app.title",
    document.title || "AutoAudit | ML Workspace",
  );

  fallbacksCaptured = true;
}


function readPreference() {
  try {
    const saved = globalThis.localStorage?.getItem(
      STORAGE_KEY,
    );

    return SUPPORTED_LANGUAGES.includes(saved)
      ? saved
      : DEFAULT_LANGUAGE;
  } catch {
    return DEFAULT_LANGUAGE;
  }
}


function savePreference(value) {
  try {
    globalThis.localStorage?.setItem(
      STORAGE_KEY,
      value,
    );
  } catch {
    // Language switching still works during this browser session.
  }
}


function flattenCatalogue(source, requestedLanguage) {
  const result = new Map();

  const invalid = () => new I18nError(
    "The translation file has an invalid format.",
    {
      code: "invalid_locale",
      language: requestedLanguage,
    },
  );

  function walk(value, prefix, depth) {
    if (!isRecord(value) || depth > 8) {
      throw invalid();
    }

    const entries = Object.entries(value);

    if (entries.length === 0) {
      throw invalid();
    }

    for (const [name, item] of entries) {
      if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(name)) {
        throw invalid();
      }

      const key = prefix
        ? `${prefix}.${name}`
        : name;

      if (typeof item === "string") {
        if (
          !item.trim()
          || item.length > 10_000
          || result.size >= 2_000
        ) {
          throw invalid();
        }

        result.set(key, item);
      } else {
        walk(item, key, depth + 1);
      }
    }
  }

  walk(source, "", 0);

  return result;
}


async function fetchCatalogue(requestedLanguage) {
  const controller = new AbortController();

  const timer = setTimeout(
    () => controller.abort(),
    LOAD_TIMEOUT_MS,
  );

  try {
    const response = await fetch(
      `/static/locales/${requestedLanguage}.json`,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        credentials: "same-origin",
        mode: "same-origin",
        redirect: "error",
        cache: "no-cache",
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      throw new I18nError(
        "The translation file could not be loaded.",
        {
          language: requestedLanguage,
          status: response.status,
        },
      );
    }

    const contentType = response.headers
      .get("content-type")
      ?.split(";", 1)[0]
      .trim()
      .toLowerCase();

    if (contentType !== "application/json") {
      throw new I18nError(
        "The translation file must contain JSON.",
        {
          code: "invalid_locale",
          language: requestedLanguage,
          status: response.status,
        },
      );
    }

    let source;

    try {
      source = await response.json();
    } catch (cause) {
      if (controller.signal.aborted) {
        throw cause;
      }

      throw new I18nError(
        "The translation file contains invalid JSON.",
        {
          code: "invalid_locale",
          language: requestedLanguage,
          cause,
        },
      );
    }

    return flattenCatalogue(
      source,
      requestedLanguage,
    );
  } catch (cause) {
    if (cause instanceof I18nError) {
      throw cause;
    }

    throw new I18nError(
      controller.signal.aborted
        ? "Loading the translation file timed out."
        : "The translation file could not be reached.",
      {
        code: controller.signal.aborted
          ? "locale_timeout"
          : "locale_unavailable",
        language: requestedLanguage,
        cause,
      },
    );
  } finally {
    clearTimeout(timer);
  }
}


function loadCatalogue(requestedLanguage) {
  if (catalogues.has(requestedLanguage)) {
    return Promise.resolve(
      catalogues.get(requestedLanguage),
    );
  }

  if (pendingLoads.has(requestedLanguage)) {
    return pendingLoads.get(requestedLanguage);
  }

  const pending = fetchCatalogue(requestedLanguage)
    .then((catalogue) => {
      catalogues.set(
        requestedLanguage,
        catalogue,
      );

      return catalogue;
    })
    .finally(() => {
      pendingLoads.delete(requestedLanguage);
    });

  pendingLoads.set(
    requestedLanguage,
    pending,
  );

  return pending;
}


export function getLanguage() {
  return language;
}


/**
 * Returns plain text.
 *
 * Example:
 *   t("history.runLabel", { id: "abc123" }, "Run {id}")
 *
 * Insert the returned value using textContent.
 */
export function t(
  key,
  parameters = {},
  fallback = key,
) {
  if (
    typeof key !== "string"
    || typeof fallback !== "string"
  ) {
    throw new TypeError(
      "Translation keys and fallbacks must be strings.",
    );
  }

  const message = (
    catalogues.get(language)?.get(key)
    ?? catalogues.get(DEFAULT_LANGUAGE)?.get(key)
    ?? htmlFallbacks.get(key)
    ?? fallback
  );

  if (!isRecord(parameters)) {
    return message;
  }

  return message.replace(
    /\{([A-Za-z][A-Za-z0-9_]*)\}/g,
    (placeholder, name) => {
      if (!Object.hasOwn(parameters, name)) {
        return placeholder;
      }

      const value = parameters[name];

      if (
        typeof value === "string"
        || typeof value === "boolean"
      ) {
        return String(value);
      }

      if (
        typeof value === "number"
        && Number.isFinite(value)
      ) {
        return String(value);
      }

      return placeholder;
    },
  );
}


/**
 * Translates static labels inside a document, element,
 * or cloned document fragment.
 *
 * Dynamic text containing parameters should be rendered
 * with t() by app.js or review.js.
 */
export function applyTranslations(root = document) {
  visitBindings(root, (element, attribute, key) => {
    const translated = t(key, {}, "");

    if (!translated) {
      return;
    }

    if (attribute === null) {
      if (element.textContent !== translated) {
        element.textContent = translated;
      }
    } else if (
      element.getAttribute(attribute) !== translated
    ) {
      element.setAttribute(
        attribute,
        translated,
      );
    }
  });
}


function syncLanguageSelector() {
  const selector = document.getElementById(
    "interface-language",
  );

  if (selector) {
    selector.value = language;
  }
}


/**
 * Resolves:
 *   true  — the language was applied.
 *   false — a newer language choice superseded this request.
 *
 * A failed change throws I18nError and preserves the
 * currently applied language.
 */
export async function setLanguage(
  requestedLanguage,
  { persist = true } = {},
) {
  if (!SUPPORTED_LANGUAGES.includes(requestedLanguage)) {
    throw new I18nError(
      "Choose English, Marathi, or Hindi.",
      {
        code: "unsupported_language",
        language: requestedLanguage,
      },
    );
  }

  captureEnglishFallbacks();

  const version = ++changeVersion;

  try {
    await Promise.all([
      loadCatalogue(DEFAULT_LANGUAGE),
      loadCatalogue(requestedLanguage),
    ]);
  } catch (error) {
    if (version !== changeVersion) {
      return false;
    }

    syncLanguageSelector();

    throw error;
  }

  // A slower response must not overwrite a newer choice.
  if (version !== changeVersion) {
    return false;
  }

  const previousLanguage = language;

  language = requestedLanguage;

  document.documentElement.lang = language;

  document.title = t(
    "app.title",
    {},
    "AutoAudit | ML Workspace",
  );

  applyTranslations();
  syncLanguageSelector();

  if (persist) {
    savePreference(language);
  }

  document.dispatchEvent(
    new CustomEvent(
      CHANGE_EVENT,
      {
        detail: Object.freeze({
          language,
          previousLanguage,
        }),
      },
    ),
  );

  return true;
}


/**
 * Call once during application startup.
 *
 * Returns:
 *   {
 *     language,
 *     requestedLanguage,
 *     error
 *   }
 *
 * Missing locale files leave the original English HTML
 * available. app.js can display the returned error.
 */
export function initI18n() {
  if (initialization === null) {
    const requestedLanguage = readPreference();

    initialization = (async () => {
      try {
        await setLanguage(
          requestedLanguage,
          { persist: false },
        );

        return {
          language,
          requestedLanguage,
          error: null,
        };
      } catch (error) {
        if (!(error instanceof I18nError)) {
          throw error;
        }

        return {
          language,
          requestedLanguage,
          error,
        };
      }
    })();
  }

  return initialization;
}


/**
 * Listener receives:
 *   { language, previousLanguage }
 *
 * Returns a function that removes the listener.
 */
export function onLanguageChange(listener) {
  if (typeof listener !== "function") {
    throw new TypeError(
      "Provide a language change listener.",
    );
  }

  const handler = (event) => {
    listener(event.detail);
  };

  document.addEventListener(
    CHANGE_EVENT,
    handler,
  );

  return () => {
    document.removeEventListener(
      CHANGE_EVENT,
      handler,
    );
  };
}


export function formatNumber(
  value,
  options = { maximumFractionDigits: 4 },
) {
  if (
    typeof value !== "number"
    || !Number.isFinite(value)
  ) {
    return "—";
  }

  return new Intl.NumberFormat(
    INTL_LOCALES[language],
    options,
  ).format(value);
}


/**
 * ISO timestamps use the browser's timezone by default.
 * Supply options.timeZone to choose another timezone.
 */
export function formatDateTime(
  value,
  options = {
    dateStyle: "medium",
    timeStyle: "short",
  },
) {
  if (
    !(value instanceof Date)
    && typeof value !== "string"
    && typeof value !== "number"
  ) {
    return "—";
  }

  if (
    typeof value === "string"
    && !value.trim()
  ) {
    return "—";
  }

  const date = value instanceof Date
    ? value
    : new Date(value);

  if (!Number.isFinite(date.getTime())) {
    return "—";
  }

  return new Intl.DateTimeFormat(
    INTL_LOCALES[language],
    options,
  ).format(date);
}