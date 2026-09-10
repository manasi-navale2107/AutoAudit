/**
 * Optional browser dictation for the problem statement.
 *
 * app.js creates this controller after initI18n(), then calls setEnabled().
 * While busy, disable experiment submission, not the whole run fieldset.
 * Call clear() before changing experiments or replacing the objective.
 *
 * stop() accepts remaining final results.
 * cancel() discards pending results.
 *
 * Importing this module does not access the microphone.
 */

import {
  getLanguage,
  onLanguageChange,
  t,
} from "./i18n.js";

const LIMIT = 5_000;

const START_TIMEOUT_MS = 15_000;
const SESSION_TIMEOUT_MS = 60_000;
const FINISH_TIMEOUT_MS = 8_000;

const LANGUAGES = Object.freeze({
  en: "en-IN",
  mr: "mr-IN",
  hi: "hi-IN",
});

const controllers = new WeakMap();

// These fallbacks work with the locale files already provided.
const EXTRA_TEXT = Object.freeze({
  languageLabel: {
    en: "Dictation language",
    mr: "बोलण्याची भाषा",
    hi: "बोलने की भाषा",
  },

  browserHelp: {
    en: "Choose the language you will speak. Your browser may send audio to an online speech service.",
    mr: "तुम्ही बोलणार ती भाषा निवडा. आवाज ओळखण्यासाठी browser ऑनलाइन speech service कडे आवाज पाठवू शकतो.",
    hi: "आप जिस भाषा में बोलेंगे, उसे चुनें. आवाज़ पहचानने के लिए browser ऑनलाइन speech service को ऑडियो भेज सकता है.",
  },

  languageUnavailable: {
    en: "The speech service does not support this language. Choose another language or type your statement.",
    mr: "या speech service मध्ये ही भाषा उपलब्ध नाही. दुसरी भाषा निवडा किंवा मजकूर टाइप करा.",
    hi: "इस speech service में यह भाषा उपलब्ध नहीं है. दूसरी भाषा चुनें या अपना विवरण टाइप करें.",
  },
});

function translate(key) {
  const name = key.slice("voice.".length);
  const fallback = EXTRA_TEXT[name];

  return t(
    key,
    {},
    fallback?.[getLanguage()] ?? fallback?.en ?? key,
  );
}

function errorKey(code) {
  switch (code) {
    case "not-allowed":
    case "NotAllowedError":
    case "SecurityError":
      return "voice.permissionDenied";

    case "audio-capture":
    case "NotFoundError":
    case "NotReadableError":
      return "voice.microphoneUnavailable";

    case "language-not-supported":
      return "voice.languageUnavailable";

    case "service-not-allowed":
    case "NotSupportedError":
      return "voice.unavailable";

    case "no-speech":
      return "voice.noSpeech";

    case "aborted":
    case "AbortError":
      return "voice.stopped";

    default:
      return "voice.failed";
  }
}

function abortRecognition(recognition) {
  try {
    recognition.abort();
  } catch {
    // Some browsers throw if capture has already ended.
  }
}

export function createVoiceController({
  onBusyChange = () => {},
} = {}) {
  if (typeof onBusyChange !== "function") {
    throw new TypeError("onBusyChange must be a function.");
  }

  function required(id, tag) {
    const element = document.getElementById(id);

    if (!element || (tag && element.tagName !== tag)) {
      throw new Error(
        `Missing or invalid voice element: #${id}`,
      );
    }

    return element;
  }

  const controls = required("voice-controls");
  const button = required("voice-input-button", "BUTTON");
  const status = required("voice-status");

  const objective = required("objective", "TEXTAREA");
  const form = required("run-form", "FORM");
  const fields = required("run-fields", "FIELDSET");
  const panel = required("new-run-panel");

  if (controllers.has(controls)) {
    throw new Error(
      "A voice controller already exists for this form.",
    );
  }

  const Recognition =
    globalThis.SpeechRecognition ??
    globalThis.webkitSpeechRecognition;

  const supported =
    globalThis.isSecureContext === true &&
    typeof Recognition === "function";

  let enabled = false;
  let destroyed = false;
  let session = null;

  let statusKey = supported ? "" : "voice.unavailable";
  let timer = null;
  let writing = false;

  const removers = [];

  // Add a separate dictation-language selector.
  const languageField = document.createElement("div");
  languageField.className = "field field--compact";

  const languageLabel = document.createElement("label");
  const languageSelect = document.createElement("select");

  languageSelect.id = "voice-language";
  languageLabel.htmlFor = languageSelect.id;

  languageLabel.setAttribute(
    "data-i18n",
    "voice.languageLabel",
  );

  for (const [code, label] of [
    ["en", "English"],
    ["mr", "मराठी"],
    ["hi", "हिन्दी"],
  ]) {
    const option = document.createElement("option");

    option.value = code;
    option.lang = code;
    option.textContent = label;

    languageSelect.append(option);
  }

  languageSelect.value = getLanguage();

  languageField.append(
    languageLabel,
    languageSelect,
  );

  const help = document.createElement("p");

  help.id = "voice-browser-help";
  help.className = "field-help";

  help.setAttribute(
    "data-i18n",
    "voice.browserHelp",
  );

  languageSelect.setAttribute(
    "aria-describedby",
    help.id,
  );

  controls.insertBefore(languageField, button);
  controls.append(help);

  function listen(target, event, handler, options) {
    target.addEventListener(event, handler, options);

    removers.push(() => {
      target.removeEventListener(event, handler, options);
    });
  }

  function canDictate() {
    return (
      !destroyed &&
      supported &&
      enabled &&
      objective.isConnected &&
      !objective.matches(":disabled") &&
      !objective.readOnly &&
      !objective.closest("[hidden]") &&
      !document.hidden
    );
  }

  function render() {
    if (destroyed) return;

    const active = session !== null;
    const key = active ? "voice.stop" : "voice.dictate";

    controls.hidden = false;

    button.hidden = !supported;
    languageField.hidden = !supported;
    help.hidden = !supported;

    button.disabled =
      !canDictate() ||
      session?.phase === "stopping";

    languageSelect.disabled =
      !canDictate() || active;

    button.setAttribute(
      "aria-pressed",
      String(active),
    );

    button.setAttribute(
      "aria-busy",
      String(active && session.phase !== "listening"),
    );

    button.setAttribute("data-i18n", key);
    button.textContent = translate(key);

    languageLabel.textContent = translate(
      "voice.languageLabel",
    );

    help.textContent = translate(
      "voice.browserHelp",
    );

    status.textContent = statusKey
      ? translate(statusKey)
      : "";

    if (statusKey) {
      status.setAttribute("data-i18n", statusKey);
    } else {
      status.removeAttribute("data-i18n");
    }
  }

  function notifyBusy() {
    try {
      onBusyChange(session !== null);
    } catch {
      console.error(
        "AutoAudit: voice state callback failed.",
      );
    }
  }

  function arm(current, delay, action) {
    clearTimeout(timer);

    timer = setTimeout(() => {
      if (session === current) {
        action();
      }
    }, delay);
  }

  function finish(current, key, abort = false) {
    if (session !== current) return;

    session = null;

    clearTimeout(timer);
    timer = null;

    const recognition = current.recognition;

    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    recognition.onaudioend = null;

    // A permission prompt can resolve after cancellation.
    // Stop any capture that begins after this session ended.
    recognition.onstart = () => {
      abortRecognition(recognition);
    };

    recognition.onaudiostart = () => {
      abortRecognition(recognition);
    };

    if (abort) {
      abortRecognition(recognition);
    }

    statusKey = key;

    render();
    notifyBusy();
  }

  function cancel() {
    if (!session) return false;

    finish(
      session,
      "voice.stopped",
      true,
    );

    return true;
  }

  function stop() {
    const current = session;

    if (!current || current.phase === "stopping") {
      return false;
    }

    if (current.phase === "starting") {
      return cancel();
    }

    current.phase = "stopping";
    statusKey = "voice.processing";

    render();

    arm(
      current,
      FINISH_TIMEOUT_MS,
      () => finish(current, "voice.failed", true),
    );

    try {
      current.recognition.stop();
    } catch (error) {
      finish(
        current,
        errorKey(error?.name),
        true,
      );
    }

    return true;
  }

  function limit() {
    return objective.maxLength >= 0
      ? Math.min(LIMIT, objective.maxLength)
      : LIMIT;
  }

  function receive(current, event) {
    if (session !== current) return;

    if (
      !canDictate() ||
      objective.value !== current.expectedValue
    ) {
      cancel();
      return;
    }

    const results = event.results;

    if (!results) return;

    const parts = [];

    for (
      let index = 0;
      index < results.length;
      index += 1
    ) {
      const result = results[index];

      if (
        !result?.isFinal ||
        current.seen.has(index)
      ) {
        continue;
      }

      const transcript = result[0]?.transcript;

      if (typeof transcript !== "string") {
        continue;
      }

      current.seen.add(index);

      if (transcript.trim()) {
        parts.push(transcript.trim());
      }
    }

    if (!parts.length) return;

    const previous = objective.value;

    const separator =
      previous && !/\s$/u.test(previous)
        ? " "
        : "";

    const next =
      previous +
      separator +
      parts.join(" ");

    // Respect the textarea's UTF-16 maxlength.
    // This also satisfies the API's character limit.
    if (next.length > limit()) {
      finish(
        current,
        "voice.lengthExceeded",
        true,
      );

      return;
    }

    current.expectedValue = next;
    current.added = true;

    objective.value = next;

    // Notify app.js so it updates counts and validation.
    writing = true;

    try {
      objective.dispatchEvent(
        new Event("input", {
          bubbles: true,
        }),
      );
    } finally {
      writing = false;
    }

    // Another input listener may have replaced the draft
    // or disabled the form while handling this event.
    if (
      session === current &&
      (!canDictate() || objective.value !== next)
    ) {
      cancel();
    }
  }

  function start() {
    if (!canDictate() || session) {
      return false;
    }

    if (objective.value.length >= limit()) {
      statusKey = "voice.lengthExceeded";
      render();

      return false;
    }

    let recognition;

    try {
      recognition = new Recognition();

      recognition.lang =
        LANGUAGES[languageSelect.value] ??
        LANGUAGES.en;

      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
    } catch (error) {
      statusKey = errorKey(error?.name);
      render();

      return false;
    }

    const current = {
      recognition,
      phase: "starting",
      expectedValue: objective.value,
      seen: new Set(),
      added: false,
    };

    session = current;
    statusKey = "voice.starting";

    recognition.onstart = () => {
      if (session !== current) {
        abortRecognition(recognition);
        return;
      }

      if (!canDictate()) {
        cancel();
        return;
      }

      if (current.phase !== "starting") return;

      current.phase = "listening";
      statusKey = "voice.listening";

      render();

      arm(
        current,
        SESSION_TIMEOUT_MS,
        stop,
      );
    };

    recognition.onaudiostart = () => {
      if (session !== current) {
        abortRecognition(recognition);
      } else if (!canDictate()) {
        cancel();
      }
    };

    recognition.onresult = (event) => {
      receive(current, event);
    };

    recognition.onerror = (event) => {
      finish(
        current,
        errorKey(event.error),
        true,
      );
    };

    recognition.onaudioend = () => {
      if (
        session !== current ||
        current.phase === "stopping"
      ) {
        return;
      }

      current.phase = "stopping";
      statusKey = "voice.processing";

      render();

      arm(
        current,
        FINISH_TIMEOUT_MS,
        () => finish(current, "voice.failed", true),
      );
    };

    recognition.onend = () => {
      finish(
        current,
        current.added
          ? "voice.added"
          : "voice.noSpeech",
      );
    };

    render();
    notifyBusy();

    // The app callback may cancel dictation.
    if (session !== current) {
      return false;
    }

    if (!canDictate()) {
      cancel();
      return false;
    }

    arm(
      current,
      START_TIMEOUT_MS,
      () => finish(current, "voice.failed", true),
    );

    try {
      recognition.start();
    } catch (error) {
      finish(
        current,
        errorKey(error?.name),
        true,
      );

      return false;
    }

    return session === current;
  }

  function clear() {
    if (destroyed) return;

    cancel();

    statusKey = supported
      ? ""
      : "voice.unavailable";

    render();
  }

  listen(button, "click", () => {
    if (session) {
      stop();
    } else {
      start();
    }
  });

  listen(
    languageSelect,
    "change",
    cancel,
  );

  listen(objective, "beforeinput", () => {
    if (!writing) cancel();
  });

  listen(objective, "input", () => {
    if (!writing) cancel();
  });

  listen(
    form,
    "submit",
    cancel,
    true,
  );

  listen(form, "reset", clear);

  listen(document, "visibilitychange", () => {
    if (document.hidden) cancel();
  });

  listen(globalThis, "pagehide", cancel);

  listen(document, "keydown", (event) => {
    if (event.key === "Escape") {
      cancel();
    }
  });

  for (const id of [
    "new-run-button",
    "dataset-select",
  ]) {
    const element = document.getElementById(id);

    if (element) {
      listen(
        element,
        id === "dataset-select" ? "change" : "click",
        cancel,
      );
    }
  }

  const observer = new MutationObserver(() => {
    if (session && !canDictate()) {
      cancel();
    }

    render();
  });

  for (const element of [
    fields,
    objective,
    panel,
  ]) {
    observer.observe(element, {
      attributes: true,

      attributeFilter: [
        "disabled",
        "readonly",
        "hidden",
      ],
    });
  }

  const unsubscribe = onLanguageChange(render);

  const controller = Object.freeze({
    get supported() {
      return supported;
    },

    get busy() {
      return session !== null;
    },

    start,
    stop,
    cancel,
    clear,

    setEnabled(value) {
      if (typeof value !== "boolean") {
        throw new TypeError(
          "enabled must be a boolean.",
        );
      }

      if (destroyed) return;

      enabled = value;

      if (!enabled) {
        cancel();
      }

      render();
    },

    destroy() {
      if (destroyed) return;

      destroyed = true;
      enabled = false;

      cancel();

      observer.disconnect();
      unsubscribe();

      removers.forEach((remove) => {
        remove();
      });

      languageField.remove();
      help.remove();

      button.disabled = true;

      button.setAttribute(
        "aria-pressed",
        "false",
      );

      button.setAttribute(
        "aria-busy",
        "false",
      );

      button.setAttribute(
        "data-i18n",
        "voice.dictate",
      );

      button.textContent = translate(
        "voice.dictate",
      );

      status.textContent = "";
      status.removeAttribute("data-i18n");

      controls.hidden = true;

      controllers.delete(controls);
    },
  });

  controllers.set(controls, controller);

  render();

  return controller;
}