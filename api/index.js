const DEFAULT_MODEL = "gemini-3.6-flash";
const FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"];
const MAX_OUTPUT_TOKENS = 400;
const THINKING_LEVEL_HISTORY = "low";
const THINKING_LEVEL_SINGLE = "minimal";
// Free-tier rate limits are tracked per model, so a 429 is worth retrying elsewhere.
const RETRYABLE_STATUSES = new Set([404, 429, 500, 502, 503, 504]);
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 1200;
const MAX_ALICE_TEXT = 1024;
const TOTAL_RESPONSE_BUDGET_MS = 3200;
const SERPER_TIMEOUT_MS = 650;
const GEMINI_TIMEOUT_MS = 2300;

const SYSTEM_PROMPT = [
  "Ты — умная Алиса. Отвечай голосом: по-русски, 1–3 коротких предложения, без списков, таблиц и разметки.",
  "Правила:",
  "Всегда отвечай на русском, даже если вопрос задан на другом языке.",
  "Не озвучивай форматирование: числа пиши словами, длинные конструкции сокращай.",
  "Если в запросе есть результаты поиска, опирайся на них и не выдумывай факты, которых там нет.",
  "Если не знаешь ответа — прямо скажи об этом и предложи, как уточнить вопрос.",
  "Если вопрос понятен, отвечай сразу, без уточняющих вопросов."
].join(" ");

function aliceResponse(text, endSession = false, history = []) {
  return {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: { history }
  };
}

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (text.length <= MAX_ALICE_TEXT) return text || "Не удалось подготовить ответ.";

  const candidate = text.slice(0, MAX_ALICE_TEXT - 1);
  const sentenceEnd = Math.max(
    candidate.lastIndexOf(". "),
    candidate.lastIndexOf("! "),
    candidate.lastIndexOf("? ")
  );
  if (sentenceEnd >= 300) return candidate.slice(0, sentenceEnd + 1).trim();

  const lastSpace = candidate.lastIndexOf(" ");
  return (lastSpace > 200 ? candidate.slice(0, lastSpace) : candidate).trim() + "…";
}

function isSearchIntent(text) {
  return /\b(latest|today|current|news|weather|price|exchange rate|recent|search|look up)\b|сегодня|сейчас|последн|новост|погод|курс валют|найди|поищи|актуальн|сколько стоит|свежие данные/i.test(text);
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter(item => item && (item.role === "user" || item.role === "model") && typeof item.text === "string")
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({ role: item.role, text: item.text.slice(0, MAX_HISTORY_MESSAGE_CHARS) }));
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(100, timeoutMs));
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  } finally {
    clearTimeout(timer);
  }
}

async function searchWeb(query, timeoutMs, stats) {
  const key = process.env.SERPER_API_KEY;
  if (!key || timeoutMs < 150) return "";

  const searchStartedAt = Date.now();

  try {
    const { response, data } = await fetchJson(
      "https://google.serper.dev/search",
      {
        method: "POST",
        headers: {
          "X-API-KEY": key,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ q: query, gl: "ru", hl: "ru", num: 3 })
      },
      Math.min(SERPER_TIMEOUT_MS, timeoutMs)
    );

    if (!response.ok) {
      console.warn("Serper request failed with HTTP", response.status);
      return "";
    }

    if (stats) stats.searchMs = Date.now() - searchStartedAt;

    const items = (Array.isArray(data.organic) ? data.organic : [])
      .slice(0, 3)
      .map((item, i) => {
        const title = String(item.title || "").slice(0, 180);
        const snippet = String(item.snippet || "").slice(0, 350);
        const link = String(item.link || "").slice(0, 240);
        return `${i + 1}. ${title}: ${snippet} ${link}`.trim();
      });

    const answerBox = data.answerBox?.answer || data.answerBox?.snippet;
    if (answerBox) items.unshift(`Краткий ответ поисковика: ${String(answerBox).slice(0, 350)}`);
    return items.length ? items.join("\n") : "";
  } catch (error) {
    console.warn("Serper request did not finish in time; continuing without search.");
    return "";
  }
}

function looksBroken(text) {
  if (!text) return true;
  return /правила:|коротких предложения|отвечай голосом|no lists|short sentences|не используй markdown/i.test(text);
}

function modelChain() {
  const configured = String(process.env.GEMINI_MODEL || DEFAULT_MODEL).trim();
  return [configured, ...FALLBACK_MODELS]
    .filter(model => /^[A-Za-z0-9._-]+$/.test(model))
    .filter((model, index, list) => list.indexOf(model) === index);
}

async function askGemini(text, history, searchContext, timeoutMs, stats) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("MISSING_GEMINI_KEY");
  if (timeoutMs < 250) throw new Error("TIME_BUDGET_EXHAUSTED");

  const models = modelChain();
  if (models.length === 0) throw new Error("INVALID_MODEL_NAME");

  const contents = safeHistory(history).map(item => ({
    role: item.role,
    parts: [{ text: item.text }]
  }));

  const prompt = searchContext
    ? `${text}\n\nРезультаты поиска в интернете:\n${searchContext}\nИспользуй их только там, где они относятся к вопросу.`
    : text;
  contents.push({ role: "user", parts: [{ text: prompt }] });

  // Thinking is on by default in Gemini 3 and shares maxOutputTokens with the answer,
  // so a small cap silently truncates the reply down to a stub.
  // Single-turn questions can think minimally; follow-ups need thought signatures,
  // which we do not store, so they stay on the low level.
  const thinkingLevel = contents.length > 1 ? THINKING_LEVEL_HISTORY : THINKING_LEVEL_SINGLE;

  const buildBody = thinking => JSON.stringify({
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents,
    generationConfig: {
      temperature: 0.5,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      ...(thinking ? { thinkingConfig: { thinkingLevel } } : {})
    }
  });

  const call = async (model, thinking, timeout) => {
    try {
      const result = await fetchJson(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": key
          },
          body: buildBody(thinking)
        },
        timeout
      );
      return { result };
    } catch (networkError) {
      if (networkError?.name !== "AbortError") throw networkError;
      return { timeout: true };
    }
  };

  const answerFrom = data => (data?.candidates?.[0]?.content?.parts || [])
    .filter(part => !part?.thought)
    .map(part => (typeof part.text === "string" ? part.text : ""))
    .join("")
    .trim();

  const timedOut = model => Object.assign(new Error("TIME_BUDGET_EXHAUSTED"), {
    detail: `модель ${model} не ответила за отведённое время`
  });

  let thinking = true;
  let budget = timeoutMs;
  let lastError;

  for (const model of models) {
    const startedAt = Date.now();
    const spent = () => Date.now() - startedAt;
    const limit = () => Math.min(GEMINI_TIMEOUT_MS, Math.max(0, budget - spent()));

    let attempt = await call(model, thinking, limit());

    // One retry without the thinking config in case this model rejects the field.
    if (!attempt.timeout && !attempt.result.response.ok && attempt.result.response.status === 400 && thinking) {
      thinking = false;
      console.warn("Gemini rejected thinkingConfig, retrying", model, "without it");
      attempt = await call(model, thinking, limit());
    }

    budget -= spent();
    if (attempt.timeout) {
      lastError = timedOut(model);
      budget = 0;
      break;
    }

    const { response, data } = attempt.result;

    if (response.ok) {
      const answer = answerFrom(data);
      if (!answer) {
        if (data?.promptFeedback?.blockReason) throw new Error("GEMINI_BLOCKED");
        throw new Error("GEMINI_EMPTY_RESPONSE");
      }
      if (looksBroken(answer)) {
        console.warn("Gemini echoed the system prompt on", model, "; trying the next model");
        lastError = Object.assign(new Error("GEMINI_REQUEST_FAILED"), {
          detail: `модель ${model} вернула некорректный ответ`
        });
        if (budget < 500) break;
        continue;
      }
      if (stats) {
        stats.geminiMs = Date.now() - startedAt;
        stats.model = model;
        stats.thinkingLevel = thinking ? thinkingLevel : "off";
      }
      return answer;
    }

    const status = Number(response.status);
    const detail = String(data?.error?.message || "").slice(0, 200);
    console.error("Gemini request failed with HTTP", status, "on model", model, detail);
    if (status === 401 || status === 403) throw new Error("GEMINI_KEY_REJECTED");

    lastError = status === 429
      ? new Error("GEMINI_QUOTA")
      : Object.assign(new Error("GEMINI_REQUEST_FAILED"), {
          detail: `HTTP ${status} на ${model}${detail ? `: ${detail}` : ""}`
        });

    // Overloaded or temporarily unavailable model: try the next one in the chain.
    if (!RETRYABLE_STATUSES.has(status)) break;
    if (budget < 500) break;
  }

  throw lastError || new Error("GEMINI_REQUEST_FAILED");
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GEMINI_KEY":
      return "Сервер пока не настроен. Проверьте переменную GEMINI_API_KEY в настройках Vercel.";
    case "GEMINI_KEY_REJECTED":
      return "Не удалось подключиться к Gemini. Проверьте API-ключ в настройках Vercel.";
    case "GEMINI_QUOTA":
      return "Сейчас превышен лимит запросов к Gemini. Попробуйте немного позже.";
    case "GEMINI_BLOCKED":
      return "Я не смогла обработать этот запрос. Попробуйте спросить иначе.";
    case "TIME_BUDGET_EXHAUSTED":
    case "AbortError":
      return "Не успела подготовить ответ. Попробуйте задать вопрос ещё раз, чуть короче.";
    default: {
      const base = "Не получилось получить ответ от нейросети. Попробуйте ещё раз чуть позже.";
      const detail = String(error?.detail || "").trim();
      return detail ? `${base} Код ошибки: ${detail}` : base;
    }
  }
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-gemini-serper",
      message: "Webhook is ready. Yandex Dialogs should send POST requests to /api."
    });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const startedAt = Date.now();
  const remainingMs = () => TOTAL_RESPONSE_BUDGET_MS - (Date.now() - startedAt);
  const body = req.body && typeof req.body === "object" ? req.body : {};
  const text = String(body?.request?.command || body?.request?.original_utterance || "").trim();
  const oldHistory = safeHistory(body?.state?.session?.history);

  if (!text) {
    return res.status(200).json(aliceResponse(
      "Привет! Я могу ответить на вопрос или найти актуальную информацию. Что хочешь узнать?",
      false,
      oldHistory
    ));
  }

  const stats = {};

  try {
    let searchContext = "";
    if (isSearchIntent(text) && process.env.SERPER_API_KEY) {
      // Search has a very short budget because Alice expects a response within 4.5 seconds.
      searchContext = await searchWeb(text, Math.min(SERPER_TIMEOUT_MS, Math.max(0, remainingMs() - 2200)), stats);
    }

    const answer = await askGemini(
      text,
      oldHistory,
      searchContext,
      Math.max(0, Math.min(GEMINI_TIMEOUT_MS, remainingMs() - 300)),
      stats
    );

    const history = [
      ...oldHistory,
      { role: "user", text: text.slice(0, MAX_HISTORY_MESSAGE_CHARS) },
      { role: "model", text: answer.slice(0, MAX_HISTORY_MESSAGE_CHARS) }
    ].slice(-MAX_HISTORY_MESSAGES);

    res.setHeader(
      "X-Alice-Timing",
      `search=${stats.searchMs ?? 0}ms gemini=${stats.geminiMs ?? 0}ms model=${stats.model ?? "none"} thinking=${stats.thinkingLevel ?? "n\/a"} total=${Date.now() - startedAt}ms`
    );
    return res.status(200).json(aliceResponse(answer, false, history));
  } catch (error) {
    // Keep a real Error object here: a bare string would break the mapping in errorForUser.
    const safeError = error?.name === "AbortError" || error?.message === "AbortError"
      ? Object.assign(new Error("TIME_BUDGET_EXHAUSTED"), { detail: "превышено время ответа" })
      : error;
    console.warn(
      "Alice webhook returned a fallback response:",
      safeError?.message || "unknown error",
      `search=${stats.searchMs ?? 0}ms gemini=${stats.geminiMs ?? 0}ms model=${stats.model ?? "none"}`
    );
    return res.status(200).json(aliceResponse(errorForUser(safeError), false, oldHistory));
  }
}

export { aliceResponse, fitAliceText, isSearchIntent, modelChain, safeHistory };
