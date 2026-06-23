/**
 * Screenshot providers for opportunity cover images.
 *
 * Env:
 *   OPPORTUNITY_SCREENSHOT_PROVIDER — screenshotone | microlink (auto if unset)
 *   SCREENSHOTONE_ACCESS_KEY — ScreenshotOne (https://screenshotone.com)
 *   MICROLINK_API_KEY — optional Microlink Pro key
 */

const SCREENSHOT_TIMEOUT_MS = 70_000;
const VIEWPORT_WIDTH = 1200;
const VIEWPORT_HEIGHT = 630;

export function getScreenshotProvider() {
  const explicit = process.env.OPPORTUNITY_SCREENSHOT_PROVIDER?.toLowerCase();
  if (explicit === "screenshotone" || explicit === "microlink") {
    return explicit;
  }
  if (process.env.SCREENSHOTONE_ACCESS_KEY) {
    return "screenshotone";
  }
  return "microlink";
}

async function fetchWithTimeout(url, options = {}, timeoutMs = SCREENSHOT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function takeScreenshotOne(pageUrl) {
  const accessKey = process.env.SCREENSHOTONE_ACCESS_KEY;
  if (!accessKey) {
    throw new Error("SCREENSHOTONE_ACCESS_KEY is not set");
  }

  const api = new URL("https://api.screenshotone.com/take");
  api.searchParams.set("access_key", accessKey);
  api.searchParams.set("url", pageUrl);
  api.searchParams.set("format", "jpg");
  api.searchParams.set("viewport_width", String(VIEWPORT_WIDTH));
  api.searchParams.set("viewport_height", String(VIEWPORT_HEIGHT));
  api.searchParams.set("block_ads", "true");
  api.searchParams.set("block_cookie_banners", "true");
  api.searchParams.set("block_banners_by_heuristics", "false");
  api.searchParams.set("block_trackers", "true");
  api.searchParams.set("delay", "0");
  api.searchParams.set("timeout", "60");
  api.searchParams.set("response_type", "by_format");
  api.searchParams.set("image_quality", "80");

  const response = await fetchWithTimeout(api.href, {}, 65_000);
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `ScreenshotOne HTTP ${response.status}${body ? `: ${body.slice(0, 120)}` : ""}`,
    );
  }

  const contentType = response.headers.get("content-type") ?? "image/jpeg";
  const buffer = Buffer.from(await response.arrayBuffer());

  const sourceUrl = new URL(api.href);
  sourceUrl.searchParams.set("access_key", "[redacted]");

  return { buffer, contentType, provider: "screenshotone", sourceUrl: sourceUrl.href };
}

async function takeMicrolink(pageUrl) {
  const api = new URL("https://api.microlink.io/");
  api.searchParams.set("url", pageUrl);
  api.searchParams.set("screenshot", "true");
  api.searchParams.set("meta", "false");
  api.searchParams.set("viewport.width", String(VIEWPORT_WIDTH));
  api.searchParams.set("viewport.height", String(VIEWPORT_HEIGHT));
  api.searchParams.set("waitForTimeout", "2000");

  if (process.env.MICROLINK_API_KEY) {
    api.searchParams.set("apiKey", process.env.MICROLINK_API_KEY);
  }

  const response = await fetchWithTimeout(api.href);
  if (!response.ok) {
    throw new Error(`Microlink HTTP ${response.status}`);
  }

  const payload = await response.json();
  if (payload.status !== "success") {
    throw new Error(payload.message ?? "Microlink screenshot failed");
  }

  const screenshotUrl = payload?.data?.screenshot?.url;
  if (!screenshotUrl) {
    throw new Error("Microlink returned no screenshot URL");
  }

  const imageResponse = await fetchWithTimeout(screenshotUrl);
  if (!imageResponse.ok) {
    throw new Error(`Microlink image HTTP ${imageResponse.status}`);
  }

  const contentType = imageResponse.headers.get("content-type") ?? "image/png";
  const buffer = Buffer.from(await imageResponse.arrayBuffer());
  return { buffer, contentType, provider: "microlink", sourceUrl: screenshotUrl };
}

export async function takePageScreenshot(pageUrl) {
  const provider = getScreenshotProvider();

  if (provider === "screenshotone") {
    return takeScreenshotOne(pageUrl);
  }

  return takeMicrolink(pageUrl);
}
