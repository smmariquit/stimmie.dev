import { datadogRum } from "@datadog/browser-rum";

let initialized = false;

export function initDatadogRum() {
  if (initialized || typeof window === "undefined") return;

  const applicationId = process.env.NEXT_PUBLIC_DD_RUM_APPLICATION_ID;
  const clientToken = process.env.NEXT_PUBLIC_DD_RUM_CLIENT_TOKEN;
  if (!applicationId || !clientToken) return;

  const isProd = process.env.NODE_ENV === "production";
  const env =
    process.env.NEXT_PUBLIC_DD_ENV ?? (isProd ? "production" : "development");

  datadogRum.init({
    applicationId,
    clientToken,
    site: "us5.datadoghq.com",
    service: "stimmie-dev",
    env,
    version: process.env.NEXT_PUBLIC_DD_VERSION ?? undefined,
    sessionSampleRate: isProd ? 15 : 100,
    sessionReplaySampleRate: 0,
    trackUserInteractions: true,
    trackResources: true,
    trackLongTasks: true,
    defaultPrivacyLevel: "mask-user-input",
    trackSessionAcrossSubdomains: false,
    allowedTracingUrls: [{ match: /\/api\//, propagatorTypes: ["tracecontext"] }],
    traceSampleRate: 10,
    beforeSend: (event) => {
      if (event.type === "view" && event.view?.url) {
        try {
          const url = new URL(event.view.url);
          url.search = "";
          url.hash = "";
          event.view.url = url.toString();
        } catch {
          // keep original url
        }
      }
      return true;
    },
  });

  initialized = true;
}
