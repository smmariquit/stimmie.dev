"use client";

import { useEffect } from "react";
import { initDatadogRum } from "@/lib/datadog-rum";

export function DatadogInit() {
  useEffect(() => {
    initDatadogRum();
  }, []);

  return null;
}
