"use client";

import Image from "next/image";
import { useState } from "react";
import { OPPORTUNITY_UNAVAILABLE_IMAGE } from "@/data/opportunities";

export default function OpportunityCoverImage({ src, alt, className = "" }) {
  const [failed, setFailed] = useState(false);
  const isPlaceholder = failed || src === OPPORTUNITY_UNAVAILABLE_IMAGE;
  const displaySrc = failed ? OPPORTUNITY_UNAVAILABLE_IMAGE : src;

  return (
    <Image
      src={displaySrc}
      alt={isPlaceholder ? "No preview available" : alt}
      width={800}
      height={450}
      quality={90}
      sizes="(max-width: 640px) 100vw, 360px"
      className={`neo-thumb-lg w-full aspect-video object-cover${isPlaceholder ? " neo-opportunity-placeholder-image" : ""}${className ? ` ${className}` : ""}`}
      onError={() => {
        if (!failed) {
          setFailed(true);
        }
      }}
    />
  );
}
