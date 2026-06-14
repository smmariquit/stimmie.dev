"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

// Live status for The Crib, my survival Minecraft server. We ping the public
// mcsrvstat.us API for mc.stimmie.dev (crib.stimmie.dev is just the website, not
// the game server). The page is statically cached, so we fetch client-side to
// keep the "currently online" count actually current.
const STATUS_URL = "https://api.mcsrvstat.us/3/mc.stimmie.dev";

export default function CribStatus() {
  // "loading" | "online" | "offline" | "error"
  const [state, setState] = useState("loading");
  const [players, setPlayers] = useState(null);

  useEffect(() => {
    let active = true;

    fetch(STATUS_URL, { cache: "no-store" })
      .then((res) =>
        res.ok ? res.json() : Promise.reject(new Error("bad status")),
      )
      .then((data) => {
        if (!active) return;
        if (data?.online) {
          setPlayers({
            online: data.players?.online ?? 0,
            max: data.players?.max ?? null,
          });
          setState("online");
        } else {
          setState("offline");
        }
      })
      .catch(() => {
        if (active) setState("error");
      });

    return () => {
      active = false;
    };
  }, []);

  let body;
  let announce;
  if (state === "online" && players) {
    const count = players.online;
    body = (
      <>
        <span className="neo-crib-count">{count.toLocaleString()}</span>{" "}
        {count === 1 ? "player" : "players"} online
        {players.max != null && (
          <span className="neo-crib-max"> / {players.max}</span>
        )}
      </>
    );
    announce = `The Crib is online with ${count} ${
      count === 1 ? "player" : "players"
    } currently playing.`;
  } else if (state === "offline") {
    body = "server is offline right now";
    announce = "The Crib server is offline right now.";
  } else if (state === "error") {
    body = "status unavailable";
    announce = "The Crib server status is unavailable.";
  } else {
    body = "checking server\u2026";
    announce = "Checking The Crib server status.";
  }

  const dotState = state === "online" ? "online" : "offline";

  return (
    <div>
      <h2 className="neo-sidebar-heading neo-accent-blue">~ the crib ~</h2>
      <p
        className="neo-crib-status"
        role="status"
        aria-live="polite"
        aria-label={announce}
      >
        <span
          className={`neo-status-dot neo-status-dot-${dotState}`}
          aria-hidden="true"
        />
        <span>{body}</span>
      </p>
      <p className="neo-crib-blurb">my survival Minecraft server that never resets</p>
      <Link
        href="https://crib.stimmie.dev"
        target="_blank"
        rel="noopener noreferrer"
      >
        visit the crib ↗
      </Link>
    </div>
  );
}
