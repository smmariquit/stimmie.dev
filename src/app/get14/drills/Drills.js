// src/app/get14/drills/Drills.js

"use client";

import { useEffect, useRef, useState } from "react";
import bank from "@/data/get14-drills.json";
import { FilterGroup, shuffled } from "../Trainer";

const ROUND_MS = 60000;
const FAST_MS = 5000;
const METER = 10;
// The ranks borrow the names of the contest rounds.
const RANKS = [
  [450, "Vivace"],
  [250, "Allegro"],
  [100, "Andante"],
  [0, "Adagio"],
];
const LANG_LABEL = {
  python: "Python",
  js: "JavaScript",
  c: "C",
  any: "any language",
};
const KIND_LABEL = {
  math: "arithmetic",
  skills: "skills",
  memory: "memory",
  traps: "traps",
};
const EMPTY = {
  score: 0,
  gain: 0,
  combo: 0,
  bestCombo: 0,
  right: 0,
  asked: 0,
  missed: [],
};

// Spaces and quotes don't count, so [1,2] matches [ 1, 2 ].
function clean(text) {
  return text.toLowerCase().replace(/[\s'"]/g, "");
}

function multiplier(combo) {
  return 1 + Math.min(3, Math.floor(combo / 3));
}

function CountUp({ value }) {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(value);
      return;
    }
    const start = performance.now();
    let frame = requestAnimationFrame(function step(now) {
      const t = Math.min(1, (now - start) / 900);
      setShown(Math.round(value * (1 - (1 - t) ** 3)));
      if (t < 1) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [value]);

  return shown;
}

function Tags({ item }) {
  return (
    <span>
      <span className="neo-tag">{LANG_LABEL[item[1]]}</span>
      <span className="neo-tag">{KIND_LABEL[item[0]]}</span>
    </span>
  );
}

export default function Drills() {
  const [kind, setKind] = useState("all");
  const [lang, setLang] = useState("all");
  const [phase, setPhase] = useState("menu");
  const [deck, setDeck] = useState([]);
  const [pos, setPos] = useState(0);
  const [timeLeft, setTimeLeft] = useState(ROUND_MS);
  const [text, setText] = useState("");
  const [miss, setMiss] = useState(false);
  const [stats, setStats] = useState(EMPTY);
  const [bests, setBests] = useState({});
  const [prevBest, setPrevBest] = useState(0);
  const inputRef = useRef(null);
  const askedAt = useRef(0);

  const view = phase === "play" && timeLeft === 0 ? "over" : phase;
  const bestKey = `${kind}-${lang}`;
  const item = deck[pos];
  const mult = multiplier(stats.combo);
  const seconds = Math.ceil(timeLeft / 1000);

  useEffect(() => {
    try {
      setBests(JSON.parse(localStorage.getItem("g14d-best")) ?? {});
    } catch {
      // Private windows can block storage, and the game works without it.
    }
  }, []);

  // The clock stops while a missed answer is on screen.
  useEffect(() => {
    if (phase !== "play" || miss) return;
    let last = performance.now();
    const tick = setInterval(() => {
      const now = performance.now();
      const passed = now - last;
      last = now;
      setTimeLeft((t) => Math.max(0, t - passed));
    }, 100);
    return () => clearInterval(tick);
  }, [phase, miss]);

  useEffect(() => {
    if (view !== "over" || stats.score <= (bests[bestKey] ?? 0)) return;
    const next = { ...bests, [bestKey]: stats.score };
    setBests(next);
    try {
      localStorage.setItem("g14d-best", JSON.stringify(next));
    } catch {
      // Private windows can block storage.
    }
  }, [view, stats.score, bests, bestKey]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: refocus on each new question
  useEffect(() => {
    if (view === "play") inputRef.current?.focus();
  }, [view, pos]);

  function start() {
    const pool = bank.items.filter(
      ([group, itemLang]) =>
        (kind === "all" || group === kind) &&
        (lang === "all" || itemLang === lang || itemLang === "any"),
    );
    setDeck(shuffled(pool));
    setPos(0);
    setTimeLeft(ROUND_MS);
    setText("");
    setMiss(false);
    setStats(EMPTY);
    setPrevBest(bests[bestKey] ?? 0);
    setPhase("play");
    askedAt.current = performance.now();
  }

  function advance() {
    setPos((p) => (p + 1) % deck.length);
    setText("");
    askedAt.current = performance.now();
  }

  function submit(event) {
    event.preventDefault();
    if (miss) {
      setMiss(false);
      advance();
      return;
    }
    const typed = text.trim();
    if (!typed) return;
    if (clean(typed) === clean(item[3])) {
      const fast = performance.now() - askedAt.current < FAST_MS;
      setStats((s) => {
        const gain = 10 * multiplier(s.combo) + (fast ? 5 : 0);
        return {
          ...s,
          score: s.score + gain,
          gain,
          combo: s.combo + 1,
          bestCombo: Math.max(s.bestCombo, s.combo + 1),
          right: s.right + 1,
          asked: s.asked + 1,
        };
      });
      advance();
    } else {
      setStats((s) => ({
        ...s,
        combo: 0,
        asked: s.asked + 1,
        missed: [...s.missed, { item, typed }],
      }));
      setMiss(true);
    }
  }

  if (view === "menu") {
    return (
      <div className="g14 g14d">
        <div className="g14-filters">
          <FilterGroup
            label="drills"
            value={kind}
            onPick={setKind}
            options={[
              { value: "all", label: "all" },
              ...Object.entries(KIND_LABEL).map(([value, label]) => ({
                value,
                label,
              })),
            ]}
          />
          <FilterGroup
            label="language"
            value={lang}
            onPick={setLang}
            options={[
              { value: "all", label: "all" },
              { value: "python", label: "Python" },
              { value: "js", label: "JavaScript" },
              { value: "c", label: "C" },
            ]}
          />
        </div>

        <ol className="neo-facts list-decimal pl-6 m-0 mt-4 space-y-1">
          <li>A round is 60 seconds. Type what each line prints or equals.</li>
          <li>Every third right answer in a row raises the multiplier.</li>
          <li>A miss pauses the clock so you can read the rule.</li>
        </ol>

        <div className="g14-actions">
          <button
            type="button"
            className="g14-btn g14-btn--main"
            onClick={start}
          >
            start a round
          </button>
        </div>

        <p className="g14-score">
          {bests[bestKey]
            ? `Your best for these drills is ${bests[bestKey]} dB.`
            : `There are ${bank.items.length} drills, and every answer was checked against a real run.`}
        </p>
      </div>
    );
  }

  if (view === "over") {
    const rankAt = RANKS.findIndex(([min]) => stats.score >= min);
    const nextRank = RANKS[rankAt - 1];
    return (
      <div className="g14 g14d">
        <section aria-labelledby="g14d-over-title">
          <h2 id="g14d-over-title" className="neo-section-title">
            time&apos;s up
          </h2>
          <p className="g14d-score g14d-score--final">
            <CountUp value={stats.score} />
            <span className="g14d-unit"> dB</span>
          </p>
          <p className="mt-2">
            You played at <strong>{RANKS[rankAt][1]}</strong>.
            {nextRank && ` ${nextRank[1]} starts at ${nextRank[0]} dB.`}
          </p>
          <p>
            You got {stats.right} of {stats.asked} right, and your longest
            streak was {stats.bestCombo}.
          </p>
          {stats.score > prevBest ? (
            <p className="g14d-newbest">
              That&apos;s your new best for these drills.
            </p>
          ) : (
            prevBest > 0 && <p>Your best for these drills is {prevBest} dB.</p>
          )}

          <div className="g14-actions">
            <button
              type="button"
              className="g14-btn g14-btn--main"
              onClick={start}
            >
              play again
            </button>
            <button
              type="button"
              className="g14-btn"
              onClick={() => setPhase("menu")}
            >
              change drills
            </button>
          </div>
        </section>

        {stats.missed.length > 0 && (
          <section aria-labelledby="g14d-missed-title">
            <h2 id="g14d-missed-title" className="neo-section-title">
              what you missed
            </h2>
            <ol className="g14d-missed">
              {stats.missed.map(({ item: missed, typed }, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: the list never reorders
                <li key={index}>
                  <Tags item={missed} />
                  <pre className="g14d-code">{missed[2]}</pre>
                  <p>
                    The answer is <code>{missed[3]}</code>, and you typed{" "}
                    <code>{typed}</code>. {bank.rules[missed[4]]}
                  </p>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="g14 g14d">
      <section aria-label="drill round">
        <div className="g14d-hud">
          <p className="g14d-score">
            <span key={stats.score} className="g14d-bump">
              {stats.score}
            </span>
            <span className="g14d-unit"> dB</span>
            {stats.gain > 0 && (
              <span key={stats.right} className="g14d-gain" aria-hidden="true">
                +{stats.gain}
              </span>
            )}
          </p>
          <div className="g14d-combo">
            <span key={mult} className="g14d-mult">
              ×{mult}
            </span>
            <span className="g14d-meter" aria-hidden="true">
              {Array.from({ length: METER }, (_, index) => (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: fixed set of bars
                  key={index}
                  className={index < stats.combo ? "is-lit" : undefined}
                />
              ))}
            </span>
            <span className="sr-only">streak of {stats.combo}</span>
          </div>
        </div>

        <div
          className="g14d-time"
          role="progressbar"
          aria-label="seconds left"
          aria-valuemin={0}
          aria-valuemax={ROUND_MS / 1000}
          aria-valuenow={seconds}
        >
          <span
            className={timeLeft < 10000 ? "is-low" : undefined}
            style={{ transform: `scaleX(${timeLeft / ROUND_MS})` }}
          />
        </div>

        <div key={pos} className={miss ? "g14d-card is-wrong" : "g14d-card"}>
          <div className="g14-meta">
            <Tags item={item} />
            <span className="g14-clock">{seconds}s</span>
          </div>
          <pre
            className={
              item[2].length <= 20 ? "g14d-code g14d-code--big" : "g14d-code"
            }
          >
            {item[2]}
          </pre>
        </div>

        <form onSubmit={submit} className="g14d-form">
          <label htmlFor="g14d-answer" className="sr-only">
            Your answer
          </label>
          <input
            id="g14d-answer"
            ref={inputRef}
            className="g14d-input"
            value={text}
            readOnly={miss}
            placeholder="your answer"
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            autoComplete="off"
            onChange={(event) => setText(event.target.value)}
          />
          <button type="submit" className="g14-btn g14-btn--main">
            {miss ? "continue" : "enter"}
          </button>
        </form>

        <div role="status" aria-live="polite">
          {miss && (
            <div className="g14-feedback g14-feedback--wrong">
              <p>
                <strong>
                  The answer is <code>{item[3]}</code>.
                </strong>{" "}
                {bank.rules[item[4]]}
              </p>
              <p className="neo-muted text-base mt-1">
                The clock is paused. Press Enter to continue.
              </p>
            </div>
          )}
        </div>

        <p className="mt-4">
          <button
            type="button"
            className="g14-reset"
            onClick={() => setTimeLeft(0)}
          >
            end the round
          </button>
        </p>
      </section>
    </div>
  );
}
