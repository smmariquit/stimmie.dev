// src/app/get14/Trainer.js

"use client";

import { useEffect, useRef, useState } from "react";
import { LANGS, LEVELS, normalize, problems } from "@/data/get14-problems";

const BY_ID = Object.fromEntries(problems.map((p) => [p.id, p]));
const LANG_LABEL = Object.fromEntries(LANGS.map((l) => [l.id, l.label]));
const KINDS = [
  { value: "all", label: "all" },
  { value: "trace", label: "trace" },
  { value: "blank", label: "fill the blank" },
  { value: "write", label: "write code" },
];
const kindOf = (p) => p.kind ?? "trace";
// Spaces and quotes don't count in a blank, so n*n matches n * n.
const squash = (text) => text.replace(/[\s'"]/g, "");

// Easy first, with the three languages taking turns inside each level.
const DEFAULT_ORDER = (() => {
  const seen = {};
  return problems
    .map((p) => {
      const key = `${p.lang}-${p.level}`;
      seen[key] = (seen[key] ?? 0) + 1;
      return { id: p.id, level: p.level, turn: seen[key] };
    })
    .sort((a, b) => a.level - b.level || a.turn - b.turn)
    .map((p) => p.id);
})();

export function shuffled(ids) {
  const out = [...ids];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function clock(seconds) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function FilterGroup({ label, options, value, onPick }) {
  return (
    // biome-ignore lint/a11y/useSemanticElements: a fieldset can't wrap flex chips cleanly
    <div className="g14-filter" role="group" aria-label={label}>
      <span className="g14-filter-label" aria-hidden="true">
        {label}
      </span>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="g14-chip"
          aria-pressed={value === option.value}
          onClick={() => onPick(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function Problem({ problem, number, total, focusAnswer, onFirstTry, onNext }) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("open");
  const [tries, setTries] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const answerRef = useRef(null);
  const nextRef = useRef(null);
  const done = status === "right" || status === "shown";

  useEffect(() => {
    if (focusAnswer) answerRef.current?.focus();
  }, [focusAnswer]);

  useEffect(() => {
    if (done) {
      nextRef.current?.focus();
      return;
    }
    const tick = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(tick);
  }, [done]);

  function check(event) {
    event.preventDefault();
    if (!text.trim() || done) return;
    const right = normalize(text) === normalize(problem.answer);
    if (tries === 0) onFirstTry(problem.id, right);
    setTries(tries + 1);
    setStatus(right ? "right" : "wrong");
  }

  function reveal() {
    if (tries === 0) onFirstTry(problem.id, false);
    setStatus("shown");
  }

  return (
    <section aria-labelledby="g14-problem-title">
      <h2 id="g14-problem-title" className="neo-section-title">
        problem {number} of {total}
      </h2>
      <div className="g14-meta">
        <span>
          <span className="neo-tag">{LANG_LABEL[problem.lang]}</span>
          <span className="neo-tag">{LEVELS[problem.level - 1]}</span>
        </span>
        <span className="g14-clock">
          <span className="sr-only">time on this problem </span>
          {clock(seconds)}
        </span>
      </div>

      <ol
        className="g14-code"
        aria-label={`${LANG_LABEL[problem.lang]} code to trace`}
      >
        {problem.code.split("\n").map((line, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: lines never reorder
          <li key={index}>{line}</li>
        ))}
      </ol>

      <form onSubmit={check} className="mt-4">
        <label className="block font-bold" htmlFor="g14-answer">
          What does it print?
        </label>
        <textarea
          id="g14-answer"
          ref={answerRef}
          className="g14-sheet mt-1"
          rows={4}
          value={text}
          readOnly={done}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          autoComplete="off"
          aria-describedby="g14-hint"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              // Keeps this same key press from also clicking "next problem".
              event.preventDefault();
              event.currentTarget.form.requestSubmit();
            }
          }}
        />
        <p id="g14-hint" className="neo-muted text-base mt-1">
          Spaces and line breaks count. Ctrl + Enter checks your answer.
        </p>
        <div className="g14-actions">
          {done ? (
            <button
              type="button"
              ref={nextRef}
              className="g14-btn g14-btn--main"
              onClick={onNext}
            >
              {number === total ? "back to problem 1" : "next problem →"}
            </button>
          ) : (
            <>
              <button
                type="submit"
                className="g14-btn g14-btn--main"
                disabled={!text.trim()}
              >
                check
              </button>
              <button type="button" className="g14-btn" onClick={reveal}>
                show answer
              </button>
            </>
          )}
        </div>
      </form>

      <div role="status" aria-live="polite">
        {status === "wrong" && (
          <p className="g14-feedback g14-feedback--wrong">
            <strong>Not quite.</strong> That was try {tries}. Trace it once
            more, or show the answer.
          </p>
        )}
        {status === "right" && (
          <div className="g14-feedback g14-feedback--right">
            <p>
              <strong>Correct!</strong> You took {clock(seconds)}.
            </p>
            <p className="mt-1">{problem.why}</p>
          </div>
        )}
        {status === "shown" && (
          <div className="g14-feedback">
            <p>
              <strong>It prints this.</strong>
            </p>
            <pre className="g14-answer">{problem.answer}</pre>
            <p className="mt-2">{problem.why}</p>
          </div>
        )}
      </div>
    </section>
  );
}

function CodeProblem({
  problem,
  number,
  total,
  focusAnswer,
  onFirstTry,
  onNext,
}) {
  const write = problem.kind === "write";
  const [text, setText] = useState(write ? problem.starter : "");
  const [status, setStatus] = useState("open");
  const [tries, setTries] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const answerRef = useRef(null);
  const nextRef = useRef(null);
  const stopped = status !== "open" && status !== "wrong";
  const canNext =
    status === "right" || status === "graded" || (status === "shown" && !write);

  useEffect(() => {
    if (focusAnswer) answerRef.current?.focus();
  }, [focusAnswer]);

  useEffect(() => {
    if (canNext) {
      nextRef.current?.focus();
      return;
    }
    if (stopped) return;
    const tick = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(tick);
  }, [canNext, stopped]);

  function check(event) {
    event.preventDefault();
    if (write || !text.trim() || stopped) return;
    const right = [problem.answer, ...(problem.accept ?? [])].some(
      (answer) => squash(answer) === squash(text),
    );
    if (tries === 0) onFirstTry(problem.id, right);
    setTries(tries + 1);
    setStatus(right ? "right" : "wrong");
  }

  function reveal() {
    if (!write && tries === 0) onFirstTry(problem.id, false);
    setStatus("shown");
  }

  function grade(right) {
    onFirstTry(problem.id, right);
    setStatus("graded");
  }

  return (
    <section aria-labelledby="g14-problem-title">
      <h2 id="g14-problem-title" className="neo-section-title">
        problem {number} of {total}
      </h2>
      <div className="g14-meta">
        <span>
          <span className="neo-tag">{LANG_LABEL[problem.lang]}</span>
          <span className="neo-tag">{LEVELS[problem.level - 1]}</span>
          <span className="neo-tag">
            {write ? "write code" : "fill the blank"}
          </span>
        </span>
        <span className="g14-clock">
          <span className="sr-only">time on this problem </span>
          {clock(seconds)}
        </span>
      </div>

      <p className="mt-2 font-bold">{problem.task}</p>

      {write ? (
        <>
          <p className="mt-2">Your function has to pass these tests.</p>
          <ul className="neo-facts list-disc pl-6 m-0 mt-1 space-y-1">
            {problem.tests.map(([expr, out]) => (
              <li key={expr}>
                <code>{expr}</code> gives <strong>{out}</strong>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <ol
          className="g14-code"
          aria-label={`${LANG_LABEL[problem.lang]} code with a blank`}
        >
          {problem.code.split("\n").map((line, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: lines never reorder
            <li key={index}>{line}</li>
          ))}
        </ol>
      )}

      <form onSubmit={check} className="mt-4">
        <label className="block font-bold" htmlFor="g14-answer">
          {write
            ? "Write it on paper first, then here."
            : "What goes in the blank?"}
        </label>
        <textarea
          id="g14-answer"
          ref={answerRef}
          className="g14-sheet mt-1"
          rows={write ? 9 : 2}
          value={text}
          readOnly={stopped}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          autoComplete="off"
          aria-describedby="g14-hint"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (!write && event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form.requestSubmit();
            }
          }}
        />
        <p id="g14-hint" className="neo-muted text-base mt-1">
          {write
            ? "You grade this one yourself. Compare it with the model solution and run your tests on paper."
            : "Spaces and quotes don't count. Enter checks your answer."}
        </p>
        <div className="g14-actions">
          {canNext ? (
            <button
              type="button"
              ref={nextRef}
              className="g14-btn g14-btn--main"
              onClick={onNext}
            >
              {number === total ? "back to problem 1" : "next problem →"}
            </button>
          ) : write ? (
            status === "shown" ? (
              <>
                <button
                  type="button"
                  className="g14-btn g14-btn--main"
                  onClick={() => grade(true)}
                >
                  mine works
                </button>
                <button
                  type="button"
                  className="g14-btn"
                  onClick={() => grade(false)}
                >
                  i missed it
                </button>
              </>
            ) : (
              <button
                type="button"
                className="g14-btn g14-btn--main"
                onClick={reveal}
              >
                show solution
              </button>
            )
          ) : (
            <>
              <button
                type="submit"
                className="g14-btn g14-btn--main"
                disabled={!text.trim()}
              >
                check
              </button>
              <button type="button" className="g14-btn" onClick={reveal}>
                show answer
              </button>
            </>
          )}
        </div>
      </form>

      <div role="status" aria-live="polite">
        {status === "wrong" && (
          <p className="g14-feedback g14-feedback--wrong">
            <strong>Not quite.</strong> That was try {tries}. Check the goal
            above, or show the answer.
          </p>
        )}
        {status === "right" && (
          <div className="g14-feedback g14-feedback--right">
            <p>
              <strong>Correct!</strong> You took {clock(seconds)}.
            </p>
            <p className="mt-1">{problem.why}</p>
          </div>
        )}
        {status === "shown" && !write && (
          <div className="g14-feedback">
            <p>
              <strong>The blank is</strong> <code>{problem.answer}</code>.
            </p>
            <p className="mt-2">{problem.why}</p>
          </div>
        )}
        {write && (status === "shown" || status === "graded") && (
          <div className="g14-feedback">
            <p>
              <strong>One model solution.</strong> Yours can look different if
              it passes the same tests.
            </p>
            <pre className="g14-answer">{problem.solution}</pre>
            <p className="mt-2">{problem.why}</p>
          </div>
        )}
      </div>
    </section>
  );
}

function ProblemView(props) {
  return props.problem.kind ? (
    <CodeProblem {...props} />
  ) : (
    <Problem {...props} />
  );
}

export default function Trainer() {
  const [lang, setLang] = useState("all");
  const [level, setLevel] = useState(0);
  const [kind, setKind] = useState("all");
  const [order, setOrder] = useState(DEFAULT_ORDER);
  const [pos, setPos] = useState(0);
  const [results, setResults] = useState({});
  const [focusAnswer, setFocusAnswer] = useState(false);

  const deck = order
    .map((id) => BY_ID[id])
    .filter(
      (p) =>
        (lang === "all" || p.lang === lang) &&
        (!level || p.level === level) &&
        (kind === "all" || kindOf(p) === kind),
    );
  const problem = deck[pos];
  const tried = Object.keys(results).length;
  const right = Object.values(results).filter(Boolean).length;

  function restart(change) {
    change();
    setPos(0);
    setFocusAnswer(false);
  }

  return (
    <div className="g14">
      <div className="g14-filters">
        <FilterGroup
          label="language"
          value={lang}
          onPick={(value) => restart(() => setLang(value))}
          options={[
            { value: "all", label: "all" },
            ...LANGS.map((l) => ({ value: l.id, label: l.label })),
          ]}
        />
        <FilterGroup
          label="type"
          value={kind}
          onPick={(value) => restart(() => setKind(value))}
          options={KINDS}
        />
        <FilterGroup
          label="level"
          value={level}
          onPick={(value) => restart(() => setLevel(value))}
          options={[
            { value: 0, label: "all" },
            ...LEVELS.map((name, index) => ({ value: index + 1, label: name })),
          ]}
        />
        <div className="g14-filter">
          <span className="g14-filter-label" aria-hidden="true">
            order
          </span>
          <button
            type="button"
            className="g14-chip"
            aria-pressed={order === DEFAULT_ORDER}
            onClick={() => restart(() => setOrder(DEFAULT_ORDER))}
          >
            easy first
          </button>
          <button
            type="button"
            className="g14-chip"
            aria-pressed={order !== DEFAULT_ORDER}
            onClick={() => restart(() => setOrder(shuffled(DEFAULT_ORDER)))}
          >
            shuffle
          </button>
        </div>
      </div>

      <p className="g14-score">
        {tried === 0 ? (
          <>
            There are {problems.length} problems. Your score shows up here after
            your first check.
          </>
        ) : (
          <>
            You got <strong>{right}</strong> of <strong>{tried}</strong> right
            on the first try.{" "}
            <button
              type="button"
              className="g14-reset"
              onClick={() => setResults({})}
            >
              reset score
            </button>
          </>
        )}
      </p>

      {problem ? (
        <ProblemView
          key={problem.id}
          problem={problem}
          number={pos + 1}
          total={deck.length}
          focusAnswer={focusAnswer}
          onFirstTry={(id, ok) => setResults((old) => ({ ...old, [id]: ok }))}
          onNext={() => {
            setPos((pos + 1) % deck.length);
            setFocusAnswer(true);
          }}
        />
      ) : (
        <p className="mt-6">No problems match these filters.</p>
      )}
    </div>
  );
}
