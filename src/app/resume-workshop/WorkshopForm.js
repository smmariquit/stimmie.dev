// src/app/resume-workshop/WorkshopForm.js

"use client";

import { useActionState, useState } from "react";
import { submitWorkshopRequest } from "./actions";

const CAL_URL = "https://cal.com/simonee";

export default function WorkshopForm() {
  const [state, formAction, pending] = useActionState(
    submitWorkshopRequest,
    null,
  );
  const [wantsCall, setWantsCall] = useState(false);

  if (state?.ok) {
    return (
      <div
        className="neo-facts mt-4 p-4"
        style={{ border: "2px solid #1a1a1a" }}
      >
        <p>
          📬 Got it! I&apos;ll get back to you on your preferred channel within
          a few days.
        </p>
        {wantsCall && (
          <p className="mt-2 font-bold">
            📅 One more step: book your call at{" "}
            <a href={CAL_URL} target="_blank" rel="noopener noreferrer">
              cal.com/simonee
            </a>
            .
          </p>
        )}
      </div>
    );
  }

  return (
    <form action={formAction} className="mt-4 space-y-3">
      <input
        type="text"
        name="website"
        tabIndex={-1}
        autoComplete="off"
        aria-hidden="true"
        style={{ display: "none" }}
      />

      <label className="block font-bold" htmlFor="rw-name">
        Name
        <input
          id="rw-name"
          name="name"
          type="text"
          required
          maxLength={200}
          className="neo-input mt-1 font-normal"
        />
      </label>

      <label className="block font-bold" htmlFor="rw-contact">
        Where do I reach you? (email, Messenger, or Discord)
        <input
          id="rw-contact"
          name="contact"
          type="text"
          required
          maxLength={300}
          placeholder="you@example.com / m.me handle / discord username"
          className="neo-input mt-1 font-normal"
        />
      </label>

      <label className="block font-bold" htmlFor="rw-need">
        What do you need?
        <select id="rw-need" name="need" className="neo-input mt-1 font-normal">
          <option>Resume review</option>
          <option>Portfolio review</option>
          <option>Interview help</option>
          <option>Something else</option>
        </select>
      </label>

      <label className="block font-bold" htmlFor="rw-link">
        Link to your resume or portfolio (optional)
        <input
          id="rw-link"
          name="link"
          type="url"
          maxLength={1000}
          placeholder="Google Drive, PDF, or website link"
          className="neo-input mt-1 font-normal"
        />
        <span className="block text-base neo-muted mt-1 font-normal">
          ⚠ Make sure the link is <strong>accessible</strong>! On Google Drive:
          Share, then set &quot;Anyone with the link&quot; to Viewer. I
          can&apos;t review what I can&apos;t open.
        </span>
      </label>

      <fieldset className="m-0 p-0" style={{ border: "none" }}>
        <legend className="font-bold p-0">How do you want the feedback?</legend>
        <label className="block mt-1">
          <input
            type="radio"
            name="feedback"
            value="Email or DM"
            defaultChecked
            onChange={() => setWantsCall(false)}
          />{" "}
          ✉ Just email or DM me the feedback
        </label>
        <label className="block mt-1">
          <input
            type="radio"
            name="feedback"
            value="Call"
            onChange={() => setWantsCall(true)}
          />{" "}
          ☎ Let&apos;s talk it through on a call
        </label>
        {wantsCall && (
          <p className="mt-2">
            📅 Book a slot at{" "}
            <a href={CAL_URL} target="_blank" rel="noopener noreferrer">
              cal.com/simonee
            </a>{" "}
            (I&apos;ll remind you after you submit too).
          </p>
        )}
      </fieldset>

      <label className="block font-bold" htmlFor="rw-notes">
        Anything else? (optional)
        <textarea
          id="rw-notes"
          name="notes"
          rows={3}
          maxLength={5000}
          className="neo-input mt-1 font-normal"
        />
      </label>

      {state?.error && (
        <p role="alert" className="font-bold" style={{ color: "#c40000" }}>
          {state.error}
        </p>
      )}

      <button
        type="submit"
        disabled={pending}
        className="neo-link-card inline-block font-bold cursor-pointer"
      >
        {pending ? "sending..." : "📨 send it over"}
      </button>
    </form>
  );
}
