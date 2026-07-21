// src/app/resume-workshop/WorkshopForm.js

"use client";

import { useActionState } from "react";
import { submitWorkshopRequest } from "./actions";

export default function WorkshopForm() {
  const [state, formAction, pending] = useActionState(
    submitWorkshopRequest,
    null,
  );

  if (state?.ok) {
    return (
      <p className="neo-facts mt-4 p-4" style={{ border: "2px solid #1a1a1a" }}>
        📬 Got it! I&apos;ll get back to you on your preferred channel within a
        few days.
      </p>
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
      </label>

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
