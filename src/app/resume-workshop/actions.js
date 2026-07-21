// src/app/resume-workshop/actions.js

"use server";

const TO = "semariquit@gmail.com";
// Prefer a verified sender from env (e.g. noreply@uplbtools.me). The Resend
// sandbox fallback only delivers to the account's own email, which is still
// fine here since that is where submissions go.
const FROM =
  process.env.RESEND_FROM_EMAIL ||
  "stimmie.dev workshop <onboarding@resend.dev>";

const FALLBACK_ERROR =
  "The form isn't working right now. Message me directly instead, the links are right below!";

export async function submitWorkshopRequest(_prevState, formData) {
  // Honeypot: humans never see this field.
  if (formData.get("website")) return { ok: true };

  const field = (key, max) =>
    String(formData.get(key) || "")
      .trim()
      .slice(0, max);

  const name = field("name", 200);
  const contact = field("contact", 300);
  const need = field("need", 100);
  const link = field("link", 1000);
  const notes = field("notes", 5000);

  if (!name || !contact) {
    return { error: "I need your name and a way to reach you." };
  }

  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) return { error: FALLBACK_ERROR };

  const text = [
    `Name: ${name}`,
    `Contact: ${contact}`,
    `Needs: ${need || "(not specified)"}`,
    `Link: ${link || "(none)"}`,
    "",
    notes || "(no notes)",
  ].join("\n");

  try {
    const response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: FROM,
        to: [TO],
        subject: `Resume workshop: ${name}`,
        text,
      }),
    });
    if (!response.ok) return { error: FALLBACK_ERROR };
  } catch {
    return { error: FALLBACK_ERROR };
  }

  return { ok: true };
}
