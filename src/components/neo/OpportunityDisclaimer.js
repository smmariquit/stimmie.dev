export default function OpportunityDisclaimer() {
  return (
    <aside
      className="neo-opportunity-disclaimer"
      aria-labelledby="opportunity-disclaimer-heading"
    >
      <h2 id="opportunity-disclaimer-heading" className="neo-opportunity-disclaimer-title">
        Read this before you apply
      </h2>
      <ul className="neo-opportunity-disclaimer-list">
        <li>
          This is a <strong>personal, incomplete roundup</strong> — not an
          official listing, job board, or endorsement of any organizer.
        </li>
        <li>
          Details go stale fast.{" "}
          <strong>Verify everything on the official site</strong> — deadlines,
          eligibility, fees, and whether applications are still open.
        </li>
        <li>
          I am <strong>not affiliated</strong> with the organizations listed
          unless explicitly stated. Links go to third-party sites I do not
          control.
        </li>
        <li>
          <strong>You are responsible</strong> for your own research, applications,
          and decisions. Do not rely on this page as your only source.
        </li>
        <li>
          No guarantees on accuracy or outcomes. If something looks wrong or
          expired, assume the official page is correct — or{" "}
          <a href="mailto:hello@stimmie.dev">tell me</a> and I will fix it when
          I can.
        </li>
      </ul>
    </aside>
  );
}
