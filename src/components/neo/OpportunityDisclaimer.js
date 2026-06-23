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
          This is a <strong>personal, incomplete roundup</strong>, not an
          official listing, job board, or endorsement of any organizer.
        </li>
        <li>
          Details go stale fast.{" "}
          <strong>Verify everything on the official site</strong> for deadlines,
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
          expired, assume the official page is correct, or{" "}
          <a href="mailto:hello@stimmie.dev">tell me</a> and I will fix it when
          I can.
        </li>
      </ul>

      <h3 className="neo-opportunity-disclaimer-subtitle">
        If you RSVP or register for an event
      </h3>
      <ul className="neo-opportunity-disclaimer-list">
        <li>
          A listing here does <strong>not</strong> mean registration is still
          open. Many events fill up, go waitlist-only, or close early.{" "}
          <strong>Check the official page</strong> before you plan around it.
        </li>
        <li>
          Waitlists are not invitations. If you are waitlisted, on a standby
          list, or still pending approval, <strong>do not treat that as a
          confirmed spot</strong>. Showing up unconfirmed adds stress for
          organizers juggling capacity, food, and security lists.
        </li>
        <li>
          <strong>No-shows hurt everyone.</strong> They waste seats someone else
          wanted, skew headcount for catering and swag, and make organizers less
          willing to run open community events. If your plans change, cancel or
          release your spot, or at least message the host.
        </li>
        <li>
          Only attend when you are <strong>officially approved or
          confirmed</strong> (accepted RSVP, ticket, email, or check-in QR,
          whatever the organizer uses). When in doubt, ask before you commute.
        </li>
      </ul>
    </aside>
  );
}
