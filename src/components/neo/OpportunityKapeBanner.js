const KAPE_URL = "https://kape.stimmie.dev";

export default function OpportunityKapeBanner() {
  return (
    <aside className="neo-kape-banner mb-4" aria-label="Support this roundup">
      <p className="neo-kape-banner-text m-0">
        This roundup is free to browse and share. If it helps you,{" "}
        <a
          href={KAPE_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="neo-kape-banner-link"
        >
          ☕ kape.stimmie.dev
        </a>{" "}
        keeps it running.
      </p>
    </aside>
  );
}
