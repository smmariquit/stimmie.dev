/**
 * 开往 (Travellings) webring badge — required to be visible when the site loads
 * (sidebar on desktop; footer on all viewports). See:
 * https://www.travellings.cn/docs/join
 */
export default function TravellingsBadge({ className = "" }) {
  return (
    <a
      href="https://www.travellings.cn/go.html"
      target="_blank"
      rel="noopener noreferrer"
      title="开往-友链接力"
      className={`neo-travellings-badge inline-flex items-center gap-2 ${className}`.trim()}
    >
      <img
        src="https://www.travellings.cn/assets/logo.gif"
        alt="开往-友链接力"
        width={100}
        height={32}
        loading="lazy"
        className="neo-travellings-logo"
      />
      <span>Travelling</span>
    </a>
  );
}
