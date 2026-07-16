import Link from "next/link";

export default function LegalMicroLink({
  href,
  children,
  subtle = true,
  className = "",
}) {
  const styles = subtle
    ? "text-[10px] opacity-40 hover:opacity-70"
    : "text-xs opacity-70 underline-offset-2 hover:underline";

  return (
    <Link href={href} className={`transition-opacity ${styles} ${className}`}>
      {children}
    </Link>
  );
}
