import Link from "next/link";

export default function PizzaFriendsPlug() {
  return (
    <div>
      <h2 className="neo-sidebar-heading neo-accent-red">~ pizza &amp; friends ~</h2>
      <p className="neo-crib-blurb">
        Unexpectedly a tech community! Dinner meetups, Discord calls, hackathons,
        and slides parties — we make cool stuff and try not to take it too
        seriously.
      </p>
      <Link
        href="https://joinpizza.fun"
        target="_blank"
        rel="noopener noreferrer"
      >
        join pizza &amp; friends ↗
      </Link>
    </div>
  );
}
