import { readFileSync } from "node:fs";
import { join } from "node:path";
import Link from "next/link";
import CopyableKey from "@/components/neo/CopyableKey";
import PageShell from "@/components/neo/PageShell";
import { PGP_KEY, SSH_KEY } from "@/data/keys";

export const metadata = {
  title: "Keys",
  description:
    "Stimmie's PGP public key and SSH public key for encrypted mail and host verification.",
};

function readPublicKey(filename) {
  return readFileSync(
    join(process.cwd(), "public", "keys", filename),
    "utf8",
  ).trim();
}

function KeyMeta({ items }) {
  return (
    <dl className="neo-key-meta">
      {items.map(([term, value]) => (
        <div key={term} className="neo-key-meta-row">
          <dt>{term}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function KeysPage() {
  const pgp = readPublicKey("pgp.asc");
  const ssh = readPublicKey("ssh.pub");

  return (
    <PageShell
      title="~ keys ~"
      intro="Public keys for encrypted mail and SSH host verification. Fingerprint-check before you trust."
      current="/keys"
      maxWidth="52rem"
    >
      <section className="mb-8" aria-labelledby="pgp-h">
        <h2 id="pgp-h" className="neo-section-title">
          pgp
        </h2>
        <KeyMeta
          items={[
            ["name", PGP_KEY.name],
            ["email", PGP_KEY.email],
            ["algorithm", PGP_KEY.algorithm],
            ["fingerprint", PGP_KEY.fingerprint],
            ["key id", PGP_KEY.keyId],
          ]}
        />
        <CopyableKey
          text={pgp}
          download={PGP_KEY.download}
          downloadName="stimmie-pgp.asc"
        />
        <p className="neo-key-hint">
          Verify with{" "}
          <code className="neo-section-slug">gpg --import pgp.asc</code> or{" "}
          <Link href={PGP_KEY.download}>{PGP_KEY.download}</Link>.
        </p>
      </section>

      <section aria-labelledby="ssh-h">
        <h2 id="ssh-h" className="neo-section-title">
          ssh
        </h2>
        <KeyMeta
          items={[
            ["key", SSH_KEY.label],
            ["comment", SSH_KEY.comment],
            ["algorithm", SSH_KEY.algorithm],
            ["fingerprint", SSH_KEY.fingerprint],
          ]}
        />
        <CopyableKey
          text={ssh}
          download={SSH_KEY.download}
          downloadName="stimmie-ssh.pub"
        />
        <p className="neo-key-hint">
          Add to <code className="neo-section-slug">~/.ssh/authorized_keys</code>{" "}
          or fetch from <Link href={SSH_KEY.download}>{SSH_KEY.download}</Link>.
        </p>
      </section>
    </PageShell>
  );
}
