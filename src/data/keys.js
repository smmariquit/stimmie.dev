// Public key metadata. Armored key material lives in public/keys/*.asc|*.pub.

export const PGP_KEY = {
  email: "semariquit@gmail.com",
  name: "Simone Ezekiel Mariquit",
  algorithm: "Ed25519",
  fingerprint: "9688 7E33 DAB8 0C44 C0DC 9B2A 82DF 34F1 45F1 4823",
  keyId: "82DF34F145F14823",
  download: "/keys/pgp.asc",
};

export const SSH_KEY = {
  label: "id_ed25519",
  comment: "stimmie@fedora",
  algorithm: "Ed25519",
  fingerprint: "SHA256:lyQEBYMNP2ZuBOTJRDqvT/XzZWlWCIfhNh9pY5UxAYk",
  download: "/keys/ssh.pub",
};
