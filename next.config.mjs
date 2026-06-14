// next.config.mjs

/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
  reactCompiler: true,

  images: {
    // Next 16 requires every requested `quality` to be allow-listed here,
    // otherwise the image optimizer responds 400 and the image fails to load.
    qualities: [75, 90, 100],
    // Allow Next/Image to serve our local brand-icon SVGs (e.g. /logos/discord.svg).
    // The CSP keeps SVGs sandboxed and blocks any inline script execution, so
    // this is safe for the trusted assets we ship from /public/logos.
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },

  env: {
    NEXT_PUBLIC_BUILD_DATE:
      process.env.NEXT_PUBLIC_BUILD_DATE ?? new Date().toISOString(),
  },
};

export default nextConfig;
