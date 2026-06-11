// src/app/layout.js

import { Analytics } from "@vercel/analytics/react";
import "./globals.css";

export const metadata = {
  // Assumes deployment at https://stimmie.dev — update if different
  metadataBase: new URL("https://stimmie.dev"),
  title: {
    default: "Stimmie - Software Engineer",
    template: "%s | Stimmie",
  },
  description: "Hi! I'm Stimmie, a software engineer with a lifelong passion for technology. Portfolio showcasing projects, talks, and experiences.",
  keywords: ["software engineer", "web developer", "data science", "portfolio", "Stimmie", "Philippines"],
  authors: [{ name: "Stimmie", url: "https://stimmie.dev" }],
  creator: "Stimmie",
  icons: {
    icon: "/icon.png",
    shortcut: "/icon.png",
    apple: "/icon.png",
  },
  openGraph: {
    title: "Stimmie - Software Engineer",
    description: "Hi! I'm Stimmie, a software engineer with a lifelong passion for technology. Portfolio showcasing projects, talks, and experiences.",
    url: "https://stimmie.dev",
    siteName: "Stimmie",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Stimmie - Software Engineer Portfolio",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Stimmie - Software Engineer",
    description: "Hi! I'm Stimmie, a software engineer with a lifelong passion for technology.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

export const viewport = {
  themeColor: "#000000",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`antialiased`}
      >
        {children}
        <Analytics />
      </body>
    </html>
  );
}
