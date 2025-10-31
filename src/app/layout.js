import "./globals.css";

export const metadata = {
  // Assumes deployment at https://stimmie.dev — update if different
  metadataBase: new URL("https://stimmie.dev"),
  title: {
    default: "Stimmie",
    template: "%s | Stimmie",
  },
  description: "A portfolio showcasing my projects and experiences.",
  icons: {
    icon: "/icon.png",
    shortcut: "/icon.png",
    apple: "/icon.png",
  },
  openGraph: {
    title: "Stimmie",
    description: "A portfolio showcasing my projects and experiences.",
    url: "https://stimmie.dev",
    siteName: "Stimmie",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Stimmie",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Stimmie",
    description: "A portfolio showcasing my projects and experiences.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
  themeColor: "#000000",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
