import { NextResponse } from 'next/server';

// Subdomain redirect mappings
// Format: subdomain -> { path -> destination }
const subdomainRedirects = {
  'workshops': {
    'dsg-codeit-day1': 'https://docs.google.com/presentation/d/...',
    'dsg-codeit-day2': 'https://docs.google.com/presentation/d/...',
    'nextstep-hacks': 'https://docs.google.com/presentation/d/...',
    // Add your workshop links here!
  },
  'talks': {
    'qcu-ml-python': 'https://docs.google.com/presentation/d/...',
    'dep-ai-study': 'https://docs.google.com/presentation/d/...',
    // Add your talk links here!
  },
  'links': {
    'freshie': 'https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/',
    'hackathons': 'https://docs.google.com/document/d/1nO2-vsOKjl4C_AngSSqc-knkUZvjJhe700unYZzcYsg/edit?usp=sharing',
    'linkedin': 'https://www.linkedin.com/in/stimmie',
    'github': 'https://www.github.com/smmariquit',
    'instagram': 'https://www.instagram.com/friedicecrm',
    'osu': 'https://osu.ppy.sh/users/14900686',
  },
};

export function middleware(request) {
  const url = request.nextUrl;
  const hostname = request.headers.get('host') || '';
  
  // Extract subdomain (e.g., "workshops" from "workshops.stimmie.dev")
  // Handle both production (stimmie.dev) and preview (vercel.app) domains
  let subdomain = null;
  
  if (hostname.includes('stimmie.dev')) {
    const parts = hostname.split('.');
    if (parts.length === 3 && parts[0] !== 'www') {
      subdomain = parts[0];
    }
  } else if (hostname.includes('vercel.app')) {
    // For Vercel preview deployments, use query param or path-based routing
    // e.g., ?subdomain=workshops or /workshops/...
  }
  
  // If we have a subdomain and it's in our mappings
  if (subdomain && subdomainRedirects[subdomain]) {
    const path = url.pathname.replace(/^\//, ''); // Remove leading slash
    const destination = subdomainRedirects[subdomain][path];
    
    if (destination) {
      return NextResponse.redirect(destination);
    }
    
    // If subdomain exists but path not found, show a list or 404
    // Optionally redirect to main domain
  }
  
  return NextResponse.next();
}

// Configure which paths the middleware runs on
export const config = {
  matcher: [
    /*
     * Match all paths except:
     * - _next (Next.js internals)
     * - static files (images, fonts, etc.)
     * - api routes
     */
    '/((?!_next|api|favicon.ico|images|logos|fonts|projects|talks|satoshi).*)',
  ],
};
