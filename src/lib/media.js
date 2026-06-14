// src/lib/media.js

import Parser from "rss-parser";

const parser = new Parser({
  customFields: {
    item: [
      ["letterboxd:filmTitle", "filmTitle"],
      ["letterboxd:filmYear", "filmYear"],
      ["letterboxd:memberRating", "rating"],
      ["letterboxd:watchedDate", "watchedDate"],
      ["description", "description"],
    ],
  },
});

// Letterboxd username
const LETTERBOXD_USERNAME = "stimmieuwu";

// Goodreads numeric user ID (Simonee Ezekiel Mariquit / goodreads.com/stimmie).
const GOODREADS_USER_ID = "186878528";

// Last.fm username
const LASTFM_USERNAME = "mistakenpog";

/**
 * Build the Last.fm "currently listening" payload.
 *
 * Last.fm itself has no native collage image and the top-artists API needs a
 * key, so we use tapmusic.net's collage generator: a dynamic 3x3 grid of the
 * user's top albums over the last 30 days. It's just an <img> src, so it stays
 * fresh on every load with no API key.
 */
export function getLastfmMusic() {
  return {
    username: LASTFM_USERNAME,
    period: "Past 30 days",
    profileUrl: `https://www.last.fm/user/${LASTFM_USERNAME}`,
    collageUrl: `https://www.tapmusic.net/collage.php?user=${LASTFM_USERNAME}&type=1month&size=3x3&caption=true`,
  };
}

/**
 * Fetch latest film from Letterboxd RSS
 */
export async function getLatestFilm() {
  try {
    const feed = await parser.parseURL(
      `https://letterboxd.com/${LETTERBOXD_USERNAME}/rss/`,
    );

    if (!feed.items || feed.items.length === 0) {
      return null;
    }

    const latestEntry = feed.items[0];

    // Extract poster image from description HTML
    let posterUrl = null;
    if (latestEntry.description) {
      const imgMatch = latestEntry.description.match(/<img[^>]+src="([^"]+)"/);
      if (imgMatch) {
        posterUrl = imgMatch[1];
      }
    }

    return {
      title:
        latestEntry.filmTitle ||
        latestEntry.title?.replace(/^.+?- /, "") ||
        "Unknown",
      year: latestEntry.filmYear || null,
      rating: latestEntry.rating ? parseFloat(latestEntry.rating) : null,
      watchedDate: latestEntry.watchedDate || latestEntry.pubDate,
      link: latestEntry.link,
      posterUrl,
    };
  } catch (error) {
    console.error("Error fetching Letterboxd RSS:", error);
    return null;
  }
}

/**
 * Fetch the most recently READ book from Goodreads RSS.
 * Sorted by `date_read` descending so the top entry is the latest finish.
 */
export async function getLatestBook() {
  try {
    const feed = await parser.parseURL(
      `https://www.goodreads.com/review/list_rss/${GOODREADS_USER_ID}?shelf=read&sort=date_read&order=d`,
    );

    const shelf = "read";

    if (!feed.items || feed.items.length === 0) {
      return null;
    }

    const latestEntry = feed.items[0];

    // Extract book cover from description HTML
    let coverUrl = null;
    if (latestEntry.description) {
      const imgMatch = latestEntry.description.match(/<img[^>]+src="([^"]+)"/);
      if (imgMatch) {
        coverUrl = imgMatch[1];
      }
    }

    // Extract author from description
    let author = null;
    if (latestEntry.description) {
      const authorMatch = latestEntry.description.match(/author:\s*([^<\n]+)/i);
      if (authorMatch) {
        author = authorMatch[1].trim();
      }
    }

    return {
      title: latestEntry.title || "Unknown",
      author,
      link: latestEntry.link,
      coverUrl,
      shelf, // 'currently-reading' or 'read'
    };
  } catch (error) {
    console.error("Error fetching Goodreads RSS:", error);
    return null;
  }
}

/**
 * Fetch all media data (for build-time fetching)
 */
export async function getAllMediaData() {
  const [film, book] = await Promise.all([getLatestFilm(), getLatestBook()]);

  const music = getLastfmMusic();

  return { film, book, music };
}
