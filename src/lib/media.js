import Parser from 'rss-parser';

const parser = new Parser({
  customFields: {
    item: [
      ['letterboxd:filmTitle', 'filmTitle'],
      ['letterboxd:filmYear', 'filmYear'],
      ['letterboxd:memberRating', 'rating'],
      ['letterboxd:watchedDate', 'watchedDate'],
      ['description', 'description'],
    ],
  },
});

// Letterboxd username
const LETTERBOXD_USERNAME = 'stimmieuwu';

// Goodreads user ID - you'll need to update this with your actual numeric ID
// Find it in your Goodreads profile URL: goodreads.com/user/show/YOUR_ID
const GOODREADS_USER_ID = '181836728'; // Update this!

// MyAnimeList username
const MAL_USERNAME = 'amorgosposter';

/**
 * Fetch currently watching anime from MyAnimeList RSS
 */
export async function getLatestAnime() {
  try {
    const feed = await parser.parseURL(`https://myanimelist.net/rss.php?type=rw&u=${MAL_USERNAME}`);
    
    if (!feed.items || feed.items.length === 0) {
      return null;
    }

    // Find the most recently updated "Watching" anime
    const watchingAnime = feed.items.find(item => 
      item.description && item.description.includes('Watching')
    );
    
    const latestEntry = watchingAnime || feed.items[0];
    
    // Parse title and type (e.g., "Anime Title - TV")
    let title = latestEntry.title || 'Unknown';
    let type = null;
    const titleMatch = title.match(/^(.+?)\s*-\s*(TV|Movie|OVA|ONA|Special|Music)$/i);
    if (titleMatch) {
      title = titleMatch[1].trim();
      type = titleMatch[2];
    }

    // Parse progress from description (e.g., "Watching - 5 of 12 episodes")
    let progress = null;
    let total = null;
    let status = 'Watching';
    if (latestEntry.description) {
      const progressMatch = latestEntry.description.match(/(\w+(?:\s+\w+)?)\s*-\s*(\d+)\s+of\s+(\d+)/i);
      if (progressMatch) {
        status = progressMatch[1];
        progress = parseInt(progressMatch[2], 10);
        total = parseInt(progressMatch[3], 10);
      }
    }

    // Extract anime ID from link to construct cover URL
    let coverUrl = null;
    if (latestEntry.link) {
      const idMatch = latestEntry.link.match(/\/anime\/(\d+)/);
      if (idMatch) {
        // MAL CDN image URL pattern
        coverUrl = `https://cdn.myanimelist.net/images/anime/${Math.floor(parseInt(idMatch[1], 10) / 10000)}/${idMatch[1]}.jpg`;
      }
    }

    return {
      title,
      type,
      status,
      progress,
      total,
      link: latestEntry.link,
      coverUrl,
    };
  } catch (error) {
    console.error('Error fetching MyAnimeList RSS:', error);
    return null;
  }
}

/**
 * Fetch latest film from Letterboxd RSS
 */
export async function getLatestFilm() {
  try {
    const feed = await parser.parseURL(`https://letterboxd.com/${LETTERBOXD_USERNAME}/rss/`);
    
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
      title: latestEntry.filmTitle || latestEntry.title?.replace(/^.+?- /, '') || 'Unknown',
      year: latestEntry.filmYear || null,
      rating: latestEntry.rating ? parseFloat(latestEntry.rating) : null,
      watchedDate: latestEntry.watchedDate || latestEntry.pubDate,
      link: latestEntry.link,
      posterUrl,
    };
  } catch (error) {
    console.error('Error fetching Letterboxd RSS:', error);
    return null;
  }
}

/**
 * Fetch currently reading / latest read from Goodreads RSS
 */
export async function getLatestBook() {
  try {
    // Try currently-reading shelf first
    let feed = await parser.parseURL(
      `https://www.goodreads.com/review/list_rss/${GOODREADS_USER_ID}?shelf=currently-reading`
    );

    let shelf = 'currently-reading';
    
    // If no books in currently-reading, try the read shelf
    if (!feed.items || feed.items.length === 0) {
      feed = await parser.parseURL(
        `https://www.goodreads.com/review/list_rss/${GOODREADS_USER_ID}?shelf=read`
      );
      shelf = 'read';
    }

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
      title: latestEntry.title || 'Unknown',
      author,
      link: latestEntry.link,
      coverUrl,
      shelf, // 'currently-reading' or 'read'
    };
  } catch (error) {
    console.error('Error fetching Goodreads RSS:', error);
    return null;
  }
}

/**
 * Fetch all media data (for build-time fetching)
 */
export async function getAllMediaData() {
  const [film, book, anime] = await Promise.all([
    getLatestFilm(),
    getLatestBook(),
    getLatestAnime(),
  ]);

  return { film, book, anime };
}
