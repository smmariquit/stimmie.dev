# Archive assets

Frozen files for historical site versions. **Do not delete** paths listed in `MANIFEST.json`.

| Path | Used by |
|------|---------|
| `mosaic/` | `/archive/v1`, `/v2` background grids |
| `v1/` | `/archive/v1` talks, projects, logos |
| `v2/` | `/v2` bento thumbnails and icons |
| `v2-screenshot.png` | `/archive` timeline |

## Maintenance

After changing live-site images that archives still reference:

```bash
npm run archive:snapshot   # copy assets into public/archive/
npm run archive:verify     # confirm MANIFEST.json entries exist
```

`npm run build` runs `archive:verify` automatically.

Live site code lives in `src/data/` and `src/app/HomeClient.js`. Archives use frozen modules in `src/archive/`.

**Do not reference `/archive/*` from the live site** — archives and production are independent bundles.
