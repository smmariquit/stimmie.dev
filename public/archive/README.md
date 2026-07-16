# Archive assets

Frozen files for historical site versions. **Do not delete** paths listed in `MANIFEST.json`.

| Path | Used by |
|------|---------|
| `mosaic/` | `/archive/v1`, `/v2` background grids |
| `v1/` | `/archive/v1` talks, projects, logos |
| `v2/` | `/v2` bento thumbnails and icons |
| `v2/screenshot.png` | `/archive` timeline (v2.0) |
| `v3/screenshot.png` | `/archive` timeline (v3.0) |
| `v1/screenshot.png` | `/archive` timeline (v1.0) |
| `carrd/screenshot.png` | `/archive` timeline (v0.2) |
| `linktree/screenshot.png` | `/archive` timeline (v0.1) |
| `himitsu/` | `/archive/himitsu`, `/archive` timeline (v0.0) |

## Maintenance

After changing live-site images that archives still reference:

```bash
npm run archive:snapshot   # copy assets into public/archive/
npm run archive:verify     # confirm MANIFEST.json entries exist
```

`npm run build` runs `archive:verify` automatically.

Live site code lives in `src/data/` and `src/app/HomeClient.js`. Archives use frozen modules in `src/archive/`.

**Do not reference `/archive/*` from the live site**: archives and production are independent bundles.
