# Datasetdokumentation

This folder contains the data card source and exporters used in this repo.

## What is here
- `AURA_Dataset_v0.2_DataCard.md`: primary markdown source for the data card.
- `src/`: card-builder source (TypeScript + styles).
- `build.config.js`: bundles browser artifacts into `dist/`.
- `export_datacard_website_like.py`: markdown -> interactive website-like HTML.
- `export_print_ready_html.py`: markdown -> print-optimized HTML.

## Prerequisites
- Node.js 18+ (npm included)
- Python 3.10+ (`python3`)

## Quick start after clone
```bash
cd Datasetdokumentation
npm install
npm run build
```

`npm run build` creates:
- `dist/card-builder.js`
- `dist/card-interactions.js`
- `dist/card-default.css`

## Regenerate data card exports
From `Datasetdokumentation/`:

```bash
npm run build:website-like
npm run build:print-html
```

Equivalent explicit commands:

```bash
python3 export_datacard_website_like.py \
  --input AURA_Dataset_v0.2_DataCard.md \
  --output AURA_Dataset_v0.2_DataCard.website_like.html

python3 export_print_ready_html.py \
  --input AURA_Dataset_v0.2_DataCard.md \
  --output AURA_Dataset_v0.2_DataCard.print.html
```

## Markdown structure contract
Use heading levels consistently:
- `#` card title
- `##` section
- `###` subsection
- `####` field

Reference example in this repo:
- `AURA_Dataset_v0.2_DataCard.md`

## Notes
- `.print.pdf` files are treated as generated artifacts and are not tracked.
- Keep `package-lock.json` committed for reproducible installs.
