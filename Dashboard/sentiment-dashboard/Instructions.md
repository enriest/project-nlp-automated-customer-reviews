Static Dashboard: Usage & Generative AI Summaries

Overview
--------
This folder contains a lightweight static dashboard for the Automated Customer Review Sentiment Analysis project. It includes charts, a small rule-based "Live Prediction" demo and a "Generative AI Summaries" section that loads pre-generated summaries from the `outputs/summaries/` folder.

Files of interest
-----------------
- `index.html` — The static dashboard page. It fetches datasets and summary JSONs from `outputs/summaries/` and renders charts.
- `outputs/summaries/index.json` — Index file created by the summarizer CLI listing available category summary files.
- `outputs/summaries/*.json` — Per-category summary files (one file per product category). Each file includes rating-level summaries (1..5).

How the Generative AI Summaries work
-----------------------------------
The generative summaries are produced by the repository's summarizer script and saved to `outputs/summaries/`.
- Each per-category JSON contains the category name, total_reviews and a `ratings` map with keys 1..5. Each rating entry includes a `summary` string and a `count`.
- The static dashboard fetches `outputs/summaries/index.json` and the category JSON chosen by the user in the "Generative AI Summaries" panel.

Serving the static dashboard locally
----------------------------------
To test the static dashboard locally you must serve it over HTTP so the browser can fetch JSON files. From the project root run one of these commands (macOS / zsh):

# Serve the Dashboard folder on port 8000
python3 -m http.server 8000 --directory Dashboard

# Or, serve the project root (useful if outputs/ lives alongside Dashboard)
python3 -m http.server 8000

Then open in your browser:
http://localhost:8000/sentiment-dashboard/index.html
(or http://localhost:8000/Dashboard/sentiment-dashboard/index.html depending on how you served the files)

If the Generative AI Summaries drop-down shows "No summaries available", verify that `outputs/summaries/index.json` exists and is reachable from the served root.

Regenerating summaries (CLI)
----------------------------
The repository includes a CLI script to create/update the generative summaries. Typical usage (from project root):

python3 scripts/summarize_by_category_and_rating.py \
    --input archive/1429_1.csv \
    --output-dir outputs/summaries \
    --top-n 10 \
    --min-reviews 50 \
    --max-reviews-per-rating 200 \
    --model-name offline_models/summarizer/sshleifer-distilbart-cnn-12-6 \
    --cache-dir offline_models/summarizer \
    --offline

Notes:
- The `--model-name` should point to a local cached Hugging Face summarization model if you run with `--offline`.
- If you don't want abstractive summaries or the model isn't available, pass `--use-extractive` to use a TF-IDF extractive fallback.
- The CLI writes one JSON file per selected category and an `index.json` listing them.

Using the Streamlit dashboard
----------------------------
There is a Streamlit app at `Dashboard/app.py` that can browse and (optionally) run the summarizer CLI from within the web UI. To run it:

pip install -r requirements.txt
streamlit run Dashboard/app.py

The Streamlit app reads the same `outputs/summaries/` files and provides a button to re-run the summarizer (this calls the CLI in a subprocess).

Troubleshooting & common issues
-------------------------------
- CORS / file:// problems: Opening `index.html` via the file protocol (double-clicking) will prevent fetch() from loading the JSON files. Always serve over HTTP.
- Files not found: Ensure `outputs/summaries/` exists and contains `index.json`. Paths the static page tries: `./outputs/summaries/index.json`, `../outputs/summaries/index.json`, `/outputs/summaries/index.json` — serve from a root where one of those paths resolves.
- Large model downloads: Cached models can be large (1GB+). Use the provided `scripts/cache_summarization_model.py` helper to download ahead of time.

Where to go next
-----------------
- If you want UI improvements (read-more, copy/share), I can add them and style the summaries.
- If you want the static page to always work regardless of hosting layout, I can add a small server-side proxy or produce a single bundled JSON file placed next to the HTML.

