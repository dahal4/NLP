# Chapter 1 — Assignment 1: POS Tagging & NER

Extract article text from a URL, then use spaCy to pull **nouns/verbs** (POS) and **named entities** (NER).

## Files

| File | Description |
|------|-------------|
| `Assignment1/POS_01.ipynb` | Main notebook |
| `Assignment1/article.txt` | Scraped article body (paragraphs only) |
| `Assignment1/POS_01.csv` | Nouns and verbs with POS tags |

## What it does

1. **Scrape** — Fetch a news article URL with `requests` + BeautifulSoup; write `<p>` text to `article.txt`.
2. **POS** — Load `en_core_web_sm`, keep alphabetic tokens whose POS is `NOUN` or `VERB`, save as CSV (`Word`, `POS_Tag`).
3. **NER** — Collect spaCy entities (`Entity`, `Entity_type`) into a DataFrame (printed; not written to disk).

Example source used in the notebook: [News Decoder — killer robots / LAWS](https://news-decoder.com/is-time-running-out-to-tame-killer-robots/).

## Setup

```bash
pip install requests beautifulsoup4 spacy pandas
python -m spacy download en_core_web_sm
```

## Run

Open `Assignment1/POS_01.ipynb` and run all cells. When prompted, paste an article URL.

Outputs:

- `article.txt` — scraped paragraphs  
- CSV of nouns/verbs (notebook default name: `POS_01.csv`)  
- NER table printed in the notebook  
