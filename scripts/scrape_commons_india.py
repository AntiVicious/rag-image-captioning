"""
Pull a diverse set of genuinely Indian-context photographs from
Wikimedia Commons via its API -- NOT arbitrary web scraping. Commons
hosts only media under structured, verified open licenses (CC0/CC-BY/
CC-BY-SA/public domain), with per-file author/license/source metadata,
which is exactly what makes this legitimate to reuse (the same source
the Tamil Nadu bus dataset we found earlier was built from).

Spans many categories (festivals, markets, temples, clothing, transport,
streets) rather than one theme, since food alone was judged insufficient
distribution-shift coverage. Captions: uses each file's own Commons
description when it reads like real English prose; falls back to a
category-derived template otherwise -- both are real, disclosed
limitations of scraped/weakly-labeled data, not hidden.

Respects Wikimedia API etiquette: descriptive User-Agent, rate-limited
requests, exponential backoff on 429s.

Usage:
    python scripts/scrape_commons_india.py \
        --out-img-dir ./coco_indian/commons/images \
        --out-ann-file ./coco_indian/commons/captions_commons.json \
        --out-attribution-file ./coco_indian/commons/attribution.jsonl \
        --id-offset 91000000 \
        --total-target 10000 \
        --per-category-cap 700
"""

import argparse
import io
import json
import os
import re
import time

import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "rag-image-captioning-research/1.0 (educational RAG captioning project)"}

# (Commons category name, human-readable phrase used in the fallback caption template)
CATEGORIES = [
    ("Holi", "the Holi festival of colours in India"),
    ("Diwali", "the Diwali festival of lights in India"),
    ("Indian weddings", "an Indian wedding"),
    ("Bazaars in India", "a bazaar in India"),
    ("Festivals of India", "a festival in India"),
    ("Temples in India", "a temple in India"),
    ("Markets in India", "a market in India"),
    ("Hindu temples in India", "a Hindu temple in India"),
    ("Street food in India", "street food in India"),
    ("Streets in India", "a street in India"),
    ("Rickshaws in India", "a rickshaw in India"),
    ("Saris", "a woman wearing a sari"),
    ("Railway stations in India", "a railway station in India"),
    ("Buses in India", "a bus in India"),
    ("Agriculture in India", "agriculture in India"),
    ("Rural life in India", "rural life in India"),
    ("Mosques in India", "a mosque in India"),
    ("Sikh gurdwaras in India", "a Sikh gurdwara in India"),
]

NON_PHOTO_BLOCKLIST = re.compile(
    r"\b(coat of arms|seal of|flag of|logo|emblem|map of|diagram|icon|"
    r"symbol|chart|graph|route map|locator map|banner|screenshot)\b",
    re.IGNORECASE,
)

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _get(session, params, retries=5):
    params = dict(params, format="json")
    for attempt in range(retries):
        resp = session.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429 or "too many requests" in resp.text.lower():
            wait = 2**attempt
            print(f"  rate limited, backing off {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(0.4)  # be polite regardless
        return resp.json()
    raise RuntimeError(f"Repeated rate limiting on {params}")


def list_category_files(session, category, cap):
    titles = []
    cmcontinue = None
    while len(titles) < cap:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": 500,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _get(session, params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles[:cap]


def fetch_image_info(session, titles):
    """Batch imageinfo lookup, 50 titles per call (Commons API limit).

    iiurlwidth=800 asks the API to also return a pre-sized 800px-wide
    thumbnail URL (thumburl) -- Wikimedia's own 429 error message for
    bulk original-file downloads explicitly asks callers to use
    thumbnails instead, so that's what download_image() prefers.
    """
    results = {}
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": 800,
        }
        data = _get(session, params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo")
            if info:
                results[page["title"]] = info[0]
    return results


def download_image(session, info, retries=5):
    """Download the thumbnail (preferred -- see fetch_image_info) or, if
    none was returned, the full original. Rate-limited the same way as
    API calls: Wikimedia's upload/thumbnail CDN enforces its own 429s,
    independent of the api.php rate limit."""
    url = info.get("thumburl") or info["url"]
    for attempt in range(retries):
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = 2**attempt
            print(f"  download rate limited, backing off {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(0.3)  # be polite regardless
        return resp.content
    raise RuntimeError(f"Repeated rate limiting downloading {url}")


def clean_description(raw_html):
    if not raw_html:
        return None
    text = _HTML_TAG.sub(" ", raw_html)
    text = _WS.sub(" ", text).strip()
    if not text or len(text) < 15 or len(text) > 400:
        return None
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    if non_ascii_ratio > 0.15:  # likely non-English
        return None
    return text


def main():
    parser = argparse.ArgumentParser(description="Scrape Indian-context photos from Wikimedia Commons")
    parser.add_argument("--out-img-dir", required=True)
    parser.add_argument("--out-ann-file", required=True)
    parser.add_argument("--out-attribution-file", required=True)
    parser.add_argument("--id-offset", type=int, default=91000000)
    parser.add_argument("--total-target", type=int, default=10000)
    parser.add_argument("--per-category-cap", type=int, default=700)
    args = parser.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)
    session = requests.Session()

    annotations = []
    attributions = []
    next_id = args.id_offset
    seen_urls = set()

    for category, phrase in CATEGORIES:
        if next_id - args.id_offset >= args.total_target:
            break
        print(f"\n=== Category:{category} ===")
        titles = list_category_files(session, category, args.per_category_cap)
        print(f"  {len(titles)} candidate files")
        if not titles:
            continue

        info_by_title = fetch_image_info(session, titles)
        kept = 0
        for title in titles:
            if next_id - args.id_offset >= args.total_target:
                break
            info = info_by_title.get(title)
            if not info:
                continue
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            if info.get("width", 0) < 200 or info.get("height", 0) < 200:
                continue
            if NON_PHOTO_BLOCKLIST.search(title):
                continue
            url = info.get("url")
            if not url or url in seen_urls:
                continue

            ext = info.get("extmetadata", {})
            description = clean_description(ext.get("ImageDescription", {}).get("value"))
            if description and NON_PHOTO_BLOCKLIST.search(description):
                description = None
            caption = description or f"A photograph of {phrase}."

            try:
                image_bytes = download_image(session, info)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                print(f"  skip (download/decode failed): {title}: {e}")
                continue

            img_id = next_id
            next_id += 1
            seen_urls.add(url)
            image.save(os.path.join(args.out_img_dir, f"{img_id}.jpg"), "JPEG", quality=90)
            annotations.append({"image_id": img_id, "id": img_id, "caption": caption})
            attributions.append(
                {
                    "image_id": img_id,
                    "title": title,
                    "author": ext.get("Artist", {}).get("value"),
                    "license": ext.get("LicenseShortName", {}).get("value"),
                    "license_url": ext.get("LicenseUrl", {}).get("value"),
                    "source_url": info.get("descriptionurl"),
                    "category": category,
                }
            )
            kept += 1
        print(f"  kept {kept} photos")

    with open(args.out_ann_file, "w") as f:
        json.dump({"annotations": annotations}, f)
    with open(args.out_attribution_file, "w") as f:
        for row in attributions:
            f.write(json.dumps(row) + "\n")

    print(f"\nTotal ingested: {len(annotations)}")
    print(f"Images: {args.out_img_dir}")
    print(f"Captions: {args.out_ann_file}")
    print(f"Attribution: {args.out_attribution_file}")
    print(f"image_id range: {args.id_offset} - {next_id - 1}")


if __name__ == "__main__":
    main()
