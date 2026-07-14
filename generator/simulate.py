import os
import json
import random
import time

import requests

API = os.environ["API_URL"].rstrip("/")
ROUNDS = int(os.environ.get("ROUNDS", "3000"))

users = [json.loads(line) for line in open("users.jsonl")]
catalog = {r["item_id"]: r["category"]
           for r in (json.loads(line) for line in open("catalog.jsonl"))}

BASE_CTR = 0.03           # click probability when NOTHING shown is relevant
MATCH_BOOST = 0.30        # + this × (share of the top-10 that matches my preference)
CONV_GIVEN_CLICK = 0.20

errors = 0
for i in range(ROUNDS):
    u = random.choice(users)
    try:
        r = requests.post(f"{API}/recommend",
                          json={"user_id": u["user_id"]}, timeout=30)
        r.raise_for_status()
        rec = r.json()
    except Exception as e:
        errors += 1
        if errors <= 3:
            print("recommend failed:", repr(e))
        if errors > 50:
            raise SystemExit("too many failures — is the index endpoint still deployed?")
        time.sleep(1)
        continue

    items = rec["items"]
    if isinstance(items, str):                  # defensive: accept either shape
        items = json.loads(items)

    match_share = (sum(catalog.get(x) == u["preferred_category"] for x in items)
                   / max(len(items), 1))
    p_click = BASE_CTR + MATCH_BOOST * match_share

    if random.random() < p_click:
        fb = {"impression_id": rec["impression_id"], "user_id": u["user_id"],
              "event": "click"}
        requests.post(f"{API}/feedback", json=fb, timeout=30)
        if random.random() < CONV_GIVEN_CLICK:
            fb["event"] = "conversion"
            requests.post(f"{API}/feedback", json=fb, timeout=30)

    if i % 200 == 0:
        print(f"{i}/{ROUNDS}  (errors so far: {errors})")
    time.sleep(0.02)

print(f"done — {ROUNDS} rounds, {errors} errors")