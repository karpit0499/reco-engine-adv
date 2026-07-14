import os
import json
import time
import random
import uuid
from datetime import datetime, timezone

from google.cloud import pubsub_v1

PROJECT = os.environ["PROJECT"]
pub = pubsub_v1.PublisherClient()
TOPIC = pub.topic_path(PROJECT, "user-events")

CATEGORIES = ["electronics", "books", "home", "sports", "beauty"]
N_ITEMS, N_USERS = 500, 300
SLEEP = float(os.environ.get("SLEEP", "0.1"))

# Seeded BEFORE the catalog/users are drawn: re-running this script always
# rebuilds the IDENTICAL catalog and the identical hidden preferences, so the
# ground-truth files you .gitignore can always be regenerated exactly.
random.seed(7)
ITEMS = {f"i{n:04d}": {"category": random.choice(CATEGORIES),
                       "price": round(random.uniform(5, 200), 2)}
         for n in range(N_ITEMS)}
USERS = {f"u{n:04d}": random.choice(CATEGORIES) for n in range(N_USERS)}

# Dump ground truth ONCE. Step 2.2 loads these into BigQuery; the retrieval
# test (Phase 9) and the simulator (Phase 12) read them.
if not os.path.exists("catalog.jsonl"):
    with open("catalog.jsonl", "w") as f:
        for iid, it in ITEMS.items():
            f.write(json.dumps({"item_id": iid, **it}) + "\n")
    with open("users.jsonl", "w") as f:
        for uid, pref in USERS.items():
            f.write(json.dumps({"user_id": uid, "preferred_category": pref}) + "\n")
    print("wrote catalog.jsonl + users.jsonl")

# Weighted so ~57% of events are click/add_to_cart/purchase — the "positive"
# interactions the two-tower model trains on.
EVENT_TYPES = ["view", "view", "view", "click", "click", "add_to_cart", "purchase"]
BY_CAT = {c: [i for i, it in ITEMS.items() if it["category"] == c] for c in CATEGORIES}
ALL_ITEMS = list(ITEMS)


def sample_item(pref):
    # 70% of interactions land in the user's preferred category — THE SIGNAL.
    return random.choice(BY_CAT[pref]) if random.random() < 0.7 else random.choice(ALL_ITEMS)


print(f"publishing to {TOPIC} — Ctrl-C to stop")
while True:
    uid = random.choice(list(USERS))
    ev = {
        "event_id": str(uuid.uuid4()),
        "user_id": uid,
        "item_id": sample_item(USERS[uid]),
        "event_type": random.choice(EVENT_TYPES),
        "event_time": datetime.now(timezone.utc).isoformat(),
    }
    pub.publish(TOPIC, json.dumps(ev).encode("utf-8"))
    time.sleep(SLEEP)