import time

import vertexai
from vertexai.resources.preview import feature_store as fs

PROJECT, REGION = "reco-engine-adv", "europe-west3"

# ── Why the name has a timestamp in it ───────────────────────────────────────
# Deleting a FeatureOnlineStore is a long-running operation. `store.delete()`
# returns as soon as Google ACCEPTS the delete, not when it FINISHES it — and
# the name stays reserved (tombstoned) for many minutes afterwards. Re-running
# with a fixed name then dies on:
#     ALREADY_EXISTS: Re-using the same name as a FeatureOnlineStore being deleted
# Meanwhile `FeatureOnlineStore.list()` returns [] the whole time, because the
# store leaves the LIST when the delete is accepted but the NAME is held until it
# completes. The list is a BILLING check, not a NAME-AVAILABILITY check.
# Don't wait on the name. The name is arbitrary. Make it unique per run.
STORE_NAME = f"reco_online_{int(time.time())}"
VIEW_NAME = "user_features_view"
DEMO_USER = "u0007"

WARM_READS = 20        # how many SUCCESSFUL reads to time
SETTLE_SECS = 20       # let the sync propagate across replicas before benchmarking
MAX_ATTEMPTS = 60      # cap total warm-loop attempts so a broken store can't hang us

vertexai.init(project=PROJECT, location=REGION)

store = None
try:
    print(f"creating the Bigtable online store '{STORE_NAME}' — BILLING STARTS NOW "
          f"(billed per node-hour) ...")
    store = fs.FeatureOnlineStore.create_bigtable_store(STORE_NAME)
    print("online store:", store.resource_name)

    # A FeatureView = "which rows to sync into the online store, and how often".
    # Point it at the _latest VIEW: the online store wants exactly ONE row per
    # entity id, and the raw append-only table has one per snapshot.
    view = store.create_feature_view(
        name=VIEW_NAME,
        source=fs.utils.FeatureViewBigQuerySource(
            uri=f"bq://{PROJECT}.reco.user_features_latest",
            entity_id_columns=["user_id"],
        ),
        sync_config="0 * * * *",          # hourly cron; we trigger one immediately below
    )
    print("feature view:", view.resource_name)

    view.sync()
    print("sync started — first sync usually takes 3–8 minutes ...")

    deadline = time.time() + 900          # give up after 15 minutes
    while time.time() < deadline:
        try:
            t0 = time.perf_counter()
            resp = view.read(key=[DEMO_USER])
            cold_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            # Print the MESSAGE, not just the class name. "NotFound" fourteen times
            # tells you nothing; the message tells you which NotFound it is.
            print("  not synced yet:", type(e).__name__, "-", str(e)[:140])
            time.sleep(30)
            continue

        # ---- First successful read. It is NOT a latency measurement. ----
        # It paid for gRPC channel setup + TLS handshake + Berlin->Frankfurt RTT.
        # Report it honestly as a cold read, then measure the steady state.
        print(f"\ncold read (channel setup + TLS + first RPC): {cold_ms:.1f} ms")
        print(resp.to_dict())

        # ---- Let the sync settle before benchmarking ----
        # The online store is eventually consistent across serving replicas. The
        # first read can succeed on one replica while the next read lands on
        # another that hasn't got the row yet — a transient 404 seconds after a
        # successful read. A benchmark that aborts on that is a bad benchmark.
        print(f"\nletting the sync settle for {SETTLE_SECS}s before benchmarking ...")
        time.sleep(SETTLE_SECS)

        lat, misses, attempts = [], 0, 0
        while len(lat) < WARM_READS and attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                t0 = time.perf_counter()
                view.read(key=[DEMO_USER])
                lat.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                # Tolerate and COUNT it. Do not let one transient 404 kill the run.
                misses += 1
                print(f"  transient read failure ({type(e).__name__}) — "
                      f"{misses} so far, retrying")
                time.sleep(2)

        if len(lat) < 3:
            print(f"\nonly {len(lat)} successful reads in {attempts} attempts — "
                  f"the store is not serving reliably. No latency number to report.")
        else:
            lat.sort()
            p50 = lat[len(lat) // 2]
            p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
            print(f"\nWARM READ  median {p50:.1f} ms   p95 {p95:.1f} ms   "
                  f"min {lat[0]:.1f} ms   max {lat[-1]:.1f} ms")
            print(f"           n={len(lat)} successful / {attempts} attempts "
                  f"({misses} transient failures)")
            print("^^^ SCREENSHOT THIS — the warm median is the online-serving number.")
        break
    else:
        print("sync did not finish within 15 min — deleting anyway, cost > curiosity")

except Exception as e:
    # Anything unexpected: say so loudly, but STILL fall through to the delete.
    print(f"\nUNEXPECTED FAILURE: {type(e).__name__}: {e}")
    print("(deleting the store anyway — the meter must stop)")

finally:
    if store is not None:
        print("\ndeleting the online store (stops the meter) ...")
        store.delete(force=True)          # force=True also deletes the child FeatureView
        print("delete accepted — the meter stops now.")
        print("(The delete is a long-running op, so the NAME stays reserved a few more")
        print(" minutes. That's why this script picks a fresh name every run.)")