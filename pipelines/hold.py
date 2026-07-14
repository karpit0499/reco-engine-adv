import sys

score = float(sys.argv[1])
print("=" * 70)
print(f"GATE CLOSED — new embeddings scored {score:.1%} and were NOT promoted.")
print("The live Vector Search index still holds the PREVIOUS item vectors.")
print()
print("⚠️  BUT: train.py already appended new USER vectors to BigQuery, and the API")
print("    reads user_embeddings_latest. The system is now IN the consistency window:")
print("    new user vectors vs. old item vectors. Retrieval quality is at chance.")
print("    Investigate, then either fix and re-run, or upsert manually to close the gap.")
print("=" * 70)
sys.exit(1)   # non-zero = the pipeline is marked FAILED. That is the point.