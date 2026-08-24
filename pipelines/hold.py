import sys


score = float(sys.argv[1])
generation_id = sys.argv[2]
print("=" * 70)
print(f"GATE CLOSED — generation {generation_id} scored {score:.1%}.")
print("Its item vectors remain immutable staged objects; the live index was not touched.")
print("Its user vectors remain in the generation staging table; the live view was not switched.")
print("The active generation pointer and champion record are unchanged.")
print("=" * 70)
sys.exit(1)
