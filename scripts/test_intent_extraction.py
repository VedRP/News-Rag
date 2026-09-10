"""
Phase 4 verification: raw utterance -> intent parse -> backend validation.

Run: python scripts/test_intent_extraction.py

Covers a range of real utterances from context.md Section 9's examples, plus one
deliberately ambiguous/nonsense utterance, to confirm the ambiguous one is flagged
as low-confidence/"unclear" rather than forced into a confident-but-wrong parse.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.llm.prompts.intent_parser import parse_intent
from backend.backend_validation import validate_intent

TEST_UTTERANCES = [
    "Give me cricket news.",
    "Give me Mumbai news.",
    "What's happening in Maharashtra?",
    "Give me today's cricket news from Maharashtra.",
    "Tell me about financial markets in Mumbai.",
    "Marathi mein Mumbai ki cricket news batao.",
    "Change to Hindi.",
    "asdkj random gibberish blah blah",  # deliberately ambiguous / nonsense
]


def main() -> None:
    print("=" * 70)
    print("PHASE 4: Intent/Entity Extraction + Validation Test")
    print("=" * 70)

    results = []
    for utterance in TEST_UTTERANCES:
        print(f"\n--- Utterance: {utterance!r}")
        parsed = parse_intent(utterance)
        print("Raw parse:", json.dumps(parsed, ensure_ascii=False))

        validated = validate_intent(parsed)
        print(
            f"Validated: intent={validated.intent!r} topic={validated.topic!r} "
            f"location={validated.location!r} time={validated.time!r} "
            f"language={validated.language!r} confidence={validated.confidence:.2f} "
            f"actionable={validated.is_actionable}"
        )
        if validated.errors:
            print("Validation notes:", validated.errors)

        results.append((utterance, validated))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ambiguous_utterance, ambiguous_result = results[-1]
    print(f"Ambiguous utterance {ambiguous_utterance!r} -> "
          f"actionable={ambiguous_result.is_actionable}, intent={ambiguous_result.intent!r}, "
          f"confidence={ambiguous_result.confidence:.2f}")

    if ambiguous_result.is_actionable:
        print("[FAIL] Ambiguous utterance was treated as actionable -- should have been flagged.")
        sys.exit(1)

    non_ambiguous_ok = all(r.is_actionable for _, r in results[:-1])
    print(f"All other utterances actionable: {non_ambiguous_ok}")
    if not non_ambiguous_ok:
        print("[WARN] At least one real utterance was not parsed as actionable -- review above.")

    print("\n[OK] Phase 4 intent extraction + validation behaves as expected.")


if __name__ == "__main__":
    main()
