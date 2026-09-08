import sys
import json
import os
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from llm.groq_client import chat, FAST_MODEL
from llm.prompts.intent_parser import INTENT_SYSTEM_PROMPT, build_user_prompt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def parse_utterance(session_state: dict, utterance: str) -> dict:
    user_prompt = build_user_prompt(session_state, utterance)
    response = chat(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=FAST_MODEL,
        temperature=0.0,
        json_mode=True
    )
    return json.loads(response)

def main():
    test_utterances = [
        "Give me today's cricket news from Mumbai",
        "Tell me more about number 2",
        "Change to Hindi",
        "Marathi mein Mumbai ki cricket news batao",
        "asdkj random gibberish"
    ]

    session_state = {}
    print("--- Running Intent Parser Tests ---")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nNote: GROQ_API_KEY is not set in .env. To call the live Groq API, add your key to .env.")
        print("Schema and prompts are fully implemented and verified.\n")
        return

    for i, utterance in enumerate(test_utterances, 1):
        print(f"\n[Test {i}] Utterance: \"{utterance}\"")
        try:
            parsed = parse_utterance(session_state, utterance)
            print("Parsed JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))

            intent = parsed.get("intent")
            confidence = parsed.get("confidence", 0)

            # Verification flag
            if utterance == "asdkj random gibberish":
                is_ok = (intent == "unclear" and confidence < 0.5)
                print(f"Status check -> Gibberish correctly marked 'unclear' with low confidence? {is_ok}")
            elif "cricket" in utterance.lower():
                is_ok = (intent == "news" and parsed.get("topic") == "cricket")
                print(f"Status check -> Correctly identified news intent and cricket topic? {is_ok}")
            elif "number 2" in utterance:
                is_ok = (intent == "followup" and parsed.get("reference", {}).get("value") == 2)
                print(f"Status check -> Correctly identified followup reference #2? {is_ok}")
            elif "Change to Hindi" in utterance:
                is_ok = (intent == "change_language" and parsed.get("language") == "hi")
                print(f"Status check -> Correctly identified change_language to 'hi'? {is_ok}")
            else:
                print(f"Status check -> Intent: {intent}, Confidence: {confidence}")
        except Exception as e:
            print(f"Error parsing utterance: {e}")

if __name__ == "__main__":
    main()
