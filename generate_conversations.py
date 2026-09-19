"""
Runs the RQ2 paired benchmark: for every scripted interview,
drives an identical turn-by-turn conversation against each of the four systems
and saves the resulting transcripts.

Usage:
    python generate_conversations.py --system gpt5_sol
    python generate_conversations.py --system gemini_pro
    python generate_conversations.py --system llama3_8b_base
    python generate_conversations.py --system dentaltrain
    python generate_conversations.py --system all      # runs all four sequentially

Run one system at a time in practice (e.g. base Llama needs a GPU session,
GPT/Gemini need their respective API keys set) -- "all" is a convenience for
environments where every dependency is already available.
"""

import argparse
import json
import time
import traceback

import config
from model_clients import get_client
import firebase_store


def run_script_against_system(client, script: dict, system_key: str) -> dict:
    """Drives one scripted interview turn-by-turn against one system.
    Returns a transcript in the same {diagnostic, messages} shape as the
    training dataset, so it can be judged with the same tooling."""
    history = []
    delay = config.get_request_delay(system_key)

    for question in script["questions"]:
        history.append({"role": "user", "content": question})
        reply = client.send(script["system_prompt"], history)
        history.append({"role": "assistant", "content": reply})
        time.sleep(delay) 

    messages = [{"role": "system", "content": script["system_prompt"]}] + history
    return {
        "script_id": script["script_id"],
        "diagnostic": script["diagnostic"],
        "messages": messages,
    }


def run_system(system_key: str, scripts: list, force_rerun: bool = False) -> list:
    print(f"\n=== Running system: {system_key} ({len(scripts)} conversations) ===")

    if not force_rerun:
        try:
            completed = firebase_store.get_completed_script_ids(system_key)
        except Exception as e:
            print(f"  (could not check Firestore for already-completed scripts: {e})")
            completed = set()
        if completed:
            before = len(scripts)
            scripts = [s for s in scripts if s["script_id"] not in completed]
            print(f"  Resuming: {before - len(scripts)} already completed successfully, "
                  f"{len(scripts)} remaining")
        if not scripts:
            print("  Nothing left to run -- all scripts already completed for this system.")
            return []

    client = get_client(system_key)
    transcripts = []

    for i, script in enumerate(scripts, 1):
        print(f"  [{i}/{len(scripts)}] {script['script_id']}")
        try:
            transcript = run_script_against_system(client, script, system_key)
        except Exception as e:
            print(f"    !! FAILED: {e}")
            traceback.print_exc()
            transcript = {
                "script_id": script["script_id"],
                "diagnostic": script["diagnostic"],
                "error": str(e),
            }
        transcripts.append(transcript)

        try:
            firebase_store.save_transcript(system_key, transcript)
        except Exception as e:
            print(f"    !! Firestore write failed for {script['script_id']}: {e}")

        time.sleep(0.5)  # gentle rate-limit buffer; adjust per provider limits

    out_path = config.OUTPUT_DIR / f"{system_key}.json"
    existing = []
    if out_path.exists() and not force_rerun:
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_ids = {t["script_id"] for t in transcripts}
        existing = [t for t in existing if t["script_id"] not in existing_ids]
    all_transcripts = existing + transcripts
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_transcripts)} total transcripts to {out_path} "
          f"({len(transcripts)} from this run)")
    return all_transcripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=list(config.SYSTEMS.keys()) + ["all"],
        required=True,
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=config.CONVERSATIONS_PER_CLASS,
        help="Conversations per diagnostic class (ignored if --scripts-json is given)",
    )
    parser.add_argument(
        "--scripts-json",
        default=None,
        help="Path to a frozen, pre-approved scripts JSON (from scripts_to_json.py). "
             "If given, this is used instead of regenerating scripts from the dataset, "
             "so every system is driven from the exact same reviewed content.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run every script even if already completed successfully in Firestore "
             "(by default, already-completed scripts are skipped automatically).",
    )
    args = parser.parse_args()

    if args.scripts_json:
        with open(args.scripts_json, "r", encoding="utf-8") as f:
            scripts = json.load(f)
        print(f"Loaded {len(scripts)} pre-approved scripts from {args.scripts_json}")
    else:
        from data_loader import build_full_benchmark_scripts
        scripts = build_full_benchmark_scripts(n_per_class=args.n_per_class)
        print(f"Generated {len(scripts)} scripted interviews across {len(config.DIAGNOSTIC_CLASSES)} classes")

    if args.system == "all":
        for key in config.SYSTEMS:
            run_system(key, scripts, force_rerun=args.force_rerun)
    else:
        run_system(args.system, scripts, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()