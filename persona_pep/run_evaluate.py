import os
import argparse

from evaluator import (
    OpenAICompatConfig,
    OpenAICompatChatClient,
    EvalConfig,
    run_instruction_evaluation,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--characters_dir", required=True)
    ap.add_argument("--instructions_json", required=True)
    ap.add_argument("--out", default="results.json")

    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--api_base", default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))

    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--limit_instructions", type=int, default=None)
    ap.add_argument("--limit_characters", type=int, default=None)

    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=20)

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY env var or pass --api_key.")

    llm_cfg = OpenAICompatConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
    )
    llm = OpenAICompatChatClient(llm_cfg)

    eval_cfg = EvalConfig(
        characters_dir=args.characters_dir,
        instructions_json=args.instructions_json,
        out_path=args.out,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        limit_instructions=args.limit_instructions,
        limit_characters=args.limit_characters,
        resume=not args.no_resume,
        save_every=args.save_every,
    )

    run_instruction_evaluation(llm=llm, cfg=eval_cfg)


if __name__ == "__main__":
    main()
