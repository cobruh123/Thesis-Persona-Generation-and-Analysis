import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from tqdm import tqdm


# ----------------------------
# Target LLM client (OpenAI-compatible)
# ----------------------------

@dataclass
class OpenAICompatConfig:
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 60
    max_retries: int = 5
    retry_backoff_sec: float = 1.5


class OpenAICompatChatClient:
    """
    Works with OpenAI API and OpenAI-compatible servers (LM Studio, vLLM, etc.)
    that implement: POST {api_base}/chat/completions
    """
    def __init__(self, cfg: OpenAICompatConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> str:
        url = self.cfg.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Some servers support seed; OpenAI supports it in some contexts.
        if seed is not None:
            payload["seed"] = seed

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = self.session.post(url, headers=headers, json=payload, timeout=self.cfg.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.cfg.retry_backoff_sec * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_sec * attempt)

        raise RuntimeError(f"LLM request failed after retries: {last_err}") from last_err


# ----------------------------
# Data loading helpers
# ----------------------------

def load_characters_from_dir(characters_dir: str) -> List[Tuple[str, str]]:
    """
    Returns list of (character_id, character_text).
    character_id defaults to filename.
    """
    items: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(characters_dir)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(characters_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            items.append((name, text))
    if not items:
        raise ValueError(f"No .txt character files found in: {characters_dir}")
    return items


def load_instruction_records(instructions_json_path: str) -> List[Dict[str, Any]]:
    """
    Expected schema (list of dicts):
    [
      {"category": "...", "technique": "...", "original_instruction": "..", "refined_instruction": "..."},
      ...
    ]

    Returns a normalized list where each item includes:
      - instruction_idx (int)
      - category (str)
      - technique (str)
      - original_instruction (str)
      - refined_instruction (str)   <-- used for target LLM prompt
    """
    with open(instructions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("instructions json must be a LIST of records.")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        category = (item.get("category") or "").strip()
        technique = (item.get("technique") or "").strip()
        original_instruction = (item.get("original_instruction") or "").strip()
        refined_instruction = (item.get("refined_instruction") or "").strip()

        if not refined_instruction:
            # if missing refined, we skip (your requirement says to use refined_instruction)
            continue

        out.append({
            "instruction_idx": i,
            "category": category,
            "technique": technique,
            "original_instruction": original_instruction,
            "refined_instruction": refined_instruction,
        })

    if not out:
        raise ValueError("No valid instruction records found (missing refined_instruction?).")

    return out


# ----------------------------
# Store progress (if something goes wrong)
# ----------------------------
def safe_append_result(out_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Crash-safe save:
    - writes to temp file
    - fsync
    - atomically replaces target json
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, out_path)


# ----------------------------
# Results IO (resume-friendly)
# ----------------------------

def _pair_key(character_id: str, instruction_idx: int) -> str:
    return f"{character_id}||{instruction_idx}"


def load_existing_results(out_path: str) -> Tuple[List[Dict[str, Any]], set]:
    """
    Returns (results_list, done_keys_set)
    done key is based on (character_id, instruction_idx)
    """
    if not os.path.exists(out_path):
        return [], set()

    with open(out_path, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = []

    if not isinstance(results, list):
        results = []

    done = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        cid = r.get("character_id")
        idx = r.get("instruction_idx")
        if isinstance(cid, str) and isinstance(idx, int):
            done.add(_pair_key(cid, idx))

    return results, done


def save_results(out_path: str, results: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


# ----------------------------
# Main evaluator
# ----------------------------

@dataclass
class EvalConfig:
    characters_dir: str
    instructions_json: str
    out_path: str = "response_from_target.json"

    max_tokens: int = 512
    temperature: float = 0.2
    seed: Optional[int] = 42

    limit_instructions: Optional[int] = None
    limit_characters: Optional[int] = None

    resume: bool = True
    save_every: int = 20  # write results.json every N new items


def run_instruction_evaluation(
    *,
    llm: OpenAICompatChatClient,
    cfg: EvalConfig,
) -> List[Dict[str, Any]]:
    characters: List[Tuple[str, str]] = load_characters_from_dir(cfg.characters_dir)
    instructions: List[Dict[str, Any]] = load_instruction_records(cfg.instructions_json)

    if cfg.limit_characters is not None:
        characters = characters[: max(0, cfg.limit_characters)]
    if cfg.limit_instructions is not None:
        instructions = instructions[: max(0, cfg.limit_instructions)]

    total = len(characters) * len(instructions)
    if total == 0:
        raise ValueError("Nothing to run: check characters_dir and instructions_json.")

    results: List[Dict[str, Any]] = []
    done = set()
    if cfg.resume:
        results, done = load_existing_results(cfg.out_path)

    pbar = tqdm(total=total, desc="Evaluating (character x instruction)", unit="item")

    
    try:
        for character_id, character_text in characters:
            for inst in instructions:
                inst_idx = inst["instruction_idx"]
                key = _pair_key(character_id, inst_idx)

                if cfg.resume and key in done:
                    continue

                refined_instruction = inst["refined_instruction"]

                try:
                    response = llm.generate(
                        system=character_text,
                        user=refined_instruction,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        seed=cfg.seed,
                    )
                    rec = {
                        "character_id": character_id,
                        "instruction_idx": inst_idx,
                        "character": character_text,
                        "category": inst.get("category", ""),
                        "technique": inst.get("technique", ""),
                        "original_instruction": inst.get("original_instruction", ""),
                        "refined_instruction": refined_instruction,
                        "response": response,
                        "model": llm.cfg.model,
                    }
                except Exception as e:
                    rec = {
                        "character_id": character_id,
                        "instruction_idx": inst_idx,
                        "character": character_text,
                        "category": inst.get("category", ""),
                        "technique": inst.get("technique", ""),
                        "original_instruction": inst.get("original_instruction", ""),
                        "refined_instruction": refined_instruction,
                        "response": "",
                        "model": llm.cfg.model,
                        "error": str(e),
                    }

                results.append(rec)
                done.add(key)
                pbar.update(1)

                # CRASH-SAFE SAVE
                safe_append_result(cfg.out_path, results)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Progress saved safely.")

    except Exception as e:
        print(f"\n[!] Unexpected error: {e}. Progress saved safely.")

    finally:
        pbar.close()
        safe_append_result(cfg.out_path, results)

    return results
