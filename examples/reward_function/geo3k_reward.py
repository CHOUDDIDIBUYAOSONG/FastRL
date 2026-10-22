import re
import os
import json
from typing import Any
from collections import defaultdict

from mathruler.grader import extract_boxed_content, grade_answer

run_time = os.environ.get("Time", "no_time")
print(f"reward run_time is {run_time}")

# =========================
# Metadata
# =========================
REWARD_NAME = "math"
REWARD_TYPE = "batch"


# =========================
# Reward functions
# =========================
def format_reward(response: str) -> float:
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    return 0.5 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Geo3K:
    - ground_truth 直接是标准答案
    - 不需要 #### 抽取
    """
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


# =========================
# Debug dump (optional)
# =========================
def dump_reward_inputs_to_txt(
        reward_inputs: list[dict],
        path: str = f"logs/geo3k/reward_inputs_{run_time}.txt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reward_inputs, f, ensure_ascii=False, indent=2)
        f.write("\n")


# =========================
# Question-level metrics
# =========================
def compute_question_metrics(
        reward_inputs: list[dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """
    单次 rollout 内：
    每个 question_id 的
    - 平均 acc
    - 平均 overall reward
    """
    stats = defaultdict(lambda: {
        "correct": 0,
        "total": 0,
        "overall_sum": 0.0,
    })

    for r in reward_inputs:
        qid = r["question_id"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", r["response"])

        acc = accuracy_reward(response, r["ground_truth"])
        fmt = format_reward(response)
        overall = acc + fmt

        stats[qid]["total"] += 1
        stats[qid]["correct"] += int(acc)
        stats[qid]["overall_sum"] += overall

    return {
        qid: {
            "acc": s["correct"] / s["total"],
            "overall": s["overall_sum"] / s["total"],
        }
        for qid, s in stats.items()
    }


# =========================
# JSONL IO (acc + overall)
# =========================
def load_acc_jsonl(path: str) -> dict[int, dict[str, list[float]]]:
    acc_map: dict[int, dict[str, list[float]]] = {}
    if not os.path.exists(path):
        return acc_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = int(obj["question_id"])
            acc_map[qid] = {
                "acc": list(obj.get("acc", [])),
                "overall": list(obj.get("overall", [])),
            }
    return acc_map


def dump_acc_jsonl(
        acc_map: dict[int, dict[str, list[float]]],
        path: str
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, metrics in acc_map.items():
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "acc": metrics["acc"],
                        "overall": metrics["overall"],
                    },
                    ensure_ascii=False
                ) + "\n"
            )


# =========================
# Update train / test logs
# =========================
def update_question_acc_jsonl(
        reward_inputs: list[dict[str, Any]],
        val_dataset_len: int,
        train_path: str = f"logs/geo3k/train_question_acc_{run_time}.jsonl",
        test_path: str = f"logs/geo3k/test_question_acc_{run_time}.jsonl"
):
    train_inputs = [r for r in reward_inputs if r["question_id"] >= val_dataset_len]
    test_inputs = [r for r in reward_inputs if r["question_id"] < val_dataset_len]

    for inputs, path in [(train_inputs, train_path), (test_inputs, test_path)]:
        if not inputs:
            continue

        acc_map = load_acc_jsonl(path)
        latest_metrics = compute_question_metrics(inputs)

        for qid, m in latest_metrics.items():
            if qid not in acc_map:
                acc_map[qid] = {"acc": [], "overall": []}
            acc_map[qid]["acc"].append(m["acc"])
            acc_map[qid]["overall"].append(m["overall"])

        dump_acc_jsonl(acc_map, path)


# =========================
# Main entry
# =========================
def compute_score(
        reward_inputs: list[dict[str, Any]],
        format_weight: float = 0.5
) -> list[dict[str, float]]:
    # dump_reward_inputs_to_txt(reward_inputs)
    val_dataset_len = 601  # geo3k
    # val_dataset_len = 754  #geoqa8k_rl

    # question-level acc & overall for GRESO！
    update_question_acc_jsonl(
        reward_inputs,
        val_dataset_len=val_dataset_len
    )

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

        scores.append({
            "overall": accuracy_score + format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return scores