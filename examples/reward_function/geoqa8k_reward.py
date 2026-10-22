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
    if grade_answer(answer, ground_truth):
        return 1.0
    if _symbolic_text_equivalent(answer, ground_truth):
        return 1.0
    return 1.0 if _numeric_unit_answer_equivalent(answer, ground_truth) else 0.0


def _normalize_symbolic_text(value: Any) -> str:
    def _normalize_number_literal(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            num = float(raw)
        except ValueError:
            return raw
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        return format(num, ".12g")

    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("−", "-")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("π", "\\pi")

    # 统一 sqrt 记号：5√{3} / 5√3 -> 5\sqrt{3}
    s = s.replace("√", "\\sqrt")
    s = re.sub(r"\\sqrt([0-9a-z])", r"\\sqrt{\1}", s)
    # 统一数字字面量：9.0\pi -> 9\pi
    s = re.sub(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)", _normalize_number_literal, s)
    return s


def _symbolic_text_equivalent(answer: Any, ground_truth: Any) -> bool:
    a = _normalize_symbolic_text(answer)
    g = _normalize_symbolic_text(ground_truth)
    if not a or not g:
        return False
    return a == g


def _normalize_unit_text(unit_text: str) -> str:
    s = unit_text.strip().lower()
    replacements = {
        "厘米": "cm",
        "公分": "cm",
        "毫米": "mm",
        "千米": "km",
        "公里": "km",
        "米": "m",
        "度": "deg",
        "°": "deg",
        "\\circ": "deg",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _parse_number_and_unit(value: Any) -> tuple[float, str] | None:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").lower()

    pattern = r"^([-+]?(?:\d+(?:\.\d+)?|\.\d+))([a-zA-Z\u4e00-\u9fff°\\]*)$"
    match = re.fullmatch(pattern, s)
    if not match:
        return None

    try:
        num = float(match.group(1))
    except ValueError:
        return None

    unit = _normalize_unit_text(match.group(2))
    return num, unit


def _numeric_unit_answer_equivalent(answer: Any, ground_truth: Any) -> bool:
    """
    数值+单位兜底：
    - 20 与 20° / 20度 视为等价
    - 58 与 58米 视为等价
    - 支持米/cm/mm/km 等常见长度单位换算
    """
    parsed_answer = _parse_number_and_unit(answer)
    parsed_gt = _parse_number_and_unit(ground_truth)
    if parsed_answer is None or parsed_gt is None:
        return False

    answer_num, answer_unit = parsed_answer
    gt_num, gt_unit = parsed_gt

    # 允许答案省略单位（例如 gt=58米，answer=58）
    if answer_unit == gt_unit or not answer_unit or not gt_unit:
        return abs(answer_num - gt_num) < 1e-9

    length_scale_to_m = {
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
        "km": 1e3,
    }
    angle_units = {"deg"}

    if answer_unit in length_scale_to_m and gt_unit in length_scale_to_m:
        answer_m = answer_num * length_scale_to_m[answer_unit]
        gt_m = gt_num * length_scale_to_m[gt_unit]
        return abs(answer_m - gt_m) < 1e-9

    if answer_unit in angle_units and gt_unit in angle_units:
        return abs(answer_num - gt_num) < 1e-9

    return False


# =========================
# Debug dump (optional)
# =========================
def dump_reward_inputs_to_txt(
        reward_inputs: list[dict],
        path: str = f"logs/geoqa8k/reward_inputs_{run_time}.txt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reward_inputs, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_reward_debug_records(
        reward_inputs: list[dict[str, Any]],
        scores: list[dict[str, float]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for reward_input, score in zip(reward_inputs, scores):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        records.append(
            {
                "question_id": reward_input.get("question_id"),
                "pre_answer": extract_boxed_content(response),
                "ground_truth": reward_input.get("ground_truth"),
                "scores": score,
                "response": reward_input.get("response"),
            }
        )
    return records


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
        train_path: str = f"logs/geoqa8k/train_question_acc_{run_time}.jsonl",
        test_path: str = f"logs/geoqa8k/test_question_acc_{run_time}.jsonl"
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
    # val_dataset_len = 601  #geo3k
    val_dataset_len = 754  # geoqa8k_rl

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

    debug_records = build_reward_debug_records(reward_inputs, scores)
    dump_reward_inputs_to_txt(debug_records)

    return scores

