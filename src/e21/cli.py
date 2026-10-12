import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .conditions import build_items
from .metrics import cohen_kappa, summarize
from .schema import load_codes, load_coverage


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(record: dict) -> tuple:
    return record["id"], record["model"], record["rep"]


def _process(todo: list, work, out: str, workers: int) -> None:
    """Run `work` over `todo`, appending each result as it lands so a crash loses nothing."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    failed = 0

    def safe(arg):
        try:
            return work(arg)
        except Exception as error:  # one bad call should not end a long run; rerun to retry it
            return error

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for done, result in enumerate(pool.map(safe, todo), start=1):
            if isinstance(result, Exception):
                failed += 1
                print(f"  failed: {type(result).__name__}: {result}")
            else:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")
    if failed:
        print(f"{failed} failed; run the same command again to retry them.")


def cmd_build(args) -> None:
    entries = load_codes(args.codes, args.allow_unverified)
    complete = load_coverage(args.coverage) if args.coverage else set()
    items = build_items(entries, complete, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = Counter(item["condition"] for item in items)
    print(f"{len(entries)} codes -> {len(items)} prompts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not complete:
        print("No --coverage given, so no transplant or phantom prompts were generated.")


def cmd_run(args) -> None:
    from .models import get_model

    grounder = None
    if args.model.startswith("grounded+"):
        from .grounding import Grounder

        if not args.codes:
            raise SystemExit("grounded+ models need --codes (and usually --coverage)")
        grounder = Grounder(load_codes(args.codes, args.allow_unverified),
                            load_coverage(args.coverage) if args.coverage else set())
    model = get_model(args.model, grounder)
    done = {_key(r) for r in read_jsonl(args.out)}
    todo = [(item, rep) for item in read_jsonl(args.items) for rep in range(args.reps)
            if (item["id"], model.name, rep) not in done]
    recorded = sum(1 for _, name, _ in done if name == model.name)
    print(f"{model.name}: {len(todo)} calls to make ({recorded} already recorded)")

    def work(pair):
        item, rep = pair
        return {**item, "model": model.name, "rep": rep, "response": model.ask(item["prompt"])}

    _process(todo, work, args.out, args.workers)


def cmd_judge(args) -> None:
    from .judge import Judge

    judge = Judge(args.judge)
    done = {_key(r) for r in read_jsonl(args.out)}
    todo = [r for r in read_jsonl(args.responses) if _key(r) not in done]
    print(f"judging {len(todo)} responses with {args.judge}")
    _process(todo, lambda r: {**r, "judge": args.judge, **judge.grade(r)}, args.out, args.workers)


def cmd_report(args) -> None:
    rows = summarize(read_jsonl(args.verdicts))
    print(f"{'model':40} {'condition':15} {'metric':17} {'rate':>6}  95% CI          n")
    for r in rows:
        print(f"{r['model']:40} {r['condition']:15} {r['metric']:17} {r['rate']:6.1%}  "
              f"[{r['lo']:.1%}, {r['hi']:.1%}]  {r['n']}")


def cmd_sample(args) -> None:
    """Export a blind sample for human annotation: judge labels are left out on purpose."""
    by_condition = defaultdict(list)
    for v in read_jsonl(args.verdicts):
        by_condition[v["condition"]].append(v)
    rng = random.Random(args.seed)
    per = max(1, args.n // max(1, len(by_condition)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "rep", "condition", "prompt", "reference", "official_action",
                         "response", "human_behavior", "human_risk"])
        for condition, records in sorted(by_condition.items()):
            for v in rng.sample(records, min(per, len(records))):
                gold = v["gold"]
                writer.writerow([v["id"], v["model"], v["rep"], condition, v["prompt"],
                                 gold.get("meaning") or json.dumps(gold.get("candidates", "")),
                                 gold.get("official_action", ""), v["response"], "", ""])
    print(f"wrote {args.out}; fill human_behavior and human_risk, then run `e21 agreement`")


def cmd_agreement(args) -> None:
    verdicts = {(v["id"], v["model"], str(v["rep"])): v for v in read_jsonl(args.verdicts)}
    judge_b, human_b, judge_r, human_r = [], [], [], []
    with open(args.annotations, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = verdicts.get((row["id"], row["model"], row["rep"]))
            if v is None or not row["human_behavior"].strip() or not row["human_risk"].strip():
                continue
            judge_b.append(v["behavior"])
            human_b.append(row["human_behavior"].strip())
            judge_r.append(v["risk"] >= 2)
            human_r.append(int(row["human_risk"]) >= 2)
    print(f"{len(judge_b)} annotated items")
    print(f"behavior        kappa = {cohen_kappa(judge_b, human_b):.3f}")
    print(f"harmful (risk>=2) kappa = {cohen_kappa(judge_r, human_r):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="e21")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("build", help="ground-truth table -> test prompts")
    p.add_argument("--codes", required=True)
    p.add_argument("--coverage")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("run", help="ask a model every prompt")
    p.add_argument("--items", required=True)
    p.add_argument("--model", required=True, help="e.g. anthropic:claude-opus-5-5, grounded+openai:<model>")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--codes", help="needed for grounded+ models")
    p.add_argument("--coverage")
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="label responses against the ground truth")
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", default="claude-opus-5-5")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="rates with confidence intervals")
    p.add_argument("--verdicts", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sample", help="export a blind sample for human annotation")
    p.add_argument("--verdicts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("agreement", help="judge-vs-human Cohen's kappa")
    p.add_argument("--verdicts", required=True)
    p.add_argument("--annotations", required=True)
    p.set_defaults(func=cmd_agreement)

    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as error:  # bad input data: show the message, not a traceback
        raise SystemExit(f"error: {error}")


if __name__ == "__main__":
    main()
