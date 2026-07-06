"""S1.v2 §1 — 기존 런 로그에서 SR@220 소급 산출, §8 JSON에 필드 추가.

로그의 "[task N] ep M | SUCCESS|fail | steps T" 라인을 파싱해
sr_at_220(성공이면서 완료 스텝 ≤220)을 run JSON에 소급 기재한다.
주의: 스텝 카운트는 wait 스텝 제외(정책 스텝만) — OpenVLA max_steps 의미와 동일.

사용: python src/diagnosis/retrofit_sr220.py <log> <json> [--dry]
"""
import argparse
import json
import re
import sys
from pathlib import Path

PAT = re.compile(r"\[task (\d+)\] ep\s*(\d+) \| (SUCCESS|fail)\s*\| steps (\d+)")


def parse(log_path):
    per_task = {}
    for m in PAT.finditer(Path(log_path).read_text()):
        tid, ep, res, steps = int(m[1]), int(m[2]), m[3], int(m[4])
        per_task.setdefault(tid, []).append((res == "SUCCESS", steps))
    return per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("json_path")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    eps = parse(args.log)
    r = json.loads(Path(args.json_path).read_text())
    n_expected = r["eval"]["n_per_task"]
    task_keys = list(r["eval"]["per_task"].keys())
    per220 = {}
    for i, tk in enumerate(task_keys):
        recs = eps.get(int(tk), [])
        # 로그에 같은 태스크가 여러 런 섞였을 수 있음 — 마지막 n_expected개 사용
        recs = recs[-n_expected:]
        if len(recs) != n_expected:
            print(f"경고: task {tk} 로그 {len(recs)}/{n_expected}개 — 스킵")
            continue
        per220[tk] = sum(ok and st <= 220 for ok, st in recs) / len(recs)
        r["eval"]["per_task"][tk]["sr_at_220"] = per220[tk]
        r["eval"]["per_task"][tk]["episodes"] = [
            {"ok": ok, "steps": st} for ok, st in recs]
    if per220:
        r["eval"]["sr_at_220"] = round(sum(per220.values()) / len(per220), 4)
        r.setdefault("notes", "")
        if "sr_at_220 retrofitted" not in r["notes"]:
            r["notes"] += " | sr_at_220 retrofitted from log (S1.v2 §1)"
    print(f"{Path(args.json_path).name}: sr_at_220 = {r['eval'].get('sr_at_220')} "
          f"(sr = {r['eval']['suite_sr']})")
    if not args.dry:
        Path(args.json_path).write_text(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
