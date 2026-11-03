import json
from pathlib import Path

ROOT = Path("d:/Lab")
pre = json.load(open(ROOT / "eval_outputs/v3_hirag_hi_pre_nothink_score.json", encoding="utf-8"))["details"]
post = json.load(open(ROOT / "eval_outputs/v3_hirag_hi_post_nothink_score.json", encoding="utf-8"))["details"]

print(f"{'QID':<4} | {'Pre H1':<6} | {'Post H1':<7} | {'Pre F1':<6} | {'Post F1':<7} | Question")
print("-" * 80)
for q_pre, q_post in zip(pre, post):
    qid = q_pre["id"]
    h_pre = q_pre["hit@1"]
    h_post = q_post["hit@1"]
    f_pre = q_pre["token_f1"]
    f_post = q_post["token_f1"]
    if h_pre != h_post or abs(f_pre - f_post) > 0.15:
        qtext = q_pre.get("doc", "")
        print(f"{qid:<4} | {str(h_pre):<6} | {str(h_post):<7} | {f_pre:<6.3f} | {f_post:<7.3f} | {qtext}")
