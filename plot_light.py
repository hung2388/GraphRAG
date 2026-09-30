import os
import json
import matplotlib.pyplot as plt
import numpy as np

import sys
import glob

# 1. Đọc dữ liệu từ file báo cáo (hỗ trợ truyền file qua CLI)
REPORT_FILE = os.path.join("reports", "evaluation_report_light.json")

print(f"[*] Đang hiển thị kết quả cho: {REPORT_FILE}")

try:
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Không tìm thấy file '{REPORT_FILE}'.")
    exit()

if not results:
    print("File JSON trống.")
    exit()

num_questions = len(results)

# 2. Xử lý các chỉ số LLM (Thang điểm 1 - 5)
llm_metrics = ["context_relevance", "context_sufficiency", "faithfulness", "answer_relevance"]
llm_labels = ["Context Relevance", "Context Sufficiency", "Faithfulness", "Answer Relevance"]

llm_scores = {m: [] for m in llm_metrics}
for r in results:
    # Lấy từ "metrics", nếu không có thì thử lấy "scores"
    data_dict = r.get("metrics", r.get("scores", {}))
    for m in llm_metrics:
        llm_scores[m].append(data_dict.get(m, 0))

avg_llm = [sum(llm_scores[m]) / num_questions for m in llm_metrics]

# 3. Xử lý các chỉ số F1 và Exact Match (Thang điểm 0 - 1)
traditional_metrics = ["f1_score", "exact_match"]
traditional_labels = ["Token F1 Score", "Exact Match (EM)"]

trad_scores = {m: [] for m in traditional_metrics}
for r in results:
    data_dict = r.get("metrics", r.get("scores", {}))
    for m in traditional_metrics:
        trad_scores[m].append(data_dict.get(m, 0.0))

avg_trad = [sum(trad_scores[m]) / num_questions for m in traditional_metrics]

# ============================================================
# BIỂU ĐỒ 1: CỘT ĐIỂM TRUNG BÌNH (LLM Metrics & Retrieval/Overlap)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cột cho LLM-as-a-Judge (1-5)
bars1 = ax1.bar(llm_labels, avg_llm, color=['#2b5c8f', '#e26d5c', '#38b000', '#7209b7'])
ax1.set_ylim(0, 5.5)
ax1.set_title(f"LIGHTRAG: LLM-as-a-Judge Metrics (Thang 1-5)\nTổng: {num_questions} câu", fontsize=12)
ax1.set_ylabel("Điểm số", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 0.1, f"{y:.2f}", ha='center', fontweight='bold')

# Cột cho F1 và EM (0-1)
bars2 = ax2.bar(traditional_labels, avg_trad, color=['#f77f00', '#4a4e69'])
ax2.set_ylim(0, 1.15)
ax2.set_title("N-gram / Overlap Metrics (Thang 0-1)", fontsize=12)
ax2.set_ylabel("Tỷ lệ", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + 0.03, f"{y:.4f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# BIỂU ĐỒ 2: ĐƯỜNG PHỔ ĐIỂM TỪNG CÂU HỎI
# ============================================================
x_axis = np.arange(1, num_questions + 1)
plt.figure(figsize=(13, 6))

colors = ['#2b5c8f', '#e26d5c', '#38b000', '#7209b7']
markers = ['o', 's', '^', 'D']

for i, m in enumerate(llm_metrics):
    plt.plot(x_axis, llm_scores[m], label=llm_labels[i], color=colors[i], marker=markers[i], linewidth=1.5, alpha=0.8)

plt.ylim(0, 5.5)
plt.title(f"LightRAG - Phổ điểm chi tiết từng câu hỏi (1 - {num_questions})", fontsize=13, pad=12)
plt.xlabel("Thứ tự câu hỏi (Index)", fontsize=11)
plt.ylabel("Điểm (1-5)", fontsize=11)
plt.xticks(x_axis)
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()