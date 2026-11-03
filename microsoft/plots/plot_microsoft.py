import os
import sys
import subprocess

# Đảm bảo in tiếng Việt trên console Windows không bị UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Kiểm tra matplotlib & numpy trong môi trường hiện tại
# Nếu người dùng chạy bằng môi trường khác (như .venv-ms), tự động chuyển sang Python đã có sẵn matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    candidate_pythons = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "Scripts", "python.exe")),
        r"D:\Lab\.venv\Scripts\python.exe",
        r"C:\Users\vusin\AppData\Local\Programs\Python\Python312\python.exe",
        "python",
    ]
    found_exec = None
    for py_exec in candidate_pythons:
        if os.path.exists(py_exec) or py_exec == "python":
            try:
                chk = subprocess.run([py_exec, "-c", "import matplotlib"], capture_output=True)
                if chk.returncode == 0:
                    found_exec = py_exec
                    break
            except Exception:
                continue

    if found_exec:
        cmd = [found_exec, os.path.abspath(__file__)] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        print("[!] Không tìm thấy môi trường Python nào đã cài matplotlib.")
        sys.exit(1)

import json

# 1. Đọc dữ liệu từ file báo cáo (hỗ trợ truyền file qua CLI)
if len(sys.argv) > 1:
    REPORT_FILE = sys.argv[1]
else:
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "evaluation_report_microsoft.json")),
        os.path.join("microsoft", "reports", "evaluation_report_microsoft.json"),
        os.path.join("reports", "evaluation_report_microsoft.json"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "baseline_ms_graphrag_78q", "reports", "evaluation_report_microsoft.json")),
    ]
    REPORT_FILE = next((c for c in candidates if os.path.exists(c)), candidates[0])

print(f"[*] Đang hiển thị kết quả cho: {REPORT_FILE}")

try:
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Không tìm thấy file '{REPORT_FILE}'.")
    sys.exit(1)

if not data:
    print("File JSON trống.")
    sys.exit(1)

# Xử lý dữ liệu (hỗ trợ cả định dạng dict mới và list cũ)
if isinstance(data, dict):
    summary = data.get("summary", {})
    details = data.get("details", [])
    num_questions = data.get("total_scored", len(details))
else:
    details = data
    num_questions = len(details)
    summary = {}

# ============================================================
# BIỂU ĐỒ 1: CỘT ĐIỂM TRUNG BÌNH (LLM Metrics & Retrieval/Overlap)
# ============================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig1.canvas.manager.set_window_title("Microsoft GraphRAG - Điểm Trung Bình")

if summary:
    # 1. Thang điểm 1 - 5 cho LLM-as-a-Judge
    llm_labels = ["Context Sufficiency", "Faithfulness"]
    avg_llm = [
        summary.get("context_sufficiency", 0.0) or 0.0,
        summary.get("faithfulness", 0.0) or 0.0,
    ]

    # 2. Thang điểm 0 - 1 cho Retrieval, Correctness và Token F1
    trad_labels = ["Hit@1", "Hit@3", "Hit@5", "Recall@5", "Entity Cov", "Correctness", "KP Recall", "Token F1"]
    avg_trad = [
        summary.get("hit@1", 0.0) or 0.0,
        summary.get("hit@3", 0.0) or 0.0,
        summary.get("hit@5", 0.0) or 0.0,
        summary.get("recall@5", 0.0) or 0.0,
        summary.get("entity_coverage", 0.0) or 0.0,
        summary.get("answer_correctness", 0.0) or 0.0,
        summary.get("key_point_recall", 0.0) or 0.0,
        summary.get("token_f1", 0.0) or 0.0,
    ]
else:
    # Fallback cho định dạng cũ
    llm_metrics = ["context_relevance", "context_sufficiency", "faithfulness", "answer_relevance"]
    llm_labels = ["Context Relevance", "Context Sufficiency", "Faithfulness", "Answer Relevance"]
    llm_scores = {m: [] for m in llm_metrics}
    for r in details:
        d = r.get("metrics", r.get("scores", {}))
        for m in llm_metrics:
            llm_scores[m].append(d.get(m, 0.0))
    avg_llm = [sum(llm_scores[m]) / max(num_questions, 1) for m in llm_metrics]

    trad_metrics = ["f1_score"]
    trad_labels = ["Token F1 Score"]
    trad_scores = {m: [] for m in trad_metrics}
    for r in details:
        d = r.get("metrics", r.get("scores", {}))
        for m in trad_metrics:
            trad_scores[m].append(d.get(m, 0.0))
    avg_trad = [sum(trad_scores[m]) / max(num_questions, 1) for m in trad_metrics]

# Cột cho LLM-as-a-Judge (1-5)
bars1 = ax1.bar(llm_labels, avg_llm, color=["#2b5c8f", "#e26d5c"], width=0.45)
ax1.set_ylim(0, 5.5)
ax1.set_title(f"MICROSOFT GRAPHRAG: LLM-as-a-Judge Metrics (Thang 1-5)\nTổng: {num_questions} câu", fontsize=12)
ax1.set_ylabel("Điểm số", fontsize=11)
ax1.grid(axis="y", linestyle="--", alpha=0.6)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, y + 0.1, f"{y:.2f}", ha="center", fontweight="bold")

# Cột cho Retrieval & Overlap (0-1)
bars2 = ax2.bar(
    trad_labels,
    avg_trad,
    color=["#3a86ff", "#00b4d8", "#06d6a0", "#2a9d8f", "#7209b7", "#ef476f", "#ffd166", "#f77f00"],
    width=0.55,
)
ax2.set_ylim(0, 1.15)
ax2.set_title("Accuracy & Retrieval Metrics (Thang 0-1)", fontsize=12)
ax2.set_ylabel("Tỷ lệ", fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.6)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.02, f"{y:.2f}" if y >= 0.01 else "0", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()

# ============================================================
# BIỂU ĐỒ 2: ĐƯỜNG PHỔ ĐIỂM TỪNG CÂU HỎI
# ============================================================
if details:
    x_axis = np.arange(1, len(details) + 1)
    fig2 = plt.figure(figsize=(14, 6))
    fig2.canvas.manager.set_window_title("Microsoft GraphRAG - Phổ Điểm Chi Tiết")

    if summary:
        suff_scores = [d.get("context_sufficiency", 0) or 0 for d in details]
        faith_scores = [d.get("faithfulness", 0) or 0 for d in details]
        corr_scores = [(d.get("answer_correctness", 0) or 0) * 5.0 for d in details]

        plt.plot(x_axis, suff_scores, label="Context Sufficiency (1-5)", color="#2b5c8f", marker="o", markersize=4, linewidth=1.2, alpha=0.85)
        plt.plot(x_axis, faith_scores, label="Faithfulness (1-5)", color="#e26d5c", marker="s", markersize=4, linewidth=1.2, alpha=0.85)
        plt.plot(x_axis, corr_scores, label="Answer Correctness (x5 scale)", color="#ef476f", marker="^", markersize=4, linewidth=1.2, alpha=0.85)
    else:
        colors = ["#2b5c8f", "#e26d5c", "#38b000", "#7209b7"]
        markers = ["o", "s", "^", "D"]
        for i, m in enumerate(llm_metrics):
            plt.plot(x_axis, llm_scores[m], label=llm_labels[i], color=colors[i], marker=markers[i], linewidth=1.5, alpha=0.8)

    plt.ylim(0, 5.5)
    plt.title(f"Microsoft GraphRAG - Phổ điểm chi tiết từng câu hỏi (1 - {len(details)})", fontsize=13, pad=12)
    plt.xlabel("Thứ tự câu hỏi (Index)", fontsize=11)
    plt.ylabel("Điểm (1-5)", fontsize=11)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

# Mở các cửa sổ figure matplotlib tương tác
plt.show()