"""tools/plot_adapter_comparison.py

Vẽ 6 biểu đồ so sánh độc lập (6 Comparison Figures) cho 6 cặp cấu hình:
1. HiRAG (No-Thinking)
2. HiRAG (Thinking)
3. LightRAG (No-Thinking)
4. LightRAG (Thinking)
5. MS GraphRAG (No-Thinking)
6. MS GraphRAG (Thinking)

Mỗi biểu đồ tập trung so sánh toàn diện các chỉ số trước (Pre-Adapter) và sau (Post-Adapter):
- Nhóm Retrieval: Hit@1, Hit@3, Hit@5, Recall@1, Recall@3, Recall@5
- Nhóm Generation Quality: Token F1, Token Precision, Token Recall, Entity Coverage
- Nhóm Hiệu năng thực thi: Latency (s) & Token Consumption
- Phân rã theo Question Type (Single-hop, Multi-attribute, Comparison, Summarization, Factoid)
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Kiểm tra matplotlib & numpy trong môi trường hiện tại
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    candidate_pythons = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe")),
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


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval_outputs"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Danh sách 6 cặp so sánh
PAIRS = [
    {
        "id": "1_hirag_nothink",
        "title": "HiRAG (No-Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_hirag_hi_pre_nothink_score.json",
        "post_file": EVAL_DIR / "v3_hirag_hi_post_nothink_score.json",
    },
    {
        "id": "2_hirag_thinking",
        "title": "HiRAG (Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_hirag_hi_pre_thinking_score.json",
        "post_file": EVAL_DIR / "v3_hirag_hi_post_thinking_score.json",
    },
    {
        "id": "3_lightrag_nothink",
        "title": "LightRAG (No-Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_lightrag_hybrid_pre_nothink_score.json",
        "post_file": EVAL_DIR / "v3_lightrag_hybrid_post_nothink_score.json",
    },
    {
        "id": "4_lightrag_thinking",
        "title": "LightRAG (Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_lightrag_hybrid_pre_thinking_score.json",
        "post_file": EVAL_DIR / "v3_lightrag_hybrid_post_thinking_score.json",
    },
    {
        "id": "5_ms_graphrag_nothink",
        "title": "MS GraphRAG (No-Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_ms_graphrag_local_pre_nothink_score.json",
        "post_file": EVAL_DIR / "v3_ms_graphrag_local_post_nothink_score.json",
    },
    {
        "id": "6_ms_graphrag_thinking",
        "title": "MS GraphRAG (Thinking): Pre-Adapter vs Post-Adapter",
        "pre_file": EVAL_DIR / "v3_ms_graphrag_local_pre_thinking_score.json",
        "post_file": EVAL_DIR / "v3_ms_graphrag_local_post_thinking_score.json",
    },
]

def load_data(filepath: Path):
    if not filepath.exists():
        print(f"[!] Warning: File không tồn tại: {filepath}")
        return {}, {}, []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary", {}), data.get("by_question_type", {}), data.get("details", [])

# Cấu hình thẩm mỹ
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

COLOR_PRE = "#2b5c8f"    # Xanh dương trầm (Pre-adapter base)
COLOR_POST = "#06d6a0"   # Xanh lục ngọc (Post-adapter)
COLOR_DELTA_POS = "#2a9d8f"
COLOR_DELTA_NEG = "#e76f51"

saved_files = []

for idx, pair in enumerate(PAIRS, 1):
    pre_sum, pre_qtypes, pre_details = load_data(pair["pre_file"])
    post_sum, post_qtypes, post_details = load_data(pair["post_file"])

    # Tạo figure kích thước chuẩn 16:9 cho Slide (15 x 8.5 inches)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    fig.suptitle(pair["title"], fontsize=15, fontweight='bold', y=0.98)
    fig.canvas.manager.set_window_title(f"Comparison {idx}: {pair['title']}")

    # -------------------------------------------------------------
    # SUBPLOT 1: TẤT CẢ CÁC CHỈ SỐ RETRIEVAL (Hit@1,3,5 & Recall@1,3,5)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    retrieval_keys = ["hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5"]
    retrieval_labels = ["Hit@1", "Hit@3", "Hit@5", "Recall@1", "Recall@3", "Recall@5"]
    
    vals_pre_ret = [pre_sum.get(k, 0.0) or 0.0 for k in retrieval_keys]
    vals_post_ret = [post_sum.get(k, 0.0) or 0.0 for k in retrieval_keys]
    
    x_ret = np.arange(len(retrieval_labels))
    w = 0.35
    
    b1_pre = ax1.bar(x_ret - w/2, vals_pre_ret, w, label="Pre-Adapter", color=COLOR_PRE, alpha=0.9, edgecolor="black", linewidth=0.5)
    b1_post = ax1.bar(x_ret + w/2, vals_post_ret, w, label="Post-Adapter", color=COLOR_POST, alpha=0.9, edgecolor="black", linewidth=0.5)
    
    ax1.set_title("(A) Retrieval Metrics (Hit@k & Recall@k)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x_ret)
    ax1.set_xticklabels(retrieval_labels, fontsize=10)
    ax1.set_ylabel("Tỷ lệ (0.0 - 1.0)", fontsize=10)
    max_ret = max(vals_pre_ret + vals_post_ret + [0.1])
    ax1.set_ylim(0, min(1.05, max_ret * 1.25))
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)
    
    for bar in b1_pre:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold')
    for bar in b1_post:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold', color="#016648")

    # -------------------------------------------------------------
    # SUBPLOT 2: GENERATION & COVERAGE (Token F1, Prec, Rec, Entity Cov)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    gen_keys = ["token_f1", "token_precision", "token_recall", "entity_coverage"]
    gen_labels = ["Token F1", "Precision", "Recall", "Entity Cov"]
    
    vals_pre_gen = [pre_sum.get(k, 0.0) or 0.0 for k in gen_keys]
    vals_post_gen = [post_sum.get(k, 0.0) or 0.0 for k in gen_keys]
    
    x_gen = np.arange(len(gen_labels))
    b2_pre = ax2.bar(x_gen - w/2, vals_pre_gen, w, label="Pre-Adapter", color="#e76f51", alpha=0.9, edgecolor="black", linewidth=0.5)
    b2_post = ax2.bar(x_gen + w/2, vals_post_gen, w, label="Post-Adapter", color="#f4a261", alpha=0.9, edgecolor="black", linewidth=0.5)
    
    ax2.set_title("(B) Generation Quality & Entity Coverage", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x_gen)
    ax2.set_xticklabels(gen_labels, fontsize=10)
    ax2.set_ylabel("Điểm số / Tỷ lệ", fontsize=10)
    max_gen = max(vals_pre_gen + vals_post_gen + [0.1])
    ax2.set_ylim(0, min(1.05, max_gen * 1.25))
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)
    
    for bar in b2_pre:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold')
    for bar in b2_post:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold', color="#a34700")

    # -------------------------------------------------------------
    # SUBPLOT 3: SO SÁNH THEO QUESTION TYPE (F1 hoặc Hit@1 theo loại câu hỏi)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    common_qtypes = ["factoid", "single_hop", "multi_attribute", "comparison", "summarization"]
    qtype_labels = ["Factoid", "Single-hop", "Multi-attrib", "Comparison", "Summary"]
    
    pre_qt_f1 = [pre_qtypes.get(qt, {}).get("token_f1", 0.0) for qt in common_qtypes]
    post_qt_f1 = [post_qtypes.get(qt, {}).get("token_f1", 0.0) for qt in common_qtypes]
    
    x_qt = np.arange(len(qtype_labels))
    b3_pre = ax3.bar(x_qt - w/2, pre_qt_f1, w, label="Pre Token F1", color="#457b9d", alpha=0.9, edgecolor="black", linewidth=0.5)
    b3_post = ax3.bar(x_qt + w/2, post_qt_f1, w, label="Post Token F1", color="#a8dadc", alpha=0.9, edgecolor="black", linewidth=0.5)
    
    ax3.set_title("(C) Token F1 theo từng loại câu hỏi (Question Type)", fontsize=11, fontweight='bold', pad=10)
    ax3.set_xticks(x_qt)
    ax3.set_xticklabels(qtype_labels, fontsize=10)
    ax3.set_ylabel("Token F1", fontsize=10)
    max_qt = max(pre_qt_f1 + post_qt_f1 + [0.1])
    ax3.set_ylim(0, max_qt * 1.3)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)
    
    for bar in b3_pre:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, y + 0.005, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold')
    for bar in b3_post:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, y + 0.005, f"{y:.3f}", ha='center', fontsize=8, fontweight='bold', color="#1d3557")

    # -------------------------------------------------------------
    # SUBPLOT 4: DELTA SUMMARY & HIỆU NĂNG THỰC THI (Latency & % Tăng giảm)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    
    # Delta của 5 chỉ số then chốt
    metric_keys = ["hit@1", "hit@5", "recall@5", "token_f1", "entity_coverage"]
    metric_names = ["Δ Hit@1", "Δ Hit@5", "Δ Rec@5", "Δ F1", "Δ Cov"]
    
    deltas = [(post_sum.get(k, 0.0) or 0.0) - (pre_sum.get(k, 0.0) or 0.0) for k in metric_keys]
    
    bar_colors = [COLOR_DELTA_POS if d >= 0 else COLOR_DELTA_NEG for d in deltas]
    x_delta = np.arange(len(metric_names))
    b4 = ax4.bar(x_delta, deltas, width=0.45, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax4.axhline(0, color="gray", linewidth=1)
    
    ax4.set_title("(D) Tăng trưởng Delta (Post - Pre) & Độ trễ (Latency)", fontsize=11, fontweight='bold', pad=10)
    ax4.set_xticks(x_delta)
    ax4.set_xticklabels(metric_names, fontsize=10)
    ax4.set_ylabel("Chênh lệch tuyệt đối", fontsize=10)
    
    min_d, max_d = min(deltas + [-0.05]), max(deltas + [0.05])
    ax4.set_ylim(min_d * 1.4, max_d * 1.4)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in b4:
        y = bar.get_height()
        offset = 0.006 if y >= 0 else -0.015
        ax4.text(bar.get_x() + bar.get_width()/2, y + offset, f"{y:+.3f}", ha='center', fontsize=9, fontweight='bold')

    # Thêm box thông số Latency và tổng câu hỏi
    lat_pre = pre_sum.get("avg_latency_sec", 0.0) or 0.0
    lat_post = post_sum.get("avg_latency_sec", 0.0) or 0.0
    lat_delta = lat_post - lat_pre
    
    info_box = (
        f"Độ trễ trung bình:\n"
        f"• Pre-Adapter:  {lat_pre:.1f} s/query\n"
        f"• Post-Adapter: {lat_post:.1f} s/query\n"
        f"• Chênh lệch:   {lat_delta:+.1f} s\n"
        f"Tổng số câu test: {len(pre_details)} câu"
    )
    ax4.text(0.96, 0.94, info_box, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_file = PLOTS_DIR / f"comparison_{pair['id']}.png"
    fig.savefig(out_file, dpi=300)
    saved_files.append(out_file)
    print(f"[*] Đã xuất hình ảnh: {out_file}")

print(f"\n[✓] Hoàn thành xuất 6 biểu đồ so sánh chi tiết!")
print(f"[*] Đang hiển thị các cửa sổ Matplotlib...")
plt.show()
