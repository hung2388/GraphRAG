# GraphRAG Project Workspace

Dự án xây dựng và đánh giá hệ thống **GraphRAG (Knowledge Graph Retrieval-Augmented Generation)** sử dụng Neo4j, Ollama (LLMs) và các kỹ thuật truy vấn đồ thị tri thức.

---

## 📁 Cấu trúc thư mục

```text
d:\Lab\
├── data/                           # Dữ liệu dùng chung cho cả 4 hệ thống RAG
│   ├── input/                     # Văn bản nguồn: quang_trung.txt, src.txt, love.txt
│   ├── evaluation/                # Bộ câu hỏi đánh giá benchmark (merged_questions.json, eval_v2.json)
│   └── cache/                     # Cache trung gian (token logs, v.v.)
│
├── eval_data_tools.py              # Tool kiểm tra span, schema và tính Hit@k / Recall@k
├── evaluation_metrics.py           # Module thước đo benchmark chuẩn hóa dùng chung
├── config.py                       # Cấu hình môi trường chung
├── tools/                          # Script tiện ích (dọn dẹp, so sánh token tiêu thụ)
│
├── HiRAG/                          # [1] HiRAG (Hierarchical Graph RAG)
│   ├── hirag/                     # Mã nguồn lõi HiRAG
│   ├── config.yaml                # Cấu hình mô hình cho HiRAG
│   ├── hi_Search_ollama.py        # Adapter chạy HiRAG với Ollama
│   ├── run_small_test.py          # Script kiểm thử pipeline HiRAG
│   ├── plots/                     # Biểu đồ đánh giá của HiRAG
│   └── reports/                   # Báo cáo đánh giá JSON của HiRAG
│
├── light/                          # [2] LightRAG
│   ├── rag_light.py               # Pipeline LightRAG
│   ├── evaluation_light.py        # Đánh giá LightRAG
│   ├── lightrag_storage/          # Lưu trữ graph & vector riêng của LightRAG
│   ├── plots/                     # Biểu đồ đánh giá của LightRAG (plot_light.py)
│   └── reports/                   # Báo cáo đánh giá JSON của LightRAG
│
├── microsoft/                      # [3] Microsoft GraphRAG (Bản chuẩn Microsoft)
│   ├── ms_graphrag.py             # Script điều khiển GraphRAG CLI với Ollama
│   ├── evaluation_microsoft.py    # Đánh giá Microsoft GraphRAG
│   ├── ragtest_ms/                # Workspace riêng của Microsoft GraphRAG
│   ├── plots/                     # Biểu đồ đánh giá (plot_microsoft.py)
│   └── reports/                   # Báo cáo đánh giá JSON (evaluation_report_microsoft.json)
│
└── my_rag_microsoft_style/         # [4] Bản tự phát triển (Custom Neo4j + MS Style)
    ├── rag_microsoft.py           # Pipeline trích xuất KG nạp Neo4j và truy vấn
    ├── evaluation_ms_graphrag.py   # Benchmark cho bản Custom Neo4j
    ├── plots/                     # Biểu đồ đánh giá (plot_ms_graphrag.py)
    └── reports/                   # Báo cáo đánh giá JSON (evaluation_report_ms_graphrag.json)
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Cấu hình biến môi trường (`.env`)
Tạo file `.env` (nếu chưa có) và cấu hình các thông số cần thiết:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=your_gemini_api_key  # Nếu dùng module thử nghiệm với Gemini
```

### 2. Xây dựng Knowledge Graph
Đặt các file văn bản nguồn `.txt` vào thư mục `data/input/`, sau đó chạy:
```bash
python kg_construction2.py
```

### 3. Đánh giá chất lượng (Evaluation Benchmark)
Chạy script benchmark hệ thống dựa trên tập câu hỏi ground truth trong `data/evaluation/`:
```bash
python evaluation.py
```
Kết quả sẽ được tự động lưu vào `reports/evaluation_report.json`.

### 4. Trực quan hóa kết quả đánh giá (Plotting)
Vẽ biểu đồ các chỉ số đánh giá (LLM-as-a-Judge, F1, Exact Match):
```bash
python result_plot.py
```

### 5. Chạy kiểm tra kết nối
```bash
python tests/test_connection.py
python tests/testolama.py
```
