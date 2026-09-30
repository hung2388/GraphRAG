# GraphRAG Project Workspace

Dự án xây dựng và đánh giá hệ thống **GraphRAG (Knowledge Graph Retrieval-Augmented Generation)** sử dụng Neo4j, Ollama (LLMs) và các kỹ thuật truy vấn đồ thị tri thức.

---

## 📁 Cấu trúc thư mục

```text
d:\Lab\
├── data/
│   ├── input/                     # Dữ liệu văn bản nguồn (.txt) để trích xuất tri thức
│   │   ├── src.txt
│   │   └── quang_trung.txt
│   ├── cache/                     # Bộ nhớ đệm và nhật ký xử lý KG
│   │   ├── kg_cache/              # Cache JSON trích xuất thực thể & quan hệ theo chunk
│   │   ├── community_summaries.json # Tóm tắt phân cụm đồ thị (Global Search)
│   │   └── failed_chunks.txt      # Ghi nhận các chunk xử lý lỗi/cần retry
│   └── evaluation/                # Tập dữ liệu kiểm thử
│       └── test_questions_groundtruth.json
│
├── reports/                       # Báo cáo và kết quả thực nghiệm
│   └── evaluation_report.json     # Kết quả benchmark chi tiết từng câu hỏi
│
├── docs/                          # Tài liệu & Trực quan hóa
│   ├── slides/                    # Các bài thuyết trình về GraphRAG
│   │   ├── graphRAG2.pptx
│   │   ├── graphRAG2 [Autosaved].pptx
│   │   └── graphRAG_KG.pptx
│   └── pipeline_visual.html       # Giao diện trực quan hóa quy trình pipeline
│
├── tests/                         # Scripts kiểm tra kết nối & module thử nghiệm
│   ├── test_connection.py         # Kiểm tra kết nối Neo4j
│   ├── test_graphrag.py           # Thử nghiệm truy vấn GraphRAG với Gemini
│   └── testolama.py               # Kiểm tra gọi mô hình qua Ollama
│
├── kg_construction2.py            # Pipeline chính: Phân đoạn văn bản, trích xuất KG, nạp vào Neo4j & truy vấn
├── evaluation.py                  # Script đánh giá chất lượng (LLM-as-a-Judge, F1, EM)
├── result_plot.py                 # Vẽ biểu đồ trực quan hóa kết quả đánh giá
├── graph_traversal.py             # Script duyệt láng giềng / truy vấn đồ thị
├── .env                           # Cấu hình biến môi trường (Neo4j, API Keys,...)
└── README.md                      # Hướng dẫn dự án
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
