"""test_lightrag.py — Kiểm tra nhanh pipeline LightRAG với mẫu nhỏ (Ollama).

Mục đích:
1. Nạp 2 đoạn văn mẫu nhỏ vào thư mục test riêng (light/test_storage).
2. Chạy thử query ở 4 mode: naive, local, global, hybrid.
3. Kiểm tra tính ổn định, tốc độ, và việc trích xuất entity/quan hệ trước khi chạy toàn bộ.
"""

import sys
import shutil
import time
import asyncio
from pathlib import Path

# Setup đường dẫn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from lightrag import QueryParam
from light.rag_light import build_rag, lightrag_answer
import config

TEST_STORAGE = ROOT / "light" / "test_storage"

SAMPLE_TEXT = """
Quang Trung (1753 – 16 tháng 9 năm 1792), tên húy là Hồ Thơm, sau đổi thành Nguyễn Huệ, là một chính trị gia, nhà quân sự tiêu biểu của Việt Nam. Ông là vị hoàng đế thứ hai của triều đại Tây Sơn.
Nguyễn Huệ cùng hai người anh em của mình là Nguyễn Nhạc và Nguyễn Lữ được gọi là Tây Sơn tam kiệt. Họ là lãnh tụ của cuộc khởi nghĩa Tây Sơn lật đổ chúa Nguyễn ở Đàng Trong, lật đổ chúa Trịnh và nhà Hậu Lê ở Đàng Ngoài, thống nhất đất nước.
Quang Trung đánh bại cuộc xâm lăng của quân Xiêm La từ phía Nam và quân Mãn Thanh từ phía Bắc trong trận Ngọc Hồi - Đống Đa năm 1789.
"""

TEST_QUERIES = [
    "Quang Trung tên khai sinh là gì và đánh bại những đội quân xâm lược nào?",
    "Tây Sơn tam kiệt gồm những ai?"
]

MODES = ["naive", "local", "global", "hybrid"]


async def main():
    print("=" * 60)
    print("BAT DAU KIEM TRA LIGHTRAG VOI MAU NHO")
    print(f"Ollama Host: {config.OLLAMA_HOST}")
    print(f"LLM: {config.LLM_MODEL} | Embed: {config.EMBED_MODEL}")
    print(f"Working Dir: {TEST_STORAGE}")
    print("=" * 60)

    # Dọn dẹp test storage cũ nếu có
    if TEST_STORAGE.exists():
        print(f"[*] Dang xoa test storage cu: {TEST_STORAGE}")
        shutil.rmtree(TEST_STORAGE, ignore_errors=True)
    TEST_STORAGE.mkdir(parents=True, exist_ok=True)

    # 1. Khởi tạo
    print("\n[1/3] Dang khoi tao LightRAG...")
    rag = await build_rag(working_dir=TEST_STORAGE)
    print("[+] Khoi tao storages thanh cong.")

    # 2. Nạp văn bản mẫu (Indexing)
    print("\n[2/3] Dang nap va trich xuat Knowledge Graph cho mau nho...")
    start_idx = time.perf_counter()
    await rag.ainsert(SAMPLE_TEXT)
    idx_time = time.perf_counter() - start_idx
    print(f"[+] Indexing mau nho thanh cong trong {idx_time:.2f} giay!")

    # 3. Query thử nghiệm
    print("\n[3/3] Dang truy van thu nghiem tren cac mode...")
    results = {}
    for q in TEST_QUERIES:
        print("-" * 50)
        print(f"Cau hoi: {q}")
        results[q] = {}
        for mode in MODES:
            start_q = time.perf_counter()
            try:
                # Query lấy câu trả lời
                ans = await rag.aquery(q, param=QueryParam(mode=mode))
                q_time = time.perf_counter() - start_q
                snippet = ans.strip().replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                print(f"  [Mode {mode:6s}] ({q_time:.2f}s) => {snippet}")
                results[q][mode] = {"status": "SUCCESS", "time": q_time, "ans": ans}
            except Exception as e:
                q_time = time.perf_counter() - start_q
                print(f"  [Mode {mode:6s}] [FAILED] ({q_time:.2f}s) => {e}")
                results[q][mode] = {"status": f"FAILED: {e}", "time": q_time}

    print("\n" + "=" * 60)
    print("TONG KET KIEM TRA LIGHTRAG")
    print(f"Indexing time: {idx_time:.2f}s")
    for q, q_data in results.items():
        print(f"Q: '{q}'")
        for m, m_data in q_data.items():
            print(f"   - {m:6s}: {m_data['status']} ({m_data.get('time', 0):.2f}s)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
