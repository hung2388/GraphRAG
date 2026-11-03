"""Tai bai Wikipedia (tieng Anh) lam data train adapter, loc ro ri voi 3 tai lieu eval.

Ket qua:
    data/train_docs/<slug>.txt   van ban sach (da cat References/External links...)
    data/train_docs/manifest.csv group, requested, title, url, words, status

Bai nao nhac toi thuc the cua tai lieu eval (BLOCK) hoac qua ngan thi bi loai.
"""

import csv
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("data/train_docs")
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAGAdapterBenchmark/1.0 (nlp_research_lab@example.com)"}
MIN_WORDS = 600
SLEEP = 0.5

HISTORY = [
    # Lý dynasty & figures
    "Lý Thái Tổ", "Lý Thái Tông", "Lý Thánh Tông", "Lý Nhân Tông", "Lý Thường Kiệt", "Lý Chiêu Hoàng",
    # Trần dynasty & figures
    "Trần Thái Tông", "Trần Thánh Tông", "Trần Nhân Tông", "Trần Hưng Đạo", "Trần Thủ Độ",
    "Trần Quang Khải", "Trần Khánh Dư", "Phạm Ngũ Lão",
    # Hồ & Early periods
    "Hồ Quý Ly", "Lê Lợi", "Lê Thánh Tông", "Nguyễn Trãi", "Đinh Bộ Lĩnh", "Lê Hoàn", "Ngô Quyền",
    "Dương Vân Nga", "Lady Triệu", "Trưng Sisters", "Lý Nam Đế", "Khúc Thừa Dụ", "Mạc Đăng Dung",
    "Phùng Hưng", "Mai Thúc Loan", "Triệu Việt Vương", "Chu Văn An",
    # Early Dynasties & Figures
    "Đinh Tiên Hoàng", "Lê Đại Hành", "Dương Tam Kha", "Ngô Xương Ngập",
    "Trần Nghệ Tông", "Trần Duệ Tông", "Trần Thuận Tông", "Giản Định Đế", "Trùng Quang Đế",
    "Fourth Chinese domination of Vietnam",
    # Trịnh Lords & Nguyễn Lords (16th-17th century)
    "Trịnh Kiểm", "Trịnh Tùng", "Trịnh Tráng", "Trịnh Tạc", "Trịnh Căn", "Trịnh Cương",
    "Nguyễn Hoàng", "Nguyễn Phúc Nguyên", "Nguyễn Phúc Lan", "Nguyễn Phúc Tần", "Nguyễn Phúc Chu",
    # 19th Century anti-colonial figures & History texts
    "Phan Đình Phùng", "Hoàng Hoa Thám", "Trương Định", "Nguyễn Tri Phương", "Hoàng Diệu",
    "Phan Bội Châu", "Phan Châu Trinh", "Đại Việt sử ký toàn thư", "Bình Ngô đại cáo",
    # Major Battles & Campaigns
    "Battle of Bạch Đằng (938)", "Battle of Bạch Đằng (1288)", "Mongol invasions of Vietnam",
    "Lam Sơn uprising", "Battle of Chi Lăng", "Battle of Như Nguyệt",
    # Dynasties (non-overlapping)
    "Lý dynasty", "Trần dynasty", "Hồ dynasty", "Mạc dynasty",
    "Early Lê dynasty", "Đinh dynasty", "Ngô dynasty",
]

SONGS = [
    # Adele
    "Rolling in the Deep", "Someone like You (Adele song)", "Set Fire to the Rain", "Hello (Adele song)",
    "Send My Love (to Your New Lover)", "When We Were Young (Adele song)", "Water Under the Bridge", "Easy on Me",
    # Lady Gaga
    "Poker Face (Lady Gaga song)", "Bad Romance", "Born This Way (song)",
    "Just Dance (Lady Gaga song)", "Telephone (Lady Gaga song)", "Paprazzi (Lady Gaga song)",
    # Katy Perry
    "Firework (song)", "Teenage Dream (Katy Perry song)", "I Kissed a Girl", "California Gurls", "Roar (song)",
    # Beyoncé
    "Single Ladies (Put a Ring on It)", "Halo (Beyoncé song)", "If I Were a Boy", "Crazy in Love",
    # Taylor Swift
    "Fearless (Taylor Swift song)", "Love Story (Taylor Swift song)", "You Belong with Me",
    "Mine (Taylor Swift song)", "Back to December", "Shake It Off", "Blank Space",
    # Bruno Mars
    "Just the Way You Are (Bruno Mars song)", "Grenade (song)", "Locked Out of Heaven", "When I Was Your Man",
    "Treasure (Bruno Mars song)", "24K Magic (song)", "That's What I Like (song)",
    # Coldplay
    "Viva la Vida", "Fix You", "The Scientist (song)", "Yellow (Coldplay song)", "Clocks (song)",
    "Paradise (Coldplay song)", "A Sky Full of Stars", "Adventure of a Lifetime",
    # Ed Sheeran
    "Shape of You", "Thinking Out Loud", "Photograph (Ed Sheeran song)", "Castle on the Hill", "Perfect (Ed Sheeran song)",
    # Maroon 5
    "She Will Be Loved", "Sugar (Maroon 5 song)", "Payphone (song)", "Maps (Maroon 5 song)",
    # Queen
    "Bohemian Rhapsody", "Don't Stop Me Now", "Somebody to Love (Queen song)", "Under Pressure", "Radio Ga Ga",
    # Michael Jackson
    "Billie Jean", "Beat It", "Bad (song)", "Thriller (song)", "Smooth Criminal", "Black or White (song)",
    # Billie Eilish & Dua Lipa
    "Bad Guy (Billie Eilish song)", "Everything I Wanted (song)", "Happier Than Ever (song)",
    "New Rules (song)", "Don't Start Now", "Levitating (song)",
    # The Beatles
    "Hey Jude", "Let It Be (song)", "Yesterday (Beatles song)", "Come Together (song)",
    # Other iconic pop hits
    "Hey, Soul Sister", "Need You Now (Lady Antebellum song)", "Tik Tok (song)",
    "Royals (song)", "Happy (Pharrell Williams song)", "Get Lucky (Daft Punk song)",
    "Chandelier (song)", "Somebody That I Used to Know", "Counting Stars (OneRepublic song)",
]

# Thuc the cua 3 tai lieu eval (khong dau, lowercase). Bai nao nhac toi -> loai de tranh ro ri.
BLOCK = [
    # 1. Love the Way You Lie & related
    "love the way you lie", "eminem", "marshall mathers", "rihanna", "skylar grey", "holly brook",
    "alex da kid", "alexander grant", "shady records", "chris brown", "kim scott", "rigo morales",
    "effigy studios", "makeba riddick", "marcos tovar", "mike strange", "joe strange",
    
    # 2. Quang Trung / Tay Son & related
    "quang trung", "nguyen hue", "tay son", "nguyen nhac", "nguyen lu", "nguyen anh",
    "truong van hien", "truong van hanh", "truong phuc loan", "ho thom", "bac binh vuong",
    "ngoc hoi", "dong da", "bui thi xuan", "tran quang dieu",
    
    # 3. Hong Bang / Hung Kings & related
    "hung vuong", "hung king", "hung kings", "hong bang", "van lang", "lac long quan",
    "au co", "kinh duong vuong", "kinh duong", "thuc phan", "an duong vuong", "son tinh",
    "thuy tinh", "tan vien", "phong chau", "loc tuc", "sung lam", "de minh", "de nghi",
    "vu tien", "than long", "dong dinh quan", "de lai", "mi nuong", "lac hau", "lac tuong",
    "thuong luong", "bach hac",
]

BLOCK_RE = re.compile(r"\b(" + "|".join(re.escape(b) for b in BLOCK) + r")\b")

CUT_RE = re.compile(
    r"\n\s*(={2,}\s*)?(References|External links|See also|Notes|Further reading|Sources|Bibliography|Citations|Footnotes)(\s*={2,})?\s*\n.*",
    re.S | re.IGNORECASE,
)


def fold(s: str) -> str:
    """Bo dau, lowercase, de so khop khong phan biet dau."""
    s = unicodedata.normalize("NFKD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", fold(title)).strip("_")


def clean(text: str) -> str:
    # Cat bo cac phan phu tro nhu References, External links,...
    text = CUT_RE.sub("", "\n" + text)
    # Don dep khoang trang thua
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def blocked_terms(text: str) -> list:
    return sorted(set(BLOCK_RE.findall(fold(text))))


def fetch(title: str):
    r = requests.get(
        API,
        headers=HEADERS,
        timeout=30,
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "titles": title,
        },
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return None, ""
    page = next(iter(pages.values()))
    if "missing" in page:
        return None, ""
    return page.get("title"), page.get("extract", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()

    print(f"[*] Bat dau crawl du lieu Wikipedia tieng Anh vao: {OUT.resolve()}")
    print(f"[*] Nguong toi thieu: {MIN_WORDS} tu. Bo loc chong ro ri: {len(BLOCK)} thuc the.\n")

    for group, titles in (("history", HISTORY), ("song", SONGS)):
        print(f"\n=== Nhóm: {group.upper()} ({len(titles)} ung vien) ===")
        for req in titles:
            try:
                title, raw = fetch(req)
            except Exception as e:
                rows.append([group, req, "", "", 0, f"error: {e}"])
                print(f"[ERROR: {e}] {req}")
                continue

            time.sleep(SLEEP)

            if title is None or not raw:
                rows.append([group, req, "", "", 0, "missing"])
                print(f"[{'missing':<40}] {req}")
                continue

            if title in seen:
                rows.append([group, req, title, "", 0, "duplicate"])
                print(f"[{'duplicate':<40}] {title}")
                continue
            seen.add(title)

            text = clean(raw)
            words = len(text.split())
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")

            hits = blocked_terms(text)
            if hits:
                status = "blocked: " + ", ".join(hits[:5])
                if len(hits) > 5:
                    status += f" (+{len(hits)-5} more)"
            elif words < MIN_WORDS:
                status = f"too short ({words})"
            else:
                status = "ok"
                out_path = OUT / f"{slugify(title)}.txt"
                out_path.write_text(text, encoding="utf-8")

            rows.append([group, req, title, url, words, status])
            print(f"[{status[:40]:<40}] {title} ({words} words)")

    manifest_path = OUT / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "requested", "title", "url", "words", "status"])
        w.writerows(rows)

    print("\n" + "=" * 60)
    print("TONG KET CRAWL")
    print("=" * 60)
    total_ok = 0
    total_words = 0
    for group in ("history", "song"):
        ok = [r for r in rows if r[0] == group and r[5] == "ok"]
        w_cnt = sum(r[4] for r in ok)
        total_ok += len(ok)
        total_words += w_cnt
        print(f"  - {group:<10}: {len(ok):>2} bai hop le, {w_cnt:>7} tu")
    print(f"  => TONG CONG: {total_ok} bai hop le, {total_words} tu")
    print(f"  - Manifest: {manifest_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
