import json
from pathlib import Path

cache_dir = Path("HiRAG/hirag_cache_v2")
if not cache_dir.exists():
    print("Cache dir does not exist yet.")
    exit(0)

files = list(cache_dir.glob("*"))
print(f"Files in {cache_dir}: {len(files)}")
for f in files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:35s}: {size_kb:.2f} KB")

# Check entities vdb if exists
vdb_ent = cache_dir / "vdb_entities.json"
if vdb_ent.exists() and vdb_ent.stat().st_size > 0:
    try:
        data = json.loads(vdb_ent.read_text(encoding="utf-8"))
        print(f"Entities in VDB: {len(data.get('data', []))}")
    except Exception:
        pass
