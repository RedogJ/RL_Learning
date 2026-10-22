"""下载固定版本的官方 Go2 MJCF 及其引用网格，已有校验通过的文件不重复下载。"""

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

from .config import MODEL_ROOT

REVISION = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
BASE_URL = f"https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/{REVISION}"


def fetch_model():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = MODEL_ROOT / "download_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    known = {f["path"]: f for f in old.get("files", [])} if old.get("revision") == REVISION else {}
    files = []

    def fetch(relative, source):
        destination = MODEL_ROOT / relative
        entry = known.get(relative)
        if (entry and destination.is_file()
                and hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]):
            files.append(entry)
            return destination
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"{BASE_URL}/{source}", timeout=40) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entry = {"path": relative, "source": f"{BASE_URL}/{source}", "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
        files.append(entry)
        manifest_path.write_text(json.dumps({"repository": "unitreerobotics/unitree_mujoco",
                                            "revision": REVISION, "files": files}, indent=2) + "\n")
        print(f"模型资源：{relative} ({len(data) / 1024:.0f} KiB)", flush=True)
        return destination

    model = fetch("go2.xml", "unitree_robots/go2/go2.xml")
    for mesh in ET.parse(model).getroot().findall("./asset/mesh"):
        name = mesh.attrib["file"]
        if Path(name).name != name:
            raise ValueError(f"非预期网格路径：{name}")
        fetch(f"assets/{name}", f"unitree_robots/go2/assets/{name}")
    fetch("LICENSE", "LICENSE")
    manifest_path.write_text(json.dumps({"repository": "unitreerobotics/unitree_mujoco",
                                        "revision": REVISION, "files": files}, indent=2) + "\n")
    print(f"模型就绪：{MODEL_ROOT}；总计 {sum(f['bytes'] for f in files) / 1024**2:.1f} MiB")


if __name__ == "__main__":
    fetch_model()
