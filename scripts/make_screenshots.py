"""Start the Streamlit app, capture screenshots, and annotate them with arrows.

    python scripts/make_screenshots.py     (writes docs/img/*.png)
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8765/"

SCENARIOS = {
    "ui_public": "?role=public_user&q=inclusion+and+exclusion+criteria",
    "ui_admin_filtered": "?role=admin&phase=Phase+II&area=Oncology&kind=table&q=dose+schedule+table",
}


def wait_up():
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("streamlit did not start")


def shoot():
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port", "8765",
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_up()
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            for name, qs in SCENARIOS.items():
                pg.goto(URL + qs)
                pg.wait_for_selector("text=Chunks indexed", timeout=120000)
                time.sleep(2)
                pg.screenshot(path=str(IMG / f"{name}_raw.png"))
            b.close()
    finally:
        proc.terminate()


def arrow(d, start, end, text, color=(220, 30, 30)):
    import math
    font = ImageFont.truetype("arial.ttf", 20)
    d.line([start, end], fill=color, width=4)
    a = math.atan2(end[1] - start[1], end[0] - start[0])
    for s in (-0.5, 0.5):
        d.line([end, (end[0] - 18 * math.cos(a + s), end[1] - 18 * math.sin(a + s))], fill=color, width=4)
    w = d.textlength(text, font=font)
    d.rectangle([start[0] - 6, start[1] - 28, start[0] + w + 6, start[1] + 2], fill=(255, 255, 220), outline=color)
    d.text((start[0], start[1] - 26), text, fill=color, font=font)


def annotate(name, notes, lm=330, rm=380):
    im = Image.open(IMG / f"{name}_raw.png").convert("RGB")
    canvas = Image.new("RGB", (im.width + lm + rm, im.height), (250, 250, 250))
    canvas.paste(im, (lm, 0))
    d = ImageDraw.Draw(canvas)
    for start, end, text in notes:
        # start.x: 'L' -> left margin, 'R' -> right margin; end is in screenshot coords.
        side, y = start
        sx = 10 if side == "L" else lm + im.width + 10
        arrow(d, (sx, y), (end[0] + lm, end[1]), text)
    canvas.save(IMG / f"{name}.png")


if __name__ == "__main__":
    shoot()
    common = [
        (("L", 120), (28, 180), "1 Role = access control"),
        (("L", 300), (28, 302), "2 Metadata filters"),
        (("L", 560), (28, 553), "3 Content type"),
        (("R", 230), (1322, 270), "4 Free-text question"),
        (("R", 330), (1100, 355), "5 Filter-effect counters"),
        (("R", 430), (870, 490), "6 Metadata badges"),
        (("R", 600), (1306, 600), "7 Retrieved chunk"),
    ]
    annotate("ui_public", common)
    annotate("ui_admin_filtered", common)
    print("done")
