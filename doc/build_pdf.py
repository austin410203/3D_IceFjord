"""Build 冰峽灣_渲染樣張與Prompt手冊.pdf from shots/*.jpg + techniques.json.

    python doc/build_pdf.py

Needs Pillow and Google Chrome (headless print-to-pdf). Captures come from the page's capture API
(window.fjord.captureBegin / batch) posted to tools/serve.py.
"""
import html
import json
import os
import subprocess
from datetime import date

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "shots")
IMG = os.path.join(ROOT, "doc", "img")
OUT_HTML = os.path.join(ROOT, "doc", "handbook.html")
OUT_PDF = os.path.join(ROOT, "冰峽灣_渲染樣張與Prompt手冊.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SITE = "https://3d-icefjord.austin410203.workers.dev"

D = json.load(open(os.path.join(ROOT, "techniques.json"), encoding="utf-8"))
T = D["techniques"]
VIEW_NAME = {v["id"]: v["name"] for v in D["views"]}
e = html.escape

os.makedirs(IMG, exist_ok=True)
for f in os.listdir(SHOTS):
    if f.endswith(".jpg"):
        im = Image.open(os.path.join(SHOTS, f)).convert("RGB")
        im.thumbnail((1280, 720), Image.LANCZOS)
        im.save(os.path.join(IMG, f), quality=82, optimize=True)


def img(name, cls=""):
    return f'<img class="{cls}" src="img/{name}.jpg">'


def prompt_box(text, label="下次這樣提示我"):
    return f'<div class="pbox"><div class="plabel">{label}</div><p>{e(text)}</p></div>'


pages = []

# ── cover
pages.append(f"""
<section class="page cover">
  {img('full_village', 'hero')}
  <div class="cover-text">
    <div class="kicker">RENDERING TECHNIQUES · PROMPT HANDBOOK</div>
    <h1>冰峽灣・渲染樣張與 Prompt 手冊</h1>
    <p>研究 Minamo（水面）和 Pond Line（金魚電車）兩個開源作品的渲染做法，在同一個冰峽灣場景裡逐項實作。
    每一頁都附上開／關對照圖，以及「下次要這樣提示 Claude」的 prompt。</p>
    <div class="meta">three.js r186 · WebGPURenderer + TSL · {date.today():%Y-%m-%d} · 互動版：{SITE}</div>
  </div>
</section>""")

# ── how to prompt
master = """目標：用 three.js r186（WebGPURenderer + TSL）做【場景描述】，單位一律用公尺。
參考圖：【附 1–3 張】。動手前先拆解：構圖和鏡頭高度、時段和太陽方向、主色票、材質清單、大氣、會動的元素。
風格：【物理寫實（Minamo 式）／藝術調色（Pond Line 式）】。
技巧（依序分階段做）：
  1. 物理天空＋環境光　2. 色調映射 AgX／Neutral　3. 柔和陰影　4. 程序化材質＋濕潤帶
  5. 物理水面　6. 透光材質　7. 倒角　8. 冷暖對比的燈光　9. 空氣透視
  10. 微動態　11. Bloom 光暈　12. 調色　13. 鏡頭質感
流程：每個階段都截圖，和參考圖比對後列出 3 個最大差異並修正，再進下一個階段。
驗收：做一個面板，可以逐項開關每個技巧、切換預設視角，並列出每一項對應的 prompt。
效能：桌機 60 fps；所有開關用 uniform 控制，切換時不重新編譯 shader。"""
rules = [
    ("給參考圖，並要求先拆解", "附 1–3 張參考圖，請我先寫出構圖、時段、色票、材質、動態五份清單，你確認後再動手。"),
    ("直接點名技巧，附上參數", "「水面要好看」太模糊；改成「reflector 平面反射＋Fresnel F0 0.02＋依像素足跡淡化波浪」，結果就會穩定。"),
    ("先定好色彩管線", "一開始就指定色調映射（AgX 或 Neutral）和曝光，後面調出來的顏色才不會整片死白、整個重調。"),
    ("分階段做，每階段截圖比對", "要求每一步都截圖，列出 3 個和參考圖最大的差異再修正。這是品質差距最大的地方。"),
    ("說清楚風格方向", "物理寫實（Minamo）或藝術調色（Pond Line），兩者的 tone mapping、Bloom、調色參數完全不同。"),
    ("要求可以切換比較", "請我做開關面板，每個技巧都能單獨關掉，一眼就看得出哪一步有效、哪一步多餘。"),
]
pages.append(f"""
<section class="page">
  <header><span class="num">PART 1</span><h2>下次怎麼提示 Claude</h2></header>
  <div class="two">
    <div class="rules">{''.join(f'<div class="rule"><b>{e(a)}</b><p>{e(b)}</p></div>' for a, b in rules)}</div>
    <div>{prompt_box(master, '萬用 PROMPT 範本（把【】換成你的內容）')}
      <p class="note">模式建議：Opus 5.5，effort 設 high（最後細修用 xhigh）；先用 Plan mode 拆解參考圖。
      一次完整場景很吃額度，所以一次只做一個階段。</p></div>
  </div>
</section>""")

# ── ladder
ladder = [("00", "原始", "AI 第一版常見的樣子")] + [(f"{i + 1:02d}", t["name"], "＋ 加上這一項") for i, t in enumerate(T)]
for chunk_i in range(0, len(ladder), 9):
    chunk = ladder[chunk_i:chunk_i + 9]
    cells = "".join(f'<figure>{img("ladder_" + n)}<figcaption><b>{n}</b> {e(name)}</figcaption></figure>' for n, name, _ in chunk)
    pages.append(f"""
<section class="page">
  <header><span class="num">PART 2</span><h2>從原始到完成：一步一步疊加 {'（續）' if chunk_i else ''}</h2>
  <span class="sub">同一個漁村視角，每一格都多開一項技巧。順序就是建議的 prompt 分階段順序。</span></header>
  <div class="grid3">{cells}</div>
</section>""")

# ── one page per technique
for i, t in enumerate(T):
    pages.append(f"""
<section class="page">
  <header><span class="num">PART 3 · 技巧 {i + 1:02d}</span><h2>{e(t['name'])}</h2><span class="sub">視角：{VIEW_NAME[t['view']]}</span></header>
  <div class="pair">
    <figure>{img('off_' + t['id'])}<figcaption class="off">關閉（其他技巧全開）</figcaption></figure>
    <figure>{img('full_' + t['view'])}<figcaption class="on">開啟</figcaption></figure>
  </div>
  <div class="two narrow">
    <div><div class="src">出處：{e(t['source'])}</div><p class="what">{e(t['what'])}</p></div>
    {prompt_box(t['prompt'])}
  </div>
</section>""")

# ── tone mapping comparison
tones = [("none", "無（直接截斷）", "太陽和水面閃光整片死白，暖色會偏成檸檬黃。"),
         ("agx", "AgX（Minamo）", "寫實、亮部柔和，飽和度稍低。"),
         ("neutral", "Neutral（Pond Line）", "保留原本的顏色，亮部帶粉橘。"),
         ("aces", "ACES Filmic", "對比強、偏電影感，暗部比較重。")]
pages.append(f"""
<section class="page">
  <header><span class="num">PART 4</span><h2>色調映射比較</h2><span class="sub">同一個 HDR 畫面，只換 tone mapping（調色關閉）</span></header>
  <div class="grid2">{''.join(f'<figure>{img("tone_" + k)}<figcaption><b>{e(n)}</b>　{e(d)}</figcaption></figure>' for k, n, d in tones)}</div>
</section>""")

# ── presets
P = D["presets"]
for k in range(0, len(P), 2):
    blocks = ""
    for p in P[k:k + 2]:
        blocks += f"""<div class="preset">
      <div class="prow">{img(f"preset_{p['id']}_village")}{img(f"preset_{p['id']}_fjord")}</div>
      <div class="two narrow"><div><b class="pname">{e(p['name'])}</b><p class="what">{e(p['desc'])}</p></div>{prompt_box(p['prompt'], '風格 PROMPT')}</div>
    </div>"""
    pages.append(f"""
<section class="page">
  <header><span class="num">PART 5</span><h2>四種預設風格 {'（續）' if k else ''}</h2><span class="sub">左：漁村近景　右：峽灣全景</span></header>
  {blocks}
</section>""")

# ── full prompt appendix
full_prompt = "\n".join(
    ["目標：用 three.js r186（WebGPURenderer + TSL）做一個藍調時刻的冰峽灣場景：雪山、冰海、漂流浮冰、亮著暖色窗燈的漁村小屋。單位一律用公尺。",
     "請依下面的技巧分階段實作，每個階段都截圖，和參考圖比對後列出 3 個最大差異並修正："]
    + [f"{i + 1}. 【{t['name']}】{t['prompt']}" for i, t in enumerate(T)]
    + ["最後做一個面板：每個技巧都能開關、四個預設視角、四個風格預設，並顯示每一項對應的 prompt。"])
pages.append(f"""
<section class="page">
  <header><span class="num">PART 6</span><h2>附錄：做出這個場景的完整 prompt</h2></header>
  {prompt_box(full_prompt, '可以直接複製使用')}
  <p class="note">互動版的面板裡也有「依目前勾選產生 prompt」按鈕，只勾想要的技巧，就會組出對應的 prompt。</p>
</section>""")

CSS = """
@page { size: A4 landscape; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; color: #1d2733; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.page { width: 297mm; height: 210mm; padding: 12mm 14mm; page-break-after: always; overflow: hidden; position: relative; background: #f6f8fa; }
header { display: flex; align-items: baseline; gap: 10px; border-bottom: 2px solid #1d2733; padding-bottom: 4mm; margin-bottom: 5mm; flex-wrap: wrap; }
header .num { font: 700 13pt/1 Georgia, serif; color: #e07a32; }
header h2 { margin: 0; font-size: 18pt; }
header .sub { color: #5d6b78; font-size: 9.5pt; margin-left: auto; }
figure { margin: 0; }
figure img { width: 100%; display: block; border-radius: 2mm; }
figcaption { font-size: 9pt; color: #3c4855; margin-top: 1.5mm; }
figcaption b { color: #e07a32; margin-right: 4px; }
figcaption.off { color: #8a96a3; } figcaption.on { color: #e07a32; font-weight: 700; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm 5mm; }
.grid3 img { height: 42mm; object-fit: cover; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm 6mm; }
.grid2 img { height: 67mm; object-fit: cover; }
.two { display: grid; grid-template-columns: 1fr 1.25fr; gap: 8mm; }
.two.narrow { margin-top: 5mm; gap: 7mm; grid-template-columns: 1fr 1.1fr; }
.src { color: #2a7fb0; font-size: 9pt; margin-bottom: 2mm; }
.what { font-size: 10pt; line-height: 1.65; margin: 0; }
.pbox { background: #fff4ea; border-left: 3px solid #e07a32; border-radius: 2mm; padding: 3mm 4mm; }
.pbox .plabel { font-size: 8pt; letter-spacing: .15em; color: #e07a32; font-weight: 700; margin-bottom: 1.5mm; }
.pbox p { margin: 0; font-size: 9.5pt; line-height: 1.6; white-space: pre-wrap; }
.rules { display: grid; gap: 3mm; }
.rule b { font-size: 11pt; } .rule p { margin: 0.5mm 0 0; font-size: 9.5pt; line-height: 1.55; color: #3c4855; }
.note { font-size: 9pt; color: #5d6b78; line-height: 1.6; margin-top: 4mm; }
.preset { margin-bottom: 5mm; }
.prow { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.prow img { width: 100%; height: 42mm; object-fit: cover; border-radius: 2mm; }
.preset .two.narrow { margin-top: 2.5mm; }
.pname { font-size: 11.5pt; }
.cover { padding: 0; background: #0d141c; color: #eef3f7; }
.cover .hero { width: 100%; height: 100%; object-fit: cover; position: absolute; inset: 0; }
.cover-text { position: absolute; left: 0; right: 0; bottom: 0; padding: 14mm 16mm 12mm; background: linear-gradient(to top, rgba(8,12,18,.92), rgba(8,12,18,.6) 70%, transparent); }
.cover .kicker { font-size: 8.5pt; letter-spacing: .3em; color: #ffb070; }
.cover h1 { font-size: 28pt; margin: 2mm 0 3mm; }
.cover p { font-size: 11pt; line-height: 1.7; max-width: 200mm; margin: 0 0 4mm; color: #d7e2ea; }
.cover .meta { font-size: 8.5pt; color: #9fb2c2; }
"""
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(f'<!doctype html><html lang="zh-Hant-TW"><head><meta charset="utf-8"><title>冰峽灣・渲染樣張與 Prompt 手冊</title><style>{CSS}</style></head><body>{"".join(pages)}</body></html>')

subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={OUT_PDF}", "file:///" + OUT_HTML.replace(os.sep, "/")], check=True, timeout=180)
print(OUT_PDF, os.path.getsize(OUT_PDF) // 1024, "KB,", len(pages), "pages")
