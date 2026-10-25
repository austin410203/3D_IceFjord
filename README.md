# 冰峽灣・渲染技巧實驗室

藍調時刻的冰峽灣（雪山、冰海、漂流浮冰、亮著暖色窗燈的漁村），把 [Minamo](https://github.com/kanaworksai-hue/minamo) 和 [Pond Line](https://pondline.scottsun.io) 的渲染做法拆成 13 個可以逐項開關的技巧，每一項都附上「下次怎麼提示 Claude」的 prompt。

- 互動版：https://3d-icefjord.austin410203.workers.dev
- PDF 手冊：`冰峽灣_渲染樣張與Prompt手冊.pdf`（開／關對照圖＋prompt）
- 研究筆記：`doc/3D渲染技巧清單_Minamo_Pondline.md`
- **潮聲島版**（`tidesong/`）：把 Unity 遊戲《潮聲島》的漁村、碼頭、潟湖和瀑布用網頁重建，套用 15 項技巧（新增：依水深吸收的潟湖、海床焦散、岸邊泡沫、植被密度＋風、遊戲原本的手繪描邊），有 3 個時段和 5 種風格預設。互動版：https://3d-icefjord.austin410203.workers.dev/tidesong/ ；和 Unity 原版的對照圖在 `tidesong/compare/`。

three.js r186（WebGPURenderer + TSL），不需要建置，用 import map 從 CDN 載入。所有開關都是 uniform 或 visible，切換時不會重新編譯 shader。

## 本機執行

```
python tools/serve.py 5261
```

打開 http://localhost:5261/ （需要支援 WebGPU 的 Chrome 或 Edge）。

## 重新產生 PDF

1. 在頁面的 console 執行 `fjord.captureBegin()`，再用 `fjord.batch([...])` 拍圖，圖會經由 `tools/serve.py` 存到 `shots/`。
2. 執行 `python doc/build_pdf.py`（需要 Pillow 和 Google Chrome）。
