#!/usr/bin/env python3
"""從影片抽出「畫面有變化」的關鍵幀，讓 AI 用看圖的方式看懂影片。

只需要 opencv-python 和 numpy（OpenCV 內建 ffmpeg 解碼，不必另外安裝 ffmpeg）。

輸出（都在 --out 目錄）：
  index.md          影片資訊、每張幀的時間與變化量、畫面靜止的區間
  overview_NN.jpg   縮圖總覽（每張最多 16 格），先看這個掌握全貌
  frames/NNN_時間.jpg  每張關鍵幀，上方黑條標示影片時間

用法：
  py extract_frames.py VIDEO --out DIR                    # 自動挑變化點
  py extract_frames.py VIDEO --out DIR --start 1:05 --end 1:20 --width 1600   # 放大某段
  py extract_frames.py VIDEO --out DIR --at 0:12,0:13.5   # 指定時間點
  py extract_frames.py VIDEO --out DIR --fps 2            # 固定每秒 N 張
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Windows 終端機預設不是 UTF-8，中文訊息會變亂碼
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("缺少套件：請執行  py -m pip install opencv-python numpy")

SMALL = (640, 360)       # 計算畫面變化用的尺寸（太小會把文字筆畫糊掉）
GRID = (64, 36)          # 把畫面切成格子，算「有幾格變了」，打幾個字也抓得到
PIXEL_DELTA = 25         # 灰階差超過這個值才算「這個像素變了」
MAX_CANDIDATES = 600     # 第一輪掃描最多看幾個時間點


def parse_time(text):
    """'75'、'1:15'、'0:01:15.5' -> 秒數"""
    text = str(text).strip()
    secs = 0.0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return secs


def fmt_time(t, sep=":"):
    t = max(t, 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f"{int(h)}{sep}{int(m):02d}{sep}{s:04.1f}"
    return f"{int(m):02d}{sep}{s:04.1f}"


def open_video(path):
    """打開影片；Windows 上 OpenCV 讀不了中文路徑時，改用暫存的英文路徑。"""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    cap.release()
    if str(path).isascii():
        return None, None
    tmpdir = tempfile.mkdtemp(prefix="vidframes_")
    alias = Path(tmpdir) / ("video" + path.suffix.lower())
    try:
        os.link(path, alias)          # 同一顆硬碟：硬連結，不佔空間
    except OSError:
        shutil.copyfile(path, alias)  # 不同硬碟：複製一份
    cap = cv2.VideoCapture(str(alias))
    if cap.isOpened():
        return cap, tmpdir
    cap.release()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return None, None


def small_gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, SMALL, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0)   # 壓掉影片壓縮造成的雜訊


def change_ratio(a, b):
    """兩張圖之間，有多少比例的格子裡出現明顯改變（0~1）。"""
    mask = (cv2.absdiff(a, b) > PIXEL_DELTA).astype(np.float32)
    cells = cv2.resize(mask, GRID, interpolation=cv2.INTER_AREA)
    return float(np.mean(cells > 0))


def read_at(cap, t, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def scan(cap, times, fps):
    """依序讀取每個時間點的畫面，回傳 [(t, 縮小灰階圖)]。間隔小就逐幀讀，間隔大就跳著讀。"""
    out = []
    if not times:
        return out
    step_frames = (times[1] - times[0]) * fps if len(times) > 1 else 1
    if step_frames > 20:
        for t in times:
            f = read_at(cap, t, fps)
            if f is not None:
                out.append((t, small_gray(f)))
        return out
    targets = [int(round(t * fps)) for t in times]
    idx = targets[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ti = 0
    while ti < len(targets):
        if not cap.grab():
            break
        if idx >= targets[ti]:
            ok, f = cap.retrieve()
            if ok:
                out.append((idx / fps, small_gray(f)))
            while ti < len(targets) and targets[ti] <= idx:
                ti += 1
        idx += 1
    return out


def select_changes(samples, threshold, max_frames):
    """跟「上一張留下來的幀」比，變化超過門檻才留下 → 同時做到場景偵測與去重複。"""
    kept = [(samples[0][0], 1.0, samples[0][1])]
    for t, img in samples[1:]:
        r = change_ratio(kept[-1][2], img)
        if r >= threshold:
            kept.append((t, r, img))
    # 最後的畫面常常是「結果」（例如錯誤停住的畫面），有變化就一定留
    last_t, last_img = samples[-1]
    if kept[-1][0] != last_t:
        r = change_ratio(kept[-1][2], last_img)
        if r >= threshold / 3:
            kept.append((last_t, r, last_img))
    dropped = 0
    if len(kept) > max_frames:
        dropped = len(kept) - max_frames
        first, last, middle = kept[0], kept[-1], kept[1:-1]
        middle = sorted(middle, key=lambda k: k[1], reverse=True)[: max_frames - 2]
        kept = [first] + sorted(middle, key=lambda k: k[0]) + [last]
    return [(t, r) for t, r, _ in kept], dropped


def label_frame(frame, t, width):
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    bar_h = max(28, frame.shape[1] // 40)
    bar = np.zeros((bar_h, frame.shape[1], 3), np.uint8)
    scale = bar_h / 36
    cv2.putText(bar, fmt_time(t), (8, int(bar_h * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)
    return np.vstack([bar, frame])


def write_jpg(path, img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        Path(path).write_bytes(buf.tobytes())   # 用 Python 寫檔，中文路徑也沒問題


def make_overviews(out_dir, labeled, cols=4, per_sheet=16, thumb_w=400):
    sheets = []
    for s in range(0, len(labeled), per_sheet):
        chunk = labeled[s : s + per_sheet]
        thumbs = []
        for img in chunk:
            h, w = img.shape[:2]
            thumbs.append(cv2.resize(img, (thumb_w, int(h * thumb_w / w)), interpolation=cv2.INTER_AREA))
        th = max(t.shape[0] for t in thumbs)
        rows = math.ceil(len(thumbs) / cols)
        sheet = np.full((rows * (th + 6) + 6, cols * (thumb_w + 6) + 6, 3), 40, np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            y, x = 6 + r * (th + 6), 6 + c * (thumb_w + 6)
            sheet[y : y + t.shape[0], x : x + thumb_w] = t
        name = f"overview_{len(sheets) + 1:02d}.jpg"
        write_jpg(out_dir / name, sheet, 85)
        sheets.append(name)
    return sheets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="輸出資料夾（預設：系統暫存區）")
    ap.add_argument("--start", default="0", help="開始時間，秒或 mm:ss")
    ap.add_argument("--end", help="結束時間，秒或 mm:ss（預設：影片結尾）")
    ap.add_argument("--max-frames", type=int, default=30, help="最多幾張關鍵幀（預設 30）")
    ap.add_argument("--threshold", type=float, default=0.005,
                    help="多少比例的格子改變才算新畫面（預設 0.005 = 0.5%%，約 10 格；調低可抓更細微的變化）")
    ap.add_argument("--width", type=int, default=1024, help="輸出幀的寬度（預設 1024；要看小字用 1600）")
    ap.add_argument("--at", help="只抽這些時間點，逗號分隔，例如 0:12,0:13.5")
    ap.add_argument("--fps", type=float, help="改成固定每秒抽 N 張（不做變化偵測）")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"找不到影片：{video}")
    cap, tmpdir = open_video(video)
    if cap is None:
        sys.exit(f"OpenCV 打不開這個影片（格式不支援或檔案損壞）：{video}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = nframes / fps if nframes > 0 else 0.0
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start = max(0.0, parse_time(args.start))
        end = min(parse_time(args.end), duration) if args.end else duration
        if end <= start:
            sys.exit(f"時間範圍不對：開始 {fmt_time(start)}、結束 {fmt_time(end)}，影片長度 {fmt_time(duration)}")
        last_frame_t = max(start, end - 1.0 / fps)

        out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix=f"frames_{video.stem[:20]}_"))
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in list(frames_dir.glob("*.jpg")) + list(out_dir.glob("overview_*.jpg")):
            old.unlink()

        dropped = 0
        precision = None
        if args.at:
            mode = "指定時間點"
            chosen = [(min(max(parse_time(x), 0.0), max(duration - 1.0 / fps, 0.0)), None)
                      for x in args.at.split(",") if x.strip()]
        elif args.fps:
            mode = f"固定每秒 {args.fps:g} 張"
            n = int((end - start) * args.fps) + 1
            times = [start + i / args.fps for i in range(n)]
            if len(times) > args.max_frames:
                dropped = len(times) - args.max_frames
                times = times[: args.max_frames]
            chosen = [(t, None) for t in times]
        else:
            mode = f"變化偵測（門檻 {args.threshold:.1%}）"
            step = max((end - start) / MAX_CANDIDATES, 1.0 / fps)
            precision = step
            times = [start + i * step for i in range(int((last_frame_t - start) / step) + 1)]
            if times[-1] < last_frame_t:
                times.append(last_frame_t)
            samples = scan(cap, times, fps)
            if not samples:
                sys.exit("讀不到任何畫面。")
            chosen, dropped = select_changes(samples, args.threshold, args.max_frames)

        rows, labeled = [], []
        for i, (t, r) in enumerate(chosen, 1):
            f = read_at(cap, t, fps)
            if f is None:
                continue
            img = label_frame(f, t, args.width)
            name = f"{i:03d}_{fmt_time(t, '-')}.jpg"
            write_jpg(frames_dir / name, img)
            labeled.append(img)
            rows.append((i, t, r, name))
        sheets = make_overviews(out_dir, labeled)

        # 兩張相鄰關鍵幀之間沒抓到變化 → 那段畫面基本靜止
        static = []
        if not args.at and not args.fps:
            for (_, t0, _, _), (_, t1, _, _) in zip(rows, rows[1:]):
                if t1 - t0 >= max(3.0, (end - start) / 20):
                    static.append((t0, t1))

        lines = [
            f"# 影片關鍵幀：{video.name}",
            "",
            f"- 檔案：`{video}`",
            f"- 長度：{fmt_time(duration)}　解析度：{vw}x{vh}　FPS：{fps:.2f}",
            f"- 抽取範圍：{fmt_time(start)} – {fmt_time(end)}　方式：{mode}",
            f"- 關鍵幀：{len(rows)} 張（寬 {min(args.width, vw)}px）",
        ]
        if precision:
            lines.append(f"- 時間精度：每 {precision:.2f} 秒檢查一次畫面，所以標示的時間是「最晚在這時已經變化」，"
                         f"實際變化發生在前 {precision:.2f} 秒內" + ("（已逐幀檢查）" if precision * fps <= 1.01 else ""))
        if dropped:
            lines.append(f"- ⚠ 還有 {dropped} 個變化點因為超過 --max-frames 被略過；"
                         "需要細節請用 --start/--end 縮小範圍重抽")
        lines += ["", "## 總覽圖（先看這些）", ""] + [f"- `{out_dir / s}`" for s in sheets]
        lines += ["", "## 關鍵幀", "", "| # | 時間 | 跟前一張相比的變化 | 檔案 |", "|---|---|---|---|"]
        for i, t, r, name in rows:
            change = "—" if r is None else ("起始" if i == 1 else f"{r:.1%}")
            lines.append(f"| {i} | {fmt_time(t)} | {change} | `frames/{name}` |")
        if static:
            lines += ["", "## 畫面幾乎沒變的區間", ""]
            lines += [f"- {fmt_time(a)} – {fmt_time(b)}" for a, b in static]
        (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"完成：{len(rows)} 張關鍵幀，{len(sheets)} 張總覽圖")
        print(f"索引：{out_dir / 'index.md'}")
        for s in sheets:
            print(f"總覽：{out_dir / s}")
        if dropped:
            print(f"注意：有 {dropped} 個變化點被略過，可用 --start/--end 縮小範圍")
    finally:
        cap.release()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
