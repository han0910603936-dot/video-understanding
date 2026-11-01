# video-understanding

讓 Claude 看懂本機影片畫面的 skill。適合用來回答影片細節問題、分析螢幕錄影中的操作步驟和錯誤畫面。

它會把影片裡「畫面有變化的瞬間」抽成截圖，每張標上影片時間，再讓 Claude 逐張看圖。打字、跳出視窗、換頁都會被抓到，游標移動則不會，所以不會產生一堆重複的圖。只看畫面，不處理聲音。

## 安裝

需要 Python 3 和 OpenCV（不需要 ffmpeg）：

```bash
pip install opencv-python numpy
```

把這個 repo 放到 Claude 的 skills 資料夾：

```bash
git clone https://github.com/han0910603936-dot/video-understanding.git ~/.claude/skills/video-understanding
```

裝好之後，直接給 Claude 一個影片檔並提問就好，例如：「幫我看這段錄影，我在第幾秒出錯？」

## 單獨使用腳本

```bash
py scripts/extract_frames.py "影片.mp4" --out 輸出資料夾
py scripts/extract_frames.py "影片.mp4" --out 輸出資料夾 --start 1:05 --end 1:20 --width 1600
py scripts/extract_frames.py "影片.mp4" --out 輸出資料夾 --at 0:12,0:13.5
```

輸出包含 `index.md`（每張幀的時間與變化量）、`overview_*.jpg`（縮圖總覽）和 `frames/`（標好時間的大圖）。所有參數請看 [SKILL.md](SKILL.md)。

## 測試

`tests/make_test_video.py` 會產生一段 30 秒的假登入錄影（0:17 跳出錯誤視窗），可以拿來試跑：

```bash
py tests/make_test_video.py test.mp4
py scripts/extract_frames.py test.mp4 --out out
```
