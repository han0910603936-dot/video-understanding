"""產生一段假的螢幕錄影：填表單 -> 按送出 -> 轉圈 -> 0:17 跳出錯誤 -> 0:25 關掉。"""
import sys, cv2, numpy as np
from pathlib import Path

out = Path(sys.argv[1])
W, H, FPS, DUR = 1280, 720, 15, 30
tmp = out.with_name("tmp_test.mp4")
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
F = cv2.FONT_HERSHEY_SIMPLEX
user, pw = "jeff_student", "********"
for i in range(DUR * FPS):
    t = i / FPS
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 50), (90, 60, 30), -1)
    cv2.putText(img, "Grade Portal - Login", (20, 35), F, 1, (255, 255, 255), 2)
    cv2.putText(img, "Username", (400, 200), F, 0.8, (40, 40, 40), 2)
    cv2.rectangle(img, (400, 215), (880, 265), (120, 120, 120), 2)
    cv2.putText(img, "Password", (400, 320), F, 0.8, (40, 40, 40), 2)
    cv2.rectangle(img, (400, 335), (880, 385), (120, 120, 120), 2)
    u = user[: max(0, int((t - 2) * 3))] if t > 2 else ""
    p = pw[: max(0, int((t - 7) * 3))] if t > 7 else ""
    cv2.putText(img, u, (410, 250), F, 0.9, (0, 0, 0), 2)
    cv2.putText(img, p, (410, 370), F, 0.9, (0, 0, 0), 2)
    pressed = 11.5 <= t < 12
    cv2.rectangle(img, (560, 430), (720, 480), (60, 140, 60) if not pressed else (30, 90, 30), -1)
    cv2.putText(img, "Submit", (590, 465), F, 0.9, (255, 255, 255), 2)
    # 滑鼠游標
    cx, cy = (640, 455) if t > 10 else (300 + int(t * 30), 600)
    cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1)
    if 12 <= t < 17:
        ang = int(t * 360) % 360
        cv2.ellipse(img, (640, 560), (25, 25), ang, 0, 270, (200, 120, 0), 4)
        cv2.putText(img, "Connecting...", (560, 620), F, 0.7, (80, 80, 80), 2)
    if 17 <= t < 25:
        ov = img.copy(); ov[:] = (60, 60, 60); img = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
        cv2.rectangle(img, (340, 230), (940, 470), (255, 255, 255), -1)
        cv2.rectangle(img, (340, 230), (940, 280), (40, 40, 200), -1)
        cv2.putText(img, "Error", (360, 265), F, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "502 Bad Gateway: server.ptivs.local", (360, 330), F, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "did not respond within 5000 ms", (360, 365), F, 0.7, (0, 0, 0), 2)
        cv2.rectangle(img, (800, 410), (920, 455), (200, 200, 200), -1)
        cv2.putText(img, "OK", (840, 442), F, 0.8, (0, 0, 0), 2)
    if t >= 25:
        cv2.putText(img, "Login failed. Please try again later.", (400, 540), F, 0.8, (30, 30, 200), 2)
    vw.write(img)
vw.release()
tmp.replace(out)
print("ok", out)
