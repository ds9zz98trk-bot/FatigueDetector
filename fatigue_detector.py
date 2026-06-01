#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════╗
║     FATIGUE DETECTOR — Детектор усталости  v2   ║
╚══════════════════════════════════════════════════╝
Установка:
    pip install opencv-python mediapipe pygame numpy pillow

Запуск:
    python fatigue_detector.py

Клавиши:  Q / ESC — выход    S — скриншот
"""

import sys, os, time, threading, wave, struct, urllib.request
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

# ─── Пороги ────────────────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.20   # EAR ниже → глаза закрыты
MAR_THRESHOLD   = 0.55   # MAR выше → рот открыт
EYE_CLOSED_SEC  = 4.0    # секунд закрытых глаз → тревога
MOUTH_OPEN_SEC  = 5.0    # секунд открытого рта → тревога

# ─── Индексы MediaPipe FaceMesh ────────────────────────────────────────────
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
MOUTH_TOP   = [82,  13,  312]
MOUTH_BOT   = [87,  14,  317]
MOUTH_LEFT  = 78
MOUTH_RIGHT = 308
MOUTH_OUTER = [61,185,40,39,37,0,267,269,270,409,
               291,375,321,405,314,17,84,181,91,146]

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

# ─── Шрифты (PIL) ──────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

def _find_font():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

_FONT_PATH = _find_font()

def _get_font(size: int):
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─── Функция рисования текста через PIL (поддержка кириллицы) ──────────────
def put_text(frame, text, pos, size=18, color=(255,255,255), bold=False):
    """Нарисовать текст с кириллицей/Unicode на OpenCV-кадре через PIL."""
    font = _get_font(size)
    # Конвертируем BGR → RGB
    img_pil = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = np.array(img_pil)[:, :, ::-1]


def put_text_shadow(frame, text, pos, size=18, color=(255,255,255)):
    """Текст с тенью для читаемости."""
    font = _get_font(size)
    img_pil = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(img_pil)
    x, y = pos
    # Тень
    draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0))
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    # Основной текст
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = np.array(img_pil)[:, :, ::-1]


# ═══════════════════════════════════════════════════════════════════════════
#  Звуковой модуль
# ═══════════════════════════════════════════════════════════════════════════

_sound_lock   = threading.Lock()
_last_beep    = {"eye": 0.0, "mouth": 0.0}
BEEP_COOLDOWN = 3.5

import tempfile as _tempfile
_TMP       = _tempfile.gettempdir()
_EYE_WAV   = os.path.join(_TMP, "_fatigue_eye.wav")
_MOUTH_WAV = os.path.join(_TMP, "_fatigue_mouth.wav")
_pygame_ok = False


def _write_wav(path: str, freq: float, duration: float,
               freq2: float = 0, vol: float = 0.82):
    """Сгенерировать WAV-файл с двухтоновым сигналом тревоги."""
    sr   = 44100
    n    = int(sr * duration)
    t    = np.linspace(0, duration, n, endpoint=False)
    w    = np.sin(2 * np.pi * freq * t) * 0.65
    if freq2:
        w += np.sin(2 * np.pi * freq2 * t) * 0.35
    # Огибающая ADSR-style
    fade_in  = int(sr * 0.03)
    fade_out = int(sr * 0.10)
    env = np.ones(n)
    env[:fade_in]   = np.linspace(0, 1, fade_in)
    env[-fade_out:] = np.linspace(1, 0, fade_out)
    w = np.clip(w * env * vol, -1.0, 1.0)
    samples = (w * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def init_sound():
    global _pygame_ok
    _write_wav(_EYE_WAV,   880, 0.8, freq2=1320)   # высокий тревожный
    _write_wav(_MOUTH_WAV, 660, 1.0, freq2=880)     # более низкий
    try:
        import pygame
        os.environ.setdefault("SDL_AUDIODRIVER", "")   # авто
        pygame.mixer.pre_init(44100, -16, 1, 1024)
        pygame.mixer.init()
        _pygame_ok = True
        print(f"  Звук: pygame {pygame.ver} | "
              f"инициализован {pygame.mixer.get_init()}")
    except Exception as e:
        print(f"  Звук: pygame недоступен ({e})")
        print("  Используется системный fallback")


def _play_wav_thread(path: str):
    """Воспроизвести WAV в отдельном потоке."""
    if _pygame_ok:
        try:
            import pygame
            snd = pygame.mixer.Sound(path)
            snd.play()
            time.sleep(snd.get_length() + 0.1)
            return
        except Exception:
            pass
    # Fallback — платформо-зависимый
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            os.system(f"afplay '{path}' &")
        else:
            # Linux: aplay / paplay / ffplay
            for player in ("aplay", "paplay", "ffplay -nodisp -autoexit"):
                if os.system(f"which {player.split()[0]} > /dev/null 2>&1") == 0:
                    os.system(f"{player} '{path}' > /dev/null 2>&1 &")
                    break
            else:
                os.system("echo -e '\\a'")
    except Exception:
        pass


def play_alert(kind: str):
    now = time.time()
    with _sound_lock:
        if now - _last_beep[kind] < BEEP_COOLDOWN:
            return
        _last_beep[kind] = now
    path = _EYE_WAV if kind == "eye" else _MOUTH_WAV
    threading.Thread(target=_play_wav_thread, args=(path,), daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════
#  Загрузка модели
# ═══════════════════════════════════════════════════════════════════════════

def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return True
    print("  Загрузка модели FaceLandmarker (~30 МБ)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if os.path.getsize(MODEL_PATH) > 100_000:
            print("  Модель загружена.")
            return True
    except Exception as e:
        print(f"  Ошибка загрузки: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  EAR / MAR
# ═══════════════════════════════════════════════════════════════════════════

def dist(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def ear(pts, idx):
    p = [pts[i] for i in idx]
    return (dist(p[1],p[5]) + dist(p[2],p[4])) / (2.0 * dist(p[0],p[3]) + 1e-6)

def mar(pts):
    top = np.mean([pts[i] for i in MOUTH_TOP], axis=0)
    bot = np.mean([pts[i] for i in MOUTH_BOT], axis=0)
    return dist(top, bot) / (dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  Детектор
# ═══════════════════════════════════════════════════════════════════════════

class FatigueDetector:
    def __init__(self, model_path: str):
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)

        self.eye_since   = None
        self.mouth_since = None
        self.eye_alerted = False
        self.mouth_alerted = False
        self._eye_prev   = False
        self._mouth_prev = False
        self.blinks = 0
        self.yawns  = 0
        self.start  = time.time()

    def process(self, rgb: np.ndarray, ts_ms: int) -> dict:
        import mediapipe as mp
        img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.lm.detect_for_video(img, ts_ms)
        now    = time.time()

        out = dict(face=False, ear_l=0, ear_r=0, mar=0,
                   eyes_closed=False, mouth_open=False,
                   eye_dur=0, mouth_dur=0,
                   eye_alert=False, mouth_alert=False, pts=[])

        if not result.face_landmarks:
            self.eye_since = self.mouth_since = None
            return out

        h, w  = rgb.shape[:2]
        lm    = result.face_landmarks[0]
        pts   = [(lm[i].x*w, lm[i].y*h) for i in range(len(lm))]
        out.update(face=True, pts=pts)

        # EAR
        el = ear(pts, LEFT_EYE)
        er = ear(pts, RIGHT_EYE)
        avg_ear = (el + er) / 2
        closed  = avg_ear < EAR_THRESHOLD
        out.update(ear_l=el, ear_r=er, eyes_closed=closed)
        if self._eye_prev and not closed:
            self.blinks += 1
        self._eye_prev = closed

        if closed:
            if self.eye_since is None:
                self.eye_since = now
            dur = now - self.eye_since
            out["eye_dur"] = dur
            if dur >= EYE_CLOSED_SEC:
                out["eye_alert"] = True
                if not self.eye_alerted:
                    self.eye_alerted = True
                    play_alert("eye")
        else:
            self.eye_since   = None
            self.eye_alerted = False

        # MAR
        mv     = mar(pts)
        opened = mv > MAR_THRESHOLD
        out.update(mar=mv, mouth_open=opened)
        if self._mouth_prev and not opened:
            self.yawns += 1
        self._mouth_prev = opened

        if opened:
            if self.mouth_since is None:
                self.mouth_since = now
            dur = now - self.mouth_since
            out["mouth_dur"] = dur
            if dur >= MOUTH_OPEN_SEC:
                out["mouth_alert"] = True
                if not self.mouth_alerted:
                    self.mouth_alerted = True
                    play_alert("mouth")
        else:
            self.mouth_since   = None
            self.mouth_alerted = False

        return out


# ═══════════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════════

C_GREEN  = (80, 220, 100)
C_YELLOW = (30, 200, 255)
C_RED    = (70,  70, 250)
C_WHITE  = (240, 240, 240)
C_GRAY   = (130, 130, 130)
C_DARK   = (18,  18,  18)
C_CYAN   = (220, 210,  40)
C_ORANGE = (40,  150, 255)


def _bar_h(frame, x, y, w, h, val, vmax, color, bg=(50,50,50)):
    ratio  = min(max(val, 0) / max(vmax, 1e-6), 1.0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80,80,80), 1)
    fw = int(w * ratio)
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x+fw, y+h), color, -1)


def _poly(frame, pts_all, indices, color, thickness=1):
    pts_sub = [pts_all[i] for i in indices if i < len(pts_all)]
    if len(pts_sub) < 2:
        return
    poly = np.array([(int(p[0]), int(p[1])) for p in pts_sub], np.int32)
    cv2.polylines(frame, [poly], True, color, thickness, cv2.LINE_AA)


def render_hud(frame, st, det, elapsed):
    H, W = frame.shape[:2]
    PW = 210   # ширина правой панели
    PX = W - PW - 6

    # ── Фоновая панель (правая) ───────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (PX-2, 50), (W-2, H-2), (12,12,12), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # ── Верхняя шапка ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, 46), (10,10,10), -1)
    put_text_shadow(frame, "FATIGUE DETECTOR", (12, 10),
                    size=22, color=C_GREEN)
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    put_text(frame, f"{mins:02d}:{secs:02d}", (W-72, 13),
             size=20, color=C_WHITE)

    if not st["face"]:
        msg = "Лицо не обнаружено"
        put_text_shadow(frame, msg, (W//2 - 130, H//2 - 15),
                        size=24, color=C_YELLOW)
        return

    # ── Ориентиры ──────────────────────────────────────────────────────────
    pts = st["pts"]
    if pts:
        _poly(frame, pts, LEFT_EYE,    C_CYAN)
        _poly(frame, pts, RIGHT_EYE,   C_CYAN)
        _poly(frame, pts, MOUTH_OUTER, C_CYAN)

    # ── Правая панель: EAR ────────────────────────────────────────────────
    py = 60
    eye_col = C_RED if st["eye_alert"] else (C_YELLOW if st["eyes_closed"] else C_GREEN)

    put_text(frame, "ГЛАЗА", (PX+4, py), size=14, color=C_GRAY)
    py += 18
    avg_ear = (st["ear_l"] + st["ear_r"]) / 2
    put_text(frame, f"EAR: {avg_ear:.3f}  (порог {EAR_THRESHOLD})",
             (PX+4, py), size=13, color=eye_col)
    py += 15
    _bar_h(frame, PX+4, py, PW-10, 9, avg_ear, 0.40, eye_col)
    # Линия порога
    tx = PX + 4 + int((PW-10) * EAR_THRESHOLD / 0.40)
    cv2.line(frame, (tx, py-1), (tx, py+10), C_YELLOW, 1)
    py += 16

    if st["eyes_closed"]:
        dur = st["eye_dur"]
        put_text(frame, f"Закрыты: {dur:.1f} / {EYE_CLOSED_SEC}с",
                 (PX+4, py), size=13, color=eye_col)
        py += 14
        _bar_h(frame, PX+4, py, PW-10, 7, dur, EYE_CLOSED_SEC, eye_col)
        py += 14
    else:
        py += 28

    # ── Правая панель: MAR ────────────────────────────────────────────────
    py += 4
    cv2.line(frame, (PX+4, py), (W-6, py), (50,50,50), 1)
    py += 6
    mouth_col = C_RED if st["mouth_alert"] else (C_ORANGE if st["mouth_open"] else C_GREEN)

    put_text(frame, "РОТ", (PX+4, py), size=14, color=C_GRAY)
    py += 18
    put_text(frame, f"MAR: {st['mar']:.3f}  (порог {MAR_THRESHOLD})",
             (PX+4, py), size=13, color=mouth_col)
    py += 15
    _bar_h(frame, PX+4, py, PW-10, 9, st["mar"], 0.90, mouth_col)
    # Линия порога
    mx = PX + 4 + int((PW-10) * MAR_THRESHOLD / 0.90)
    cv2.line(frame, (mx, py-1), (mx, py+10), C_YELLOW, 1)
    py += 16

    if st["mouth_open"]:
        dur = st["mouth_dur"]
        put_text(frame, f"Открыт: {dur:.1f} / {MOUTH_OPEN_SEC}с",
                 (PX+4, py), size=13, color=mouth_col)
        py += 14
        _bar_h(frame, PX+4, py, PW-10, 7, dur, MOUTH_OPEN_SEC, mouth_col)
        py += 14
    else:
        py += 28

    # ── Статистика ────────────────────────────────────────────────────────
    py += 6
    cv2.line(frame, (PX+4, py), (W-6, py), (50,50,50), 1)
    py += 8
    put_text(frame, "СТАТИСТИКА", (PX+4, py), size=13, color=C_GRAY)
    py += 17
    put_text(frame, f"Морганий:  {det.blinks}", (PX+4, py), size=14, color=C_WHITE)
    py += 17
    put_text(frame, f"Зеваний:   {det.yawns}",  (PX+4, py), size=14, color=C_WHITE)
    py += 17
    # Статус
    e_txt = "ЗАКРЫТЫ" if st["eyes_closed"]  else "открыты"
    m_txt = "ОТКРЫТ"  if st["mouth_open"]   else "закрыт"
    e_col = eye_col   if st["eyes_closed"]  else C_GREEN
    m_col = mouth_col if st["mouth_open"]   else C_GREEN
    put_text(frame, f"Глаза: {e_txt}", (PX+4, py), size=13, color=e_col)
    py += 15
    put_text(frame, f"Рот:   {m_txt}", (PX+4, py), size=13, color=m_col)

    # ── Тревожные баннеры (снизу) ─────────────────────────────────────────
    bh = 52
    bw = PX - 6

    if st["mouth_alert"]:
        ov = frame.copy()
        cv2.rectangle(ov, (4, H-bh*2-8), (bw, H-bh-6), (0, 80, 0), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (4, H-bh*2-8), (bw, H-bh-6), (0,150,0), 1)
        put_text_shadow(frame, "!! ЗЕВОТА ОБНАРУЖЕНА",
                        (14, H-bh*2+4), size=18, color=C_WHITE)
        put_text(frame, "Сделайте разминку или отдохните",
                 (14, H-bh*2+26), size=13, color=C_YELLOW)

    if st["eye_alert"]:
        ov = frame.copy()
        cv2.rectangle(ov, (4, H-bh-4), (bw, H-2), (120, 0, 0), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (4, H-bh-4), (bw, H-2), (200,50,50), 1)
        put_text_shadow(frame, "!! ВНИМАНИЕ: ЗАСЫПАЕТЕ!",
                        (14, H-bh+6), size=19, color=C_WHITE)
        put_text(frame, "Остановитесь и отдохните!",
                 (14, H-bh+30), size=13, color=C_YELLOW)

    # ── Подсказка ─────────────────────────────────────────────────────────
    put_text(frame, "Q - выход   S - скриншот",
             (8, H - 4), size=11, color=(70,70,70))


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 56)
    print("    FATIGUE DETECTOR  v2  —  Детектор усталости")
    print("=" * 56)

    if not _FONT_PATH:
        print("  ПРЕДУПРЕЖДЕНИЕ: TTF-шрифт не найден.")
        print("  Установите DejaVu или Liberation Fonts для кириллицы.")
    else:
        print(f"  Шрифт: {os.path.basename(_FONT_PATH)}")

    init_sound()

    print("  Проверка модели...")
    if not download_model():
        print("\n  ОШИБКА: модель недоступна.")
        print("  Скачайте вручную:")
        print(f"    {MODEL_URL}")
        print(f"  и сохраните как: {MODEL_PATH}")
        sys.exit(1)

    print("  Инициализация FaceLandmarker...")
    try:
        det = FatigueDetector(MODEL_PATH)
    except Exception as e:
        print(f"  ОШИБКА инициализации: {e}")
        sys.exit(1)
    print("  OK. Открываю камеру...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ОШИБКА: камера не найдена (индекс 0).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("  Камера запущена. Нажмите Q для выхода.\n")
    start      = time.time()
    frame_n    = 0
    screenshot = 0

    cv2.namedWindow("Fatigue Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fatigue Detector", 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("  Ошибка чтения кадра.")
            break

        frame_n += 1
        elapsed = time.time() - start
        ts_ms   = int(elapsed * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st  = det.process(rgb, ts_ms)

        render_hud(frame, st, det, elapsed)

        fps = frame_n / max(elapsed, 1e-3)
        put_text(frame, f"FPS {fps:.0f}", (W := frame.shape[1]) - 70,
                 size=12, color=(80,80,80)) if False else \
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (frame.shape[1]-70, 42),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (80,80,80), 1)

        cv2.imshow("Fatigue Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            screenshot += 1
            fn = f"fatigue_{screenshot:03d}.jpg"
            cv2.imwrite(fn, frame)
            print(f"  Скриншот: {fn}")

    cap.release()
    cv2.destroyAllWindows()

    total = int(time.time() - start)
    print("\n" + "═"*40)
    print("  ИТОГИ СЕССИИ")
    print("═"*40)
    print(f"  Время:     {total//60:02d}:{total%60:02d}")
    print(f"  Морганий:  {det.blinks}")
    print(f"  Зеваний:   {det.yawns}")
    print("═"*40)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "="*56)
        print("  ОШИБКА ЗАПУСКА:")
        print("="*56)
        traceback.print_exc()
        print("="*56)
    finally:
        input("\nНажмите Enter для выхода...")
