#!/usr/bin/env python3
"""
Test model nhận diện lệnh giọng nói (KWS) theo thời gian thực trên máy tính.

Cài đặt:   pip install tensorflow sounddevice numpy scipy
Chạy:
    python realtime_kws.py --list-devices                         # xem danh sách micro
    python realtime_kws.py --model kws_out/kws_int8.tflite        # dùng micro mặc định
    python realtime_kws.py --model kws_out/kws_int8.tflite --device 2 --thr 0.85
    python realtime_kws.py --model kws_out/kws_int8.tflite --wav test.wav   # phát lại file wav (không cần micro)

Cơ chế (giống cách sẽ chạy trên ESP32):
    cửa sổ trượt 2 s, cứ mỗi --hop giây chạy model một lần
    -> cổng năng lượng (bỏ qua khi quá yên) -> lệnh chỉ được chấp nhận khi xác suất >= --thr
       trong --votes cửa sổ liên tiếp -> nghỉ --cooldown giây và chỉ kích hoạt lại sau khi
       thấy 2 cửa sổ liên tiếp không phải lệnh.

Các hằng số đặc trưng bên dưới PHẢI khớp với train_kws.py / kws_config.h.
"""
import argparse, csv, os, sys, threading, time, wave
from math import gcd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

SR = 16000
WIN = 2 * SR
N_FFT, FRAME, HOP = 512, 400, 320
N_MEL, FMIN, FMAX = 40, 60.0, 7600.0
SCALE = 0.25
N_FRAMES = 1 + (WIN - FRAME) // HOP
CLASSES = ["background", "unknown", "turn_on", "turn_off", "low", "medium", "high", "max"]
CMD_IDS = [2, 3, 4, 5, 6, 7]
PRETTY = {"turn_on": "TURN ON", "turn_off": "TURN OFF", "low": "SET LOW",
          "medium": "SET MEDIUM", "high": "SET HIGH", "max": "SET MAX"}


# ----------------------------- model + đặc trưng -----------------------------
class Engine:
    def __init__(self, path):
        import tensorflow as tf
        self.tf = tf
        self.mel = tf.signal.linear_to_mel_weight_matrix(N_MEL, N_FFT // 2 + 1, SR, FMIN, FMAX)
        if path.endswith(".tflite"):
            try:                                   # LiteRT (mới) nếu có, nếu không dùng tf.lite
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                Interpreter = tf.lite.Interpreter
            it = Interpreter(model_path=path)
            it.allocate_tensors()
            inp, out = it.get_input_details()[0], it.get_output_details()[0]

            def run(f):
                if inp["dtype"] == np.int8:
                    s, z = inp["quantization"]
                    q = np.clip(np.round(f / s + z), -128, 127).astype(np.int8)
                else:
                    q = f.astype(np.float32)
                it.set_tensor(inp["index"], q)
                it.invoke()
                o = it.get_tensor(out["index"])[0]
                if out["dtype"] == np.int8:
                    s, z = out["quantization"]
                    o = (o.astype(np.float32) - z) * s
                return o.astype(np.float32)
            self.run, self.kind = run, "TFLite " + ("int8" if inp["dtype"] == np.int8 else "float")
        else:
            m = tf.keras.models.load_model(path)
            self.run = lambda f: m(f, training=False).numpy()[0]
            self.kind = "Keras float"

    def feat(self, x):
        tf = self.tf
        s = tf.signal.stft(x[None].astype(np.float32), FRAME, HOP, N_FFT,
                           window_fn=tf.signal.hann_window, pad_end=False)
        lm = tf.math.log(tf.matmul(tf.math.square(tf.abs(s)), self.mel) + 1e-6)
        lm = lm - tf.reduce_mean(lm, axis=[1, 2], keepdims=True)
        return (lm * SCALE)[..., tf.newaxis].numpy()

    def probs(self, x):
        z = self.run(self.feat(x))
        e = np.exp(z - z.max())
        return e / e.sum()


# ----------------------------- quyết định lệnh -----------------------------
class Decider:
    def __init__(self, thr, votes, cooldown):
        self.thr, self.votes, self.cooldown = thr, votes, cooldown
        self.streak, self.last, self.cool_until, self.armed, self.quiet = 0, None, -1e9, True, 0

    def update(self, p, t):
        c = int(np.argmax(p))
        is_cmd = c in CMD_IDS
        if not self.armed:                      # chờ thấy 2 cửa sổ "không phải lệnh" mới nhận lệnh tiếp
            self.quiet = 0 if is_cmd else self.quiet + 1
            if self.quiet >= 2 and t >= self.cool_until:
                self.armed = True
            return None
        if t < self.cool_until:
            return None
        if is_cmd and p[c] >= self.thr:
            self.streak = self.streak + 1 if c == self.last else 1
            self.last = c
            if self.streak >= self.votes:
                self.streak, self.last, self.armed, self.quiet = 0, None, False, 0
                self.cool_until = t + self.cooldown
                return c, float(p[c])
        else:
            self.streak, self.last = 0, None
        return None


def frame_db(x):
    """Mức (dBFS) của khung 20 ms to nhất trong cửa sổ."""
    n = len(x) // 320
    r = np.sqrt((x[: n * 320].reshape(n, 320) ** 2).mean(1)).max() if n else 0.0
    return 20 * np.log10(r + 1e-9), r


class Session:
    def __init__(self, a, eng, show_status):
        self.a, self.eng, self.show_status = a, eng, show_status
        self.dec = Decider(a.thr, a.votes, a.cooldown)
        self.events, self.n, self.infer_ms = [], 0, 0.0
        self.last_len = 0
        self.log = None
        if a.log:
            self.log = open(a.log, "w", newline="", encoding="utf-8")
            self.csv = csv.writer(self.log)
            self.csv.writerow(["t", "level_db", "gate_open"] + CLASSES + ["trigger"])
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)

    def feed(self, x, t):
        lvl_db, rms = frame_db(x)
        gate = rms >= self.a.gate
        if gate:
            t0 = time.perf_counter()
            p = self.eng.probs(x)
            self.infer_ms += (time.perf_counter() - t0) * 1000
            self.n += 1
        else:
            p = np.zeros(len(CLASSES), np.float32)
            p[0] = 1.0                           # quá yên -> coi là background
        ev = self.dec.update(p, t)
        top = int(np.argmax(p))
        if self.show_status:
            bar = "#" * int(round(np.clip((lvl_db + 60) / 60, 0, 1) * 20))
            line = (f"{t:6.1f}s [{bar:<20}] {lvl_db:6.1f}dB {'*' if gate else '.'} "
                    f"{CLASSES[top]:10s} {p[top]:.2f} | bg {p[0]:.2f} unk {p[1]:.2f}")
            self.last_len = len(line)
            sys.stdout.write("\r" + line + "    ")
            sys.stdout.flush()
        if self.a.verbose:
            sys.stdout.write("\r" + " " * (self.last_len + 4) + "\r")
            print(f"{t:6.1f}s " + " ".join(f"{c[:6]}={v:.2f}" for c, v in zip(CLASSES, p)))
        if ev:
            c, pc = ev
            self.events.append((t, CLASSES[c], pc))
            sys.stdout.write("\r" + " " * (self.last_len + 4) + "\r")
            print(f">>> [{t:6.1f}s] LỆNH: {PRETTY[CLASSES[c]]}  (độ tin cậy {pc:.2f})")
            if self.a.save_dir:
                self.save(x, t, CLASSES[c], pc)
        if self.log:
            self.csv.writerow([f"{t:.2f}", f"{lvl_db:.1f}", int(gate)] + [f"{v:.3f}" for v in p]
                              + [CLASSES[ev[0]] if ev else ""])

    def save(self, x, t, name, p):
        path = os.path.join(self.a.save_dir, f"trig_{t:08.2f}_{name}_{p:.2f}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

    def close(self):
        if self.log:
            self.log.close()


# ----------------------------- nguồn âm thanh -----------------------------
def to_16k(x, rate):
    if rate == SR:
        return x
    from scipy.signal import resample_poly
    g = gcd(SR, rate)
    return resample_poly(x, SR // g, rate // g).astype(np.float32)


def fit(x):
    if len(x) >= WIN:
        return x[-WIN:]
    return np.concatenate([np.zeros(WIN - len(x), np.float32), x])


def read_wav(path):
    with wave.open(path) as w:
        rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        assert sw == 2, "chỉ hỗ trợ wav 16-bit"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return to_16k(x, rate)


def run_wav(a, eng):
    x = read_wav(a.wav)
    ses = Session(a, eng, show_status=False)
    hop = int(a.hop * SR)
    print(f"Phát lại '{a.wav}' ({len(x) / SR:.1f} s), hop {a.hop}s, mô hình: {eng.kind}")
    for pos in range(WIN, len(x) + 1, hop):          # bắt đầu khi bộ đệm 2 s đã đầy (giống chế độ micro)
        ses.feed(x[pos - WIN:pos], pos / SR)
    ses.close()
    print(f"Xong. {len(ses.events)} lệnh được nhận; trung bình {ses.infer_ms / max(1, ses.n):.1f} ms/lần suy luận trên PC.")


def run_mic(a, eng):
    import sounddevice as sd
    dev = a.device
    info = sd.query_devices(dev, "input")
    rate = SR
    try:
        sd.check_input_settings(device=dev, samplerate=SR, channels=1)
    except Exception:
        rate = int(info["default_samplerate"])
    buf = np.zeros(int(WIN * rate / SR) + rate // 10, np.float32)
    lock = threading.Lock()

    def cb(indata, frames, tm, status):
        n = len(indata)
        with lock:
            buf[:-n] = buf[n:]
            buf[-n:] = indata[:, 0]

    print(f"Micro: {info['name']} @ {rate} Hz | mô hình: {eng.kind}")
    print(f"thr={a.thr} votes={a.votes} cooldown={a.cooldown}s gate={a.gate} | Ctrl+C để thoát.")
    print("Ký hiệu: '*' = có tiếng (qua cổng năng lượng), '.' = quá yên, bỏ qua.\n")
    ses = Session(a, eng, show_status=True)
    t0 = time.monotonic()
    nxt = t0 + a.hop
    try:
        with sd.InputStream(device=dev, samplerate=rate, channels=1, dtype="float32",
                            blocksize=rate // 10, callback=cb):
            while True:
                time.sleep(max(0.0, nxt - time.monotonic()))
                nxt += a.hop
                t = time.monotonic() - t0
                if t < 2.0:                       # chờ đầy bộ đệm 2 s
                    continue
                with lock:
                    snap = buf.copy()
                ses.feed(fit(to_16k(snap, rate)), t)
    except KeyboardInterrupt:
        pass
    finally:
        ses.close()
        print(f"\nĐã dừng. {len(ses.events)} lệnh; {ses.infer_ms / max(1, ses.n):.1f} ms/lần suy luận trên PC.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kws_out/kws_int8.tflite", help=".tflite (int8/float) hoặc .keras")
    ap.add_argument("--wav", help="phát lại file wav thay vì dùng micro")
    ap.add_argument("--device", default=None, help="chỉ số hoặc tên micro")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--hop", type=float, default=0.2, help="chu kỳ chạy model (giây)")
    ap.add_argument("--thr", type=float, default=0.8, help="ngưỡng xác suất để nhận lệnh")
    ap.add_argument("--votes", type=int, default=2, help="số cửa sổ liên tiếp phải cùng một lệnh")
    ap.add_argument("--cooldown", type=float, default=1.0, help="thời gian nghỉ sau mỗi lệnh (giây)")
    ap.add_argument("--gate", type=float, default=0.01, help="ngưỡng RMS khung 20 ms để chạy model (0 = tắt cổng)")
    ap.add_argument("--verbose", action="store_true", help="in xác suất của mọi lớp ở mỗi cửa sổ")
    ap.add_argument("--log", help="ghi CSV xác suất từng cửa sổ (để chỉnh ngưỡng)")
    ap.add_argument("--save-dir", help="lưu cửa sổ 2 s mỗi lần kích hoạt (để nghe lại / lọc báo nhầm)")
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if a.device is not None and str(a.device).isdigit():
        a.device = int(a.device)
    eng = Engine(a.model)
    run_wav(a, eng) if a.wav else run_mic(a, eng)


if __name__ == "__main__":
    main()