#!/usr/bin/env python3
"""
Train TC-ResNet8 nhận diện 6 lệnh giọng nói cho ESP32 (keyword spotting) - int8 / TFLite Micro.
File ĐỘC LẬP: không cần train_kws.py hay compare_models.py.

Cấu trúc thư mục dữ liệu (giải nén các zip vào CÙNG một thư mục):
    data/
      dataset2/ {turn_on,turn_off,low,medium,high,max,unknown/**,background_noise/**}
      dataset3/ ...
      dataset4/ ...

Chạy (không cần tham số - mọi thứ chỉnh ở khối CẤU HÌNH CHẠY bên dưới):
    python train_tc_resnet8.py          # Colab: !python train_tc_resnet8.py
Cờ dòng lệnh (tùy chọn) chỉ ghi đè giá trị mặc định, ví dụ: python train_tc_resnet8.py --epochs 2

    MODE = "final" (mặc định): chừa một số GIỌNG làm validation, train phần còn lại, in val_acc + đánh giá theo SNR;
                    nếu RETRAIN_FULL = True thì train LẠI trên toàn bộ dữ liệu rồi mới xuất model cho ESP32
    MODE = "cv"   : mỗi lần chừa nguyên một dataset (dataset3, dataset4) làm validation -> số liệu đánh giá chéo

Kết quả trong OUT_DIR:
    kws_int8.tflite, kws_model.keras, kws_model_data.cc/.h, kws_config.h, val_split.json
Nạp lên ESP32: kws_model_data.cc/.h + kws_config.h (CÙNG một lần train; tham số lượng tử hóa khác nhau giữa các lần).
Model xuất ra là logits (không có softmax): firmware tự lấy argmax/softmax.

8 lớp: background, unknown, turn_on, turn_off, low, medium, high, max
"""
import argparse, glob, os, re, wave, json
import numpy as np
import tensorflow as tf

# ================================ CẤU HÌNH CHẠY (chỉnh ở đây) ================================
DATA_DIR = "data/data"                  # thư mục chứa dataset2/3/4
OUT_DIR = "tc_resnet8_out"         # nơi ghi model, kws_config.h, val_split.json, saved_model
MODE = "final"                  # "final" | "cv"
EPOCHS = 150
SEED = 0
BATCH_SIZE = 32
LR = 3e-3                          # learning rate đầu của CosineDecay

# nhiễu / tăng cường khi train
SNR_MIN, SNR_MAX = -5.0, 20.0      # dB
USE_NOISE_MIX = True              # False: KHÔNG trộn nhiễu/babble vào mẫu lệnh khi train (lớp background vẫn lấy từ noise)
P_NOISE = 0.85                     # xác suất trộn nhiễu cho mỗi mẫu
BABBLE_PROB = 0.3                  # tỉ lệ nhiễu là tiếng nhiều người nói (0 = tắt)
EXTRA_NOISE_DIRS = []              # thư mục wav nhiễu bổ sung, vd ["extra_noise"]

# mode cv
CV_HELD_OUT = ["dataset3", "dataset4"]

# mode final: tập validation chừa ra
VAL_FRAC = 0.15                    # tỉ lệ chừa làm validation (0 = train toàn bộ, không có val)
VAL_EVERY = 5                      # in val_loss/val_acc mỗi N epoch
SPLIT_BY = "speaker"               # "speaker": chừa nguyên từng GIỌNG làm val (khuyên dùng) | "random": phân tầng, cùng giọng có thể ở cả 2 tập | "dataset": chừa nguyên dataset
HOLDOUT_DATASETS = ["dataset4"]    # dùng khi SPLIT_BY = "dataset"
SPLIT_SEED = 0                     # seed riêng cho việc chia -> cùng tập val giữa các lần chạy
# SPLIT_BY = "speaker": chọn giọng làm validation
VAL_SPEAKERS = []                  # liệt kê tay các giọng làm val (vd ["dataset4:alex", "tts:en-GB-RyanNeural"]); rỗng = tự chọn ngẫu nhiên (cố định theo SPLIT_SEED) theo 2 số bên dưới
VAL_REAL_SPEAKERS = 2              # số nhóm giọng THẬT tự chọn làm val (đo trên giọng người, không chỉ TTS)
VAL_TTS_VOICES = 6                 # số giọng TTS tự chọn làm val (6/44 giọng ~ 14%)
RETRAIN_FULL = True                # True: sau khi xong, train lại trên toàn bộ dữ liệu để xuất model (khuyên dùng, gấp đôi thời gian)

# kiến trúc TC-ResNet8
CHANNELS = [16, 24, 32, 48]        # số kênh: lớp đầu + 3 khối residual (mỗi khối stride 2 theo thời gian). [16,24,24,32,32,48,48] = TC-ResNet14
DROPOUT = 0.2                      # dropout trước lớp phân loại

# sau khi train: kiểm tra bộ nhớ ESP32 (arena đo bằng TFLite Micro thật nếu đã `pip install tflite-micro`)
CHECK_MEMORY = True
RAM_BUDGET_KB = 320                # DRAM nội ESP32 (giả định; ESP32-S3 điền theo board)
OTHER_RAM_KB = 80                  # FreeRTOS, stack, I2S DMA, OLED, PWM, (WiFi nếu dùng)... GIẢ ĐỊNH: hãy đo bằng heap_caps_get_free_size() trong firmware trống
FLASH_BUDGET_KB = 1000             # flash dành cho model (phụ thuộc bảng phân vùng)
DATA_DIR_FALLBACKS = ["data", "../data", "/content/data/data", "/content/data"]   # thử khi DATA_DIR không có dataset*
# ==============================================================================================


# ----------------------------- cấu hình (phải khớp code C trên ESP32) -----------------------------
SR = 16000
WIN = 2 * SR                      # cửa sổ 2 giây
N_FFT, FRAME, HOP = 512, 400, 320  # 25 ms / hop 20 ms -> 99 khung cho 2 s
N_MEL, FMIN, FMAX = 40, 60.0, 7600.0
SCALE = 0.25                       # đặc trưng = (logmel - mean(logmel)) * SCALE
N_FRAMES = 1 + (WIN - FRAME) // HOP

CLASSES = ["background", "unknown", "turn_on", "turn_off", "low", "medium", "high", "max"]
CMD_DIRS = ["turn_on", "turn_off", "low", "medium", "high", "max"]
BG, UNK = 0, 1
SWEEP = [None, 20, 10, 5, 0]       # SNR (dB) dùng khi đánh giá chéo; None = sạch
CMD_IDS = [CLASSES.index(c) for c in CMD_DIRS]
TTS_RE = re.compile(r"^(Aria|Guy|Jenny|Steffan)_")
TTS_BULK_RE = re.compile(r"^tts_(.+?)_t\d+\.wav$")
NAMED_RE = re.compile(r"^([A-Za-z]+?)\s*(?:\(\d+\)|\d*)\.wav$")   # alex (3).wav, Mark2.wav, jes.wav
TIMESTAMP_RE = re.compile(r"^(?:[a-z_]+_)?\d{10,}\.wav$")           # turn_on_1792220780227.wav, unknown_17916....wav


def speaker_of(path, ds):
    """Suy ra ĐỊNH DANH GIỌNG từ tên file (dữ liệu không có nhãn người nói).
    - tts_en-GB-RyanNeural_t3.wav -> "tts:en-GB-RyanNeural"        (1 giọng TTS)
    - Aria_take_2.wav             -> "take:Aria"                    (giọng TTS đặt tên)
    - Mark1.wav / alex (3).wav    -> "dataset4:mark" / "dataset4:alex"
    - <lớp>_<timestamp>.wav       -> "<dataset>:rec" (ghi âm thật, KHÔNG biết ai nói -> coi cả nhóm là 1 giọng)"""
    n = os.path.basename(path)
    m = TTS_BULK_RE.match(n)
    if m:
        return "tts:" + m.group(1)
    if TTS_RE.match(n):
        return "take:" + n.split("_")[0]
    if TIMESTAMP_RE.match(n):
        return f"{ds}:rec"
    m = NAMED_RE.match(n)
    if m:
        return f"{ds}:{m.group(1).lower()}"
    return f"{ds}:rec"


def kind_of(spk):
    return "tts" if spk.startswith(("tts:", "take:")) else "real"


# ----------------------------- đọc dữ liệu -----------------------------
def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def read_noise_any(path):
    """Đọc file nhiễu bổ sung (wav 16-bit, mono/stereo, tần số bất kỳ) -> mono 16 kHz."""
    with wave.open(path) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        assert w.getsampwidth() == 2, f"chỉ hỗ trợ wav 16-bit: {path}"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if rate != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, rate)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    return x


def load_extra_noise(dirs):
    out = []
    for d in dirs or []:
        fs = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
        print(f"  nhiễu bổ sung: {len(fs)} file từ {d}")
        out += [read_noise_any(f) for f in fs]
    return out


def collect(root):
    items, noise = [], []          # items: (path, label, dataset, tts/real, speaker_id)
    root = os.path.abspath(root)
    ds_dirs = sorted(d for d in glob.glob(os.path.join(root, "dataset*")) if os.path.isdir(d))
    if not ds_dirs:
        raise SystemExit(f"Không thấy thư mục dataset* trong '{root}'. Kiểm tra lại --data.")
    for ds_path in ds_dirs:
        ds = os.path.basename(ds_path)
        n_cmd = 0
        for c in CMD_DIRS:
            for f in sorted(glob.glob(os.path.join(ds_path, c, "*.wav"))):
                spk = speaker_of(f, ds)
                items.append((f, CLASSES.index(c), ds, kind_of(spk), spk)); n_cmd += 1
        n_unk = 0
        for f in sorted(glob.glob(os.path.join(ds_path, "unknown", "**", "*.wav"), recursive=True)):
            spk = speaker_of(f, ds)
            items.append((f, UNK, ds, kind_of(spk), spk)); n_unk += 1
        # thư mục noise: chấp nhận mọi tên có chứa "noise" hoặc "background"
        n_noise = 0
        for d in sorted(os.listdir(ds_path)):
            if os.path.isdir(os.path.join(ds_path, d)) and any(k in d.lower() for k in ("noise", "background")):
                for f in sorted(glob.glob(os.path.join(ds_path, d, "**", "*.wav"), recursive=True)):
                    noise.append((f, ds)); n_noise += 1
        print(f"  {ds}: {n_cmd} lệnh, {n_unk} unknown, {n_noise} noise")
    if not noise:
        raise SystemExit("Không tìm thấy file noise nào (thư mục background_noise/**.wav). "
                         "Hãy kiểm tra cấu trúc: data/datasetX/background_noise/<loại>/*.wav")
    if not any(it[1] == UNK for it in items):
        raise SystemExit("Không tìm thấy file unknown (datasetX/unknown/**.wav).")
    return items, noise


# ----------------------------- đặc trưng (giống hệt phía ESP32) -----------------------------
_MEL = tf.signal.linear_to_mel_weight_matrix(N_MEL, N_FFT // 2 + 1, SR, FMIN, FMAX)


def logmel(wav):                   # wav: (B, WIN) float32 trong [-1, 1]
    s = tf.signal.stft(wav, FRAME, HOP, N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    p = tf.math.square(tf.abs(s))
    lm = tf.math.log(tf.matmul(p, _MEL) + 1e-6)
    lm = lm - tf.reduce_mean(lm, axis=[1, 2], keepdims=True)   # bất biến theo độ lớn (gain)
    return (lm * SCALE)[..., tf.newaxis]


def features(wavs, bs=128):
    return np.concatenate([logmel(tf.constant(wavs[i:i + bs])).numpy() for i in range(0, len(wavs), bs)])


# ----------------------------- xử lý / tăng cường tín hiệu -----------------------------
def speech_rms(x):
    n = len(x) // 160
    e = (x[: n * 160].reshape(n, 160) ** 2).mean(1)
    k = max(1, int(0.3 * n))
    return float(np.sqrt(np.sort(e)[-k:].mean()) + 1e-9)


def fit_window(x, rng=None, jitter=0.3):
    """Cắt/đệm về đúng 2 s. Clip dài (unknown 3 s) được cắt quanh vùng có tiếng."""
    if len(x) > WIN:
        n = len(x) // 160
        e = (x[: n * 160].reshape(n, 160) ** 2).mean(1)
        centre = int((np.arange(n) * e).sum() / (e.sum() + 1e-12) * 160)
        if rng is not None:
            centre += int(rng.uniform(-jitter, jitter) * SR)
        start = int(np.clip(centre - WIN // 2, 0, len(x) - WIN))
        return x[start:start + WIN].copy()
    out = np.zeros(WIN, np.float32)
    out[: len(x)] = x
    return out


def shift(x, s):
    out = np.zeros_like(x)
    if s > 0:
        out[s:] = x[:-s]
    elif s < 0:
        out[:s] = x[-s:]
    else:
        out[:] = x
    return out


def crop_noise(pool, rng):
    n = pool[rng.integers(len(pool))]
    if len(n) <= WIN:
        return fit_window(n)
    st = rng.integers(0, len(n) - WIN + 1)
    return n[st:st + WIN].copy()


def mix(x, noise, snr_db):
    g = speech_rms(x) / (speech_rms(noise) * 10 ** (snr_db / 20))
    return x + noise * g


def babble(speech, rng):
    """Nhiễu "nhiều người nói chuyện": cộng 4-7 đoạn tiếng nói chồng lên nhau."""
    y = np.zeros(WIN, np.float32)
    for _ in range(int(rng.integers(4, 8))):
        s = shift(speech[rng.integers(len(speech))], int(rng.uniform(-0.8, 0.8) * SR))
        y += s * 10 ** (rng.uniform(-6, 0) / 20) / (speech_rms(s) + 1e-9)
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


class NoiseSource:
    """Nguồn nhiễu khi train: file nhiễu thật + (tùy chọn) babble tạo từ chính tiếng nói trong tập train."""
    def __init__(self, pool, speech, babble_prob):
        self.pool, self.speech, self.pb = pool, speech, babble_prob

    def sample(self, rng):
        if self.speech and rng.random() < self.pb:
            return babble(self.speech, rng)
        return crop_noise(self.pool, rng)


def augment(x, src, rng, cfg):
    x = fit_window(x, rng)
    if rng.random() < 0.3:                                   # đổi tốc độ nhẹ
        f = rng.uniform(0.9, 1.1)
        x = np.interp(np.arange(0, len(x) - 1, f), np.arange(len(x)), x).astype(np.float32)
        x = fit_window(x)
    x = shift(x, int(rng.uniform(-0.4, 0.4) * SR))
    if cfg.noise_mix and rng.random() < cfg.p_noise:     # trộn nhiễu với SNR ngẫu nhiên
        x = mix(x, src.sample(rng), rng.uniform(cfg.snr_min, cfg.snr_max))
    x = x * 10 ** (rng.uniform(-18, 6) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def background_sample(src, rng):
    x = src.sample(rng)
    r = rng.random()
    if r < 0.1:
        x = x * 0.01                                         # gần như im lặng
    else:
        x = x * 10 ** (rng.uniform(-20, 6) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def spec_augment(f, rng):
    f = f.copy()
    for i in range(len(f)):
        t = rng.integers(0, 7); t0 = rng.integers(0, N_FRAMES - t)
        f[i, t0:t0 + t, :, :] = 0
        m = rng.integers(0, 4); m0 = rng.integers(0, N_MEL - m)
        f[i, :, m0:m0 + m, :] = 0
    return f


# ----------------------------- mô hình TC-ResNet8 -----------------------------
def build_model():
    """TC-ResNet (Choi et al., 2019): coi 40 băng mel là KÊNH, conv chỉ chạy theo trục thời gian (kernel 9), mỗi khối
    stride 2 + shortcut 1x1. Lớp đầu viết bằng Conv2D có kernel (3, N_MEL) thay cho Reshape để TFLite không sinh
    toán tử SHAPE/PACK. Chỉ dùng Conv2D / Add / ReLU / BatchNorm (gộp vào conv khi chuyển int8) / Mean / Dense / Pad."""
    L = tf.keras.layers

    def bn_act(x, act=True):
        x = L.BatchNormalization()(x)
        return L.ReLU()(x) if act else x

    inp = L.Input((N_FRAMES, N_MEL, 1))
    x = L.ZeroPadding2D(((1, 1), (0, 0)))(inp)
    x = L.Conv2D(CHANNELS[0], (3, N_MEL), padding="valid", use_bias=False)(x)       # (T, 1, C0)
    x = bn_act(x)
    for ch in CHANNELS[1:]:
        sc = bn_act(L.Conv2D(ch, 1, strides=(2, 1), padding="same", use_bias=False)(x), act=False)
        y = bn_act(L.Conv2D(ch, (9, 1), strides=(2, 1), padding="same", use_bias=False)(x))
        y = bn_act(L.Conv2D(ch, (9, 1), padding="same", use_bias=False)(y), act=False)
        x = L.ReLU()(L.Add()([sc, y]))
    x = L.GlobalAveragePooling2D()(x)
    if DROPOUT:
        x = L.Dropout(DROPOUT)(x)
    out = L.Dense(len(CLASSES))(x)                           # logits (softmax tính khi suy luận)
    return tf.keras.Model(inp, out, name="tc_resnet8")


# ----------------------------- đánh giá -----------------------------
def build_eval(items, noise_files, wav_cache, noise_wavs, snr_db=None, seed=123):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for p, lab, *_ in items:
        x = fit_window(wav_cache[p])
        if snr_db is not None and noise_wavs:
            x = np.clip(mix(x, crop_noise(noise_wavs, rng), snr_db), -1, 1)
        X.append(x.astype(np.float32)); y.append(lab)
    for n in noise_wavs:                                     # lớp background: cắt các đoạn 2 s cố định
        for st in range(0, max(1, len(n) - WIN + 1), WIN):
            X.append(fit_window(n[st:st + WIN])); y.append(BG)
    return np.stack(X), np.array(y)


def report(y, pred, title, verbose=True):
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    if verbose:
        print(f"\n=== {title} ===")
        print("recall theo lớp:")
        for i, c in enumerate(CLASSES):
            if cm[i].sum():
                print(f"  {c:11s} {cm[i, i] / cm[i].sum() * 100:5.1f}%  (n={cm[i].sum()})")
    is_cmd = np.isin(y, CMD_IDS)
    cmd_acc = (pred[is_cmd] == y[is_cmd]).mean()
    miss = (~np.isin(pred[is_cmd], CMD_IDS)).mean()
    neg = ~is_cmd
    fa = np.isin(pred[neg], CMD_IDS).mean()
    if not verbose:
        return cmd_acc, fa
    print(f"  -> độ chính xác 6 lệnh: {cmd_acc * 100:.1f}% | lệnh bị bỏ sót (ra background/unknown): {miss * 100:.1f}%"
          f" | báo nhầm (unknown/background -> lệnh): {fa * 100:.1f}%")
    print("ma trận nhầm lẫn (hàng = thật, cột = đoán):")
    print("             " + " ".join(f"{c[:6]:>6s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:11s}" + " ".join(f"{v:6d}" for v in cm[i]))
    return cmd_acc, fa


def predict(model, X):
    return model.predict(features(X), verbose=0).argmax(1)


def predict_tflite(path, F):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    out = []
    for f in F:
        q = np.clip(np.round(f / s + z), -128, 127).astype(np.int8)[None]
        it.set_tensor(i["index"], q); it.invoke()
        out.append(it.get_tensor(o["index"])[0].argmax())
    return np.array(out)


# ----------------------------- huấn luyện -----------------------------
def train(train_items, noise_pool, wav_cache, epochs, rng, cfg, bs=BATCH_SIZE, val=None):
    cmds = [it for it in train_items if it[1] in CMD_IDS]
    unks = [it for it in train_items if it[1] == UNK]
    speech = [fit_window(wav_cache[p]) for p, *_ in train_items] if cfg.babble_prob > 0 else []
    src = NoiseSource(noise_pool, speech, cfg.babble_prob)
    n_bg, n_unk = max(40, len(cmds) // 6), min(len(unks), max(60, len(cmds) // 4))
    steps = epochs * int(np.ceil((len(cmds) + n_unk + n_bg) / bs))
    lr = tf.keras.optimizers.schedules.CosineDecay(LR, steps)
    model = build_model()
    model.compile(tf.keras.optimizers.Adam(lr),
                  tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0),
                  metrics=["accuracy"])
    for ep in range(epochs):
        sel = [unks[i] for i in rng.choice(len(unks), n_unk, replace=False)]
        W, y = [], []
        for p, lab, *_ in cmds + sel:
            W.append(augment(wav_cache[p], src, rng, cfg)); y.append(lab)
        for _ in range(n_bg):
            W.append(background_sample(src, rng)); y.append(BG)
        F = spec_augment(features(np.stack(W)), rng)
        Y = tf.keras.utils.to_categorical(y, len(CLASSES))
        h = model.fit(F, Y, batch_size=bs, epochs=1, shuffle=True, verbose=0)
        if (ep + 1) % VAL_EVERY == 0 or ep == 0:
            msg = f"  epoch {ep + 1:3d}/{epochs}  loss={h.history['loss'][0]:.3f}  acc={h.history['accuracy'][0]:.3f}"
            if val is not None:
                vl, va = model.evaluate(val[0], val[1], batch_size=128, verbose=0)
                msg += f"  val_loss={vl:.3f}  val_acc={va:.3f}"
            print(msg, flush=True)
    return model


def convert_int8(model, rep_F, path):
    saved = os.path.join(os.path.dirname(os.path.abspath(path)), "saved_model")   # thư mục trung gian, cạnh file .tflite
    model.export(saved)                                            # Keras 3: xuất SavedModel
    conv = tf.lite.TFLiteConverter.from_saved_model(saved)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    def rep_gen():
        for f in rep_F:
            yield [f[None].astype(np.float32)]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    open(path, "wb").write(data)
    print(f"  đã ghi {path} ({len(data) / 1024:.1f} KB)")
    return path


def export_c(tflite_path, out_dir):
    data = open(tflite_path, "rb").read()
    it = tf.lite.Interpreter(model_path=tflite_path); it.allocate_tensors()
    s, z = it.get_input_details()[0]["quantization"]
    os_, oz = it.get_output_details()[0]["quantization"]
    with open(os.path.join(out_dir, "kws_model_data.cc"), "w") as f:
        f.write('#include "kws_model_data.h"\nalignas(16) const unsigned char g_kws_model[] = {\n')
        for i in range(0, len(data), 16):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",\n")
        f.write(f"}};\nconst unsigned int g_kws_model_len = {len(data)};\n")
    with open(os.path.join(out_dir, "kws_model_data.h"), "w") as f:
        f.write("#pragma once\nextern const unsigned char g_kws_model[];\nextern const unsigned int g_kws_model_len;\n")
    with open(os.path.join(out_dir, "kws_config.h"), "w") as f:
        f.write("#pragma once\n// Tham số đặc trưng - phải khớp tuyệt đối với train_tc_resnet8.py\n")
        f.write(f"#define KWS_SAMPLE_RATE {SR}\n#define KWS_WINDOW_SAMPLES {WIN}\n#define KWS_N_FFT {N_FFT}\n")
        f.write(f"#define KWS_FRAME_LEN {FRAME}\n#define KWS_HOP {HOP}\n#define KWS_N_MEL {N_MEL}\n#define KWS_N_FRAMES {N_FRAMES}\n")
        f.write(f"#define KWS_FMIN {FMIN}f\n#define KWS_FMAX {FMAX}f\n#define KWS_FEATURE_SCALE {SCALE}f\n")
        f.write(f"#define KWS_IN_SCALE {float(s)}f\n#define KWS_IN_ZERO_POINT {int(z)}\n")
        f.write(f"#define KWS_OUT_SCALE {float(os_)}f\n#define KWS_OUT_ZERO_POINT {int(oz)}\n")
        f.write(f"#define KWS_NUM_CLASSES {len(CLASSES)}\n")
        f.write("static const char* const KWS_LABELS[KWS_NUM_CLASSES] = {" + ", ".join(f'"{c}"' for c in CLASSES) + "};\n")
    print("  đã ghi kws_model_data.cc/.h và kws_config.h")


# ----------------------------- tách tập validation -----------------------------
def split_by_speaker(items, noise, seed):
    """Chia val theo GIỌNG: mọi clip của một giọng nằm trọn ở train HOẶC val -> val đo trên giọng chưa từng thấy.
    Giọng làm val: VAL_SPEAKERS nếu có, ngược lại tự chọn VAL_TTS_VOICES giọng TTS + VAL_REAL_SPEAKERS nhóm giọng thật.
    Lưu ý: nhóm "<dataset>:rec" (tên file là timestamp) có thể gồm nhiều người nhưng không phân biệt được -> chia theo cả nhóm."""
    groups = {}
    for it in items:
        groups.setdefault(it[4], []).append(it)
    unknown = [s for s in VAL_SPEAKERS if s not in groups]
    if unknown:
        raise SystemExit(f"VAL_SPEAKERS có giọng không tồn tại: {unknown}\nGiọng hiện có: {sorted(groups)}")
    if VAL_SPEAKERS:
        held = list(VAL_SPEAKERS)
    else:
        rng = np.random.default_rng(seed)
        tts = sorted(s for s in groups if kind_of(s) == "tts")
        real = sorted(s for s in groups if kind_of(s) == "real")
        held = list(rng.choice(tts, min(VAL_TTS_VOICES, len(tts)), replace=False)) if tts else []
        # giọng thật: ưu tiên giọng đặt tên (có người nói rõ ràng) để mỗi giọng đều có đủ 6 lệnh
        held += list(rng.choice(real, min(VAL_REAL_SPEAKERS, len(real)), replace=False)) if real else []
    held = set(held)
    tr = [it for it in items if it[4] not in held]
    te = [it for it in items if it[4] in held]
    # file noise không gắn với người nói: chia ngẫu nhiên theo dataset (giữ >= 1 file ở mỗi tập nếu có thể)
    rng = np.random.default_rng(seed)
    tr_n, te_n, by_ds = [], [], {}
    for n in noise:
        by_ds.setdefault(n[1], []).append(n)
    for _, v in sorted(by_ds.items()):
        idx = rng.permutation(len(v)); k = min(max(int(round(0.15 * len(v))), 1), len(v) - 1) if len(v) >= 2 else 0
        te_n += [v[i] for i in idx[:k]]; tr_n += [v[i] for i in idx[k:]]
    print("\n  giọng làm VALIDATION (không xuất hiện ở train):")
    for sp in sorted(held):
        g = groups[sp]
        print(f"    {sp:38s} {len(g):4d} clip  ({sum(1 for x in g if x[1] in CMD_IDS)} lệnh, {sum(1 for x in g if x[1] == UNK)} unknown)")
    n_real_val = sum(1 for x in te if x[3] == "real")
    print(f"  val: {len(te)} clip, trong đó {n_real_val} clip giọng thật | train: {len(tr)} clip, "
          f"{sum(1 for x in tr if x[3] == 'real')} clip giọng thật")
    for c, name in enumerate(CLASSES[1:], start=1):
        if not any(x[1] == c for x in te):
            print(f"  (cảnh báo: val không có clip nào của lớp '{name}')")
    return tr, te, tr_n, te_n


def split_holdout(items, noise, val_frac, split_by, holdout_ds, seed):
    """Chừa một tập validation khỏi dữ liệu train.
    split_by="random" : mỗi nhóm (lớp, dataset, tts/real) lấy val_frac làm validation (phân tầng, mỗi nhóm >= 1 mẫu).
                        Lưu ý: cùng một người nói có thể nằm ở cả train lẫn val -> điểm số hơi lạc quan.
    split_by="dataset": toàn bộ các dataset trong holdout_ds làm validation (người nói chưa từng thấy)."""
    if split_by == "speaker":
        return split_by_speaker(items, noise, seed)
    if split_by == "dataset":
        isin = lambda d: d in holdout_ds
        tr = [it for it in items if not isin(it[2])]; te = [it for it in items if isin(it[2])]
        tr_n = [n for n in noise if not isin(n[1])]; te_n = [n for n in noise if isin(n[1])]
        return tr, te, tr_n, te_n
    if val_frac <= 0:
        return list(items), [], list(noise), []
    rng = np.random.default_rng(seed)

    def pick(group):                                   # trả về (phần train, phần val)
        idx = rng.permutation(len(group))
        k = int(round(val_frac * len(group)))
        k = min(max(k, 1), len(group) - 1) if len(group) >= 2 else 0
        return [group[i] for i in idx[k:]], [group[i] for i in idx[:k]]

    def by_key(seq, key):
        g = {}
        for s in seq:
            g.setdefault(key(s), []).append(s)
        tr_, te_ = [], []
        for _, v in sorted(g.items()):
            a, b = pick(v); tr_ += a; te_ += b
        return tr_, te_

    tr, te = by_key(items, lambda it: (it[1], it[2], it[3]))
    tr_n, te_n = by_key(noise, lambda n: n[1])
    return tr, te, tr_n, te_n


def make_val_features(items, noise, wav_cache, val_noise_wavs):
    """Dựng (features, one-hot) cho tập validation (clip sạch + đoạn noise cố định) để theo dõi lúc train."""
    X, y = build_eval(items, noise, wav_cache, val_noise_wavs)
    return features(X), tf.keras.utils.to_categorical(y, len(CLASSES))


def evaluate_model(model, tr, te, noise, val_noise, wav_cache, out_dir, fname, title):
    """Đánh giá float + quét SNR + int8 TFLite trên tập validation te. Trả về (sweep, acc_int8, fa_int8, đường dẫn .tflite)."""
    Xc, yc = build_eval(te, noise, wav_cache, val_noise)
    report(yc, predict(model, Xc), f"{title} - sạch (float)")
    sweep = {}
    for snr in SWEEP:                                       # quét SNR trên tập validation
        X_, y_ = build_eval(te, noise, wav_cache, val_noise, snr_db=snr)
        sweep[snr] = report(y_, predict(model, X_), "", verbose=False)
    rep = features(np.stack([fit_window(wav_cache[it[0]]) for it in tr[::3]]))
    path = convert_int8(model, rep, os.path.join(out_dir, f"{fname}_int8.tflite"))
    pi = predict_tflite(path, features(Xc))
    acc_q, fa_q = report(yc, pi, f"{title} - sạch (INT8 TFLite)")
    return sweep, acc_q, fa_q, path


def print_summary(results):
    print("\n######## TỔNG KẾT (mỗi ô: độ chính xác 6 lệnh / tỉ lệ báo nhầm) ########")
    print("tập val    | " + " | ".join(f"{'sạch' if s is None else f'SNR {s:d} dB':>13s}" for s in SWEEP) + " | INT8 sạch")
    for name, sw, aq, fq in results:
        print(f"{name:10s} | " + " | ".join(f"{sw[s][0] * 100:5.1f}%/{sw[s][1] * 100:4.1f}%" for s in SWEEP)
              + f" | {aq * 100:5.1f}%/{fq * 100:4.1f}%")


# ----------------------------- main -----------------------------
# ----------------------------- kiểm tra bộ nhớ ESP32 -----------------------------
class _Quiet:
    """Tắt stdout/stderr ở mức file descriptor (TFLM in lỗi từ C++ mỗi lần thử arena quá nhỏ)."""
    def __enter__(self):
        self.fds = [os.dup(1), os.dup(2)]
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 1); os.dup2(self.null, 2)

    def __exit__(self, *a):
        os.dup2(self.fds[0], 1); os.dup2(self.fds[1], 2)
        for f in self.fds + [self.null]:
            os.close(f)


def tflm_arena_min(path):
    """Arena nhỏ nhất (KB) mà TFLite Micro THẬT cấp phát và chạy được. None nếu chưa cài tflite-micro."""
    try:
        from tflite_micro.python.tflite_micro import runtime
    except Exception:
        return None
    data = open(path, "rb").read()

    def ok(n):
        try:
            with _Quiet():
                it = runtime.Interpreter.from_bytes(data, arena_size=int(n))
                it.set_input(np.zeros(it.get_input_details(0)["shape"], np.int8), 0)
                it.invoke()
            return True
        except Exception:
            return False
    hi = max(64 * 1024, 8 * len(data))
    while not ok(hi):
        hi *= 2
        if hi > 16 * 1024 * 1024:
            return None
    lo = 1024
    while hi - lo > 256:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi / 1024


_RESOLVER = {"ADD": "AddAdd", "CONV_2D": "AddConv2D", "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
             "FULLY_CONNECTED": "AddFullyConnected", "MEAN": "AddMean", "PAD": "AddPad",
             "AVERAGE_POOL_2D": "AddAveragePool2D", "RESHAPE": "AddReshape", "SOFTMAX": "AddSoftmax",
             "QUANTIZE": "AddQuantize", "DEQUANTIZE": "AddDequantize"}


def report_memory(path):
    size_kb = os.path.getsize(path) / 1024
    ops = []
    try:
        it = tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=
                                 tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        ops = sorted({o["op_name"] for o in it._get_ops_details()})
    except Exception as e:
        print(f"  (không đọc được danh sách toán tử: {e})")
    arena = tflm_arena_min(path)
    audio_kb, feat_kb = WIN * 2 / 1024, N_FRAMES * N_MEL / 1024
    print("\n######## BỘ NHỚ TRÊN ESP32 ########")
    print(f"  flash : file model {size_kb:.0f} KB / {FLASH_BUDGET_KB} KB (giả định)")
    if arena is None:
        print("  arena : chưa đo được (cài `pip install tflite-micro` rồi chạy lại để đo bằng TFLite Micro thật)")
    else:
        need = arena + audio_kb + feat_kb + OTHER_RAM_KB
        verdict = "KHÔNG VỪA" if need > RAM_BUDGET_KB else ("SÁT" if need > 0.8 * RAM_BUDGET_KB else "VỪA")
        print(f"  arena : {arena:.1f} KB (TFLite Micro thật)")
        print(f"  RAM   : {need:.0f} KB cần = arena {arena:.0f} + âm thanh 2 s int16 {audio_kb:.0f} + đặc trưng {feat_kb:.0f} "
              f"+ hệ thống {OTHER_RAM_KB} (giả định) -> {verdict} so với {RAM_BUDGET_KB} KB")
        print("  Số liệu trên chỉ là ước lượng: hãy đối chiếu bằng interpreter.arena_used_bytes() và heap_caps_get_free_size() trên board.")
    if ops:
        print("  Toán tử cần đăng ký trong MicroMutableOpResolver:")
        for o in ops:
            print(f"      resolver.{_RESOLVER.get(o, 'Add<' + o + '>')}();")


def resolve_data_dir(d):
    for c in [d] + DATA_DIR_FALLBACKS:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "dataset*")):
            if c != d:
                print(f"  (DATA_DIR '{d}' không có dataset*, dùng '{c}')")
            return c
    raise SystemExit(f"Không thấy dataset* trong {[d] + DATA_DIR_FALLBACKS}. Sửa DATA_DIR ở khối cấu hình.")


def main():
    # Mọi cờ dòng lệnh chỉ GHI ĐÈ giá trị mặc định ở khối cấu hình trên cùng; không truyền gì vẫn chạy được.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--mode", choices=["cv", "final"], default=MODE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snr-min", type=float, default=SNR_MIN, help="SNR thấp nhất khi trộn nhiễu lúc train (dB)")
    ap.add_argument("--snr-max", type=float, default=SNR_MAX, help="SNR cao nhất khi trộn nhiễu lúc train (dB)")
    ap.add_argument("--noise-mix", action=argparse.BooleanOptionalAction, default=USE_NOISE_MIX,
                    help="bật/tắt trộn nhiễu vào mẫu lệnh khi train")
    ap.add_argument("--p-noise", type=float, default=P_NOISE, help="xác suất trộn nhiễu cho mỗi mẫu")
    ap.add_argument("--babble-prob", type=float, default=BABBLE_PROB, help="tỉ lệ nhiễu là tiếng nhiều người nói chuyện (0 = tắt)")
    ap.add_argument("--extra-noise", action="append", help="thư mục wav nhiễu bổ sung (ghi đè EXTRA_NOISE_DIRS)")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC, help="tỉ lệ dữ liệu chừa làm validation khi --split-by random (0 = không chừa)")
    ap.add_argument("--split-by", choices=["speaker", "random", "dataset"], default=SPLIT_BY)
    ap.add_argument("--holdout-ds", nargs="+", default=HOLDOUT_DATASETS, help="dataset làm validation khi --split-by dataset")
    ap.add_argument("--retrain-full", action=argparse.BooleanOptionalAction, default=RETRAIN_FULL,
                    help="sau khi xong, train lại trên TOÀN BỘ dữ liệu để xuất model")
    a, _ = ap.parse_known_args()                             # bỏ qua đối số lạ (vd "-f kernel.json" do Colab/Jupyter tự thêm)
    a.extra_noise = a.extra_noise or EXTRA_NOISE_DIRS
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed); tf.random.set_seed(a.seed)

    items, noise = collect(resolve_data_dir(a.data))
    wav_cache = {p: read_wav(p) for p, *_ in items}
    noise_cache = {p: read_wav(p) for p, _ in noise}
    extra = load_extra_noise(a.extra_noise)
    print(f"{len(items)} clip + {len(noise)} file noise (+{len(extra)} nhiễu bổ sung); model {N_FRAMES}x{N_MEL}")
    print(f"trộn nhiễu khi train: {'BẬT' if a.noise_mix else 'TẮT'}"
          + (f" (p={a.p_noise}, SNR {a.snr_min}..{a.snr_max} dB, babble={a.babble_prob})" if a.noise_mix else ""))

    if a.mode == "cv":
        results = []
        for held in CV_HELD_OUT:
            print(f"\n######## validation = người nói '{held}', train trên phần còn lại ########")
            tr = [it for it in items if it[2] != held]
            te = [it for it in items if it[2] == held]
            val_noise = [noise_cache[p] for p, d in noise if d == held]
            pool = [noise_cache[p] for p, d in noise if d != held] + extra
            val = make_val_features(te, noise, wav_cache, val_noise)
            model = train(tr, pool, wav_cache, a.epochs, rng, a, val=val)
            if held == CV_HELD_OUT[0]:
                model.summary(print_fn=lambda s: print(s) if "Total params" in s else None)
            sweep, aq, fq, _ = evaluate_model(model, tr, te, noise, val_noise, wav_cache, a.out,
                                              f"cv_{held}", held)
            results.append((held, sweep, aq, fq))
        print_summary(results)
        return

    # ---- mode final: chừa tập validation, train phần còn lại, theo dõi val_acc, rồi xuất model ----
    tr, va, tr_noise, va_noise = split_holdout(items, noise, a.val_frac, a.split_by, a.holdout_ds, SPLIT_SEED)
    print(f"\nchia dữ liệu ({a.split_by}): train {len(tr)} clip + {len(tr_noise)} noise | "
          f"val {len(va)} clip + {len(va_noise)} noise")
    if not tr_noise and not extra:
        raise SystemExit("Tập train không còn file noise nào. Đổi cách chia hoặc thêm EXTRA_NOISE_DIRS.")
    val, va_noise_wavs = None, []
    if va:
        with open(os.path.join(a.out, "val_split.json"), "w", encoding="utf-8") as f:
            json.dump({"val_speakers": sorted({it[4] for it in va}), "val_clips": [it[0] for it in va], "val_noise": [p for p, _ in va_noise]},
                      f, ensure_ascii=False, indent=1)
        va_noise_wavs = [noise_cache[p] for p, _ in va_noise]
        if not va_noise_wavs:
            print("  (cảnh báo: không có file noise nào trong tập val -> lớp background không được đánh giá)")
        val = make_val_features(va, noise, wav_cache, va_noise_wavs)

    pool = [noise_cache[p] for p, _ in tr_noise] + extra
    model = train(tr, pool, wav_cache, a.epochs, rng, a, val=val)
    if va:
        sweep, aq, fq, path = evaluate_model(model, tr, va, noise, va_noise_wavs, wav_cache, a.out, "kws", "VAL")
        print_summary([("val", sweep, aq, fq)])
    else:
        rep = features(np.stack([fit_window(wav_cache[it[0]]) for it in tr[::2]]))
        path = convert_int8(model, rep, os.path.join(a.out, "kws_int8.tflite"))

    if va and a.retrain_full:                                # train lại trên toàn bộ dữ liệu để xuất model cuối
        print("\n######## train lại trên TOÀN BỘ dữ liệu để xuất model ########")
        full_pool = [noise_cache[p] for p, _ in noise] + extra
        model = train(items, full_pool, wav_cache, a.epochs, rng, a)
        rep = features(np.stack([fit_window(wav_cache[it[0]]) for it in items[::2]]))
        path = convert_int8(model, rep, os.path.join(a.out, "kws_int8.tflite"))
    model.save(os.path.join(a.out, "kws_model.keras"))
    export_c(path, a.out)
    if CHECK_MEMORY:
        report_memory(path)


if __name__ == "__main__":
    main()