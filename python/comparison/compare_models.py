#!/usr/bin/env python3
"""
Train và SO SÁNH nhiều kiến trúc KWS có thể chạy trên ESP32 (TFLite Micro, int8).

Dùng lại toàn bộ pipeline của train_kws.py (dữ liệu, đặc trưng log-mel 99x40, tăng cường, trộn nhiễu,
chuyển int8) -> chỉ khác kiến trúc, nên so sánh công bằng. Đặt file này CẠNH train_kws.py.

Colab: upload train_kws.py + compare_models.py vào /content, (tùy chọn) !pip install tflite-micro, rồi
    %run compare_models.py --smoke      (hoặc không có --smoke để chạy thật)

Chạy (không cần tham số - mọi thứ chỉnh ở khối CẤU HÌNH bên dưới):
    python compare_models.py
Cờ dòng lệnh chỉ ghi đè mặc định (tùy chọn):
    python compare_models.py --smoke                 # chạy thử 2 epoch để kiểm tra mọi model build/convert được
    python compare_models.py --models ds_cnn_s bc_resnet_1 --epochs 80

Kết quả (trong OUT_DIR):
    results.csv           mỗi dòng = 1 lần train (model x fold x seed); có RESUME nên chạy lại sẽ bỏ qua dòng đã xong
    summary.md            bảng tổng kết trung bình theo model (dán thẳng vào báo cáo)
    compare.png           biểu đồ độ chính xác int8 theo MACs và theo kích thước model
    models/*.tflite       model int8 của từng lần train
    export/<model>/       (EXPORT_BEST=True) kws_model_data.cc/.h + kws_config.h của model tốt nhất
"""
import argparse, csv, glob, os, time, types
import numpy as np
import tensorflow as tf


# ================================ CẤU HÌNH CHẠY (chỉnh ở đây) ================================
# --- nơi tìm train_kws.py (bắt buộc phải có; trên Colab hãy upload nó vào /content) ---
TRAIN_KWS_DIRS = [".", "/content", "/content/AI-training-voice", "/content/drive/MyDrive/AI-training-voice",
                  "/content/drive/MyDrive"]

# --- dữ liệu ---
DATA_DIR = "data/data"                   # thư mục chứa dataset2/3/4; nếu sai, tự thử các đường dẫn trong DATA_DIR_FALLBACKS
DATA_DIR_FALLBACKS = ["data", "../data", "/content/data/data", "/content/data"]
EXTRA_NOISE_DIRS = []                    # thư mục wav nhiễu bổ sung, vd ["extra_noise"]
OUT_DIR = "compare_out"

# --- chọn model so sánh (xem MODEL_ZOO bên dưới) ---
MODELS = ["tiny_conv", "ds_cnn_s", "ds_cnn_m", "tc_resnet8", "tc_resnet14",
          "bc_resnet_1", "bc_resnet_3", "matchboxnet_3x2x64"]

# --- cách đánh giá ---
SPLIT_MODE = "cv"                        # "cv": chừa nguyên dataset (CV_HELD_OUT) làm val, mỗi dataset một fold | "speaker": 1 fold chia theo giọng
CV_HELD_OUT = ["dataset3", "dataset4"]   # giọng thật chưa từng thấy; nên giữ cả 2 để số liệu ít nhiễu
SEEDS = [0]                              # thêm seed (vd [0, 1, 2]) để lấy trung bình; tốn thời gian x số seed

# --- huấn luyện (mọi model dùng chung để công bằng) ---
EPOCHS = 100                              # train_kws mặc định 150; 60 đủ để xếp hạng, tăng lên khi chốt model
MODEL_EPOCHS = {}                        # ghi đè riêng từng model, vd {"bc_resnet_3": 100}
LR = 3e-3
MODEL_LR = {}                            # ghi đè LR riêng, vd {"matchboxnet_3x2x64": 1e-3}
BATCH_SIZE = 32
VAL_EVERY = 10                           # in val_acc mỗi N epoch
SNR_MIN, SNR_MAX = -5.0, 20.0            # dB, trộn nhiễu khi train
USE_NOISE_MIX = True
P_NOISE = 0.85
BABBLE_PROB = 0.3

# --- tùy chọn kiến trúc ---
USE_DILATION = False                     # áp cho bc_resnet_*/matchboxnet_* (không hậu tố). Dilation làm converter sinh SPACE_TO_BATCH_ND/BATCH_TO_SPACE_ND và KHÔNG gộp được BatchNorm -> to và chậm hơn;
                                         # muốn so sánh thì thêm các biến thể "*_dil" trong MODEL_ZOO vào MODELS
DROPOUT = 0.2                            # dropout trước lớp phân loại (tiny_conv/ds_cnn_m/tc_resnet/matchboxnet)

# --- ngân sách bộ nhớ ESP32: SỬA theo board/firmware của bạn. Các số dưới là GIẢ ĐỊNH, không phải số đo ---
MEASURE_TFLM = True                      # đo arena bằng TFLite Micro thật (cần: pip install tflite-micro); False/không cài -> dùng cận dưới từ TFLite
TFLM_MATCH_SAMPLES = 64                  # số mẫu val so kết quả TFLM với TFLite để chứng minh model chạy đúng trên TFLM
RAM_BUDGET_KB = 320                      # DRAM nội của ESP32 (tài liệu ESP-IDF: 320 KB DRAM + 200 KB IRAM); ESP32-S3 dùng 512 KB tổng, tự điền
STATIC_DRAM_LIMIT_KB = 160               # ESP32: tổng DRAM cấp phát TĨNH tối đa 160 KB; arena lớn hơn phải cấp phát bằng heap
OTHER_RAM_KB = 80                        # FreeRTOS, stack, I2S DMA, OLED, PWM... (+ WiFi/BT nếu dùng). Hãy ĐO trong firmware trống: heap_caps_get_free_size(MALLOC_CAP_INTERNAL)
FLASH_BUDGET_KB = 1000                   # flash dành cho model (phụ thuộc bảng phân vùng); model + code phải vừa phân vùng app
USE_PSRAM = False                        # True: đặt arena ở PSRAM (chậm hơn nhưng không chiếm DRAM nội)
PSRAM_KB = 0                             # dung lượng PSRAM khả dụng khi USE_PSRAM = True (vd 2048)
TIGHT_RATIO = 0.8                        # dùng quá tỉ lệ này của ngân sách -> báo "SÁT"

# --- đo đạc / báo cáo ---
SORT_BY = "avg_q_acc"                    # cột dùng để xếp hạng: avg_q_acc (TB độ chính xác int8 qua các mức SNR) | q_acc | q_acc_snr0 | q_acc_snr10 ...
ESP32_MACS_PER_MS = None                 # tùy chọn: hiệu chỉnh bằng 1 phép đo thật (MACs / ms của ESP32 bạn) -> in thêm cột ước lượng ms
RESUME = True                            # đọc results.csv có sẵn, bỏ qua tổ hợp (model, fold, seed) đã xong
EXPORT_BEST = True                       # xuất C array của model tốt nhất (lấy từ fold đầu, seed đầu)
EXPORT_MODELS = []                       # ép xuất thêm các model chỉ định, vd ["ds_cnn_s", "bc_resnet_1"]
MAKE_PLOT = True
SMOKE_TEST = False                       # True: 2 epoch, 1 seed -> chỉ để kiểm tra pipeline
# Toán tử mà TFLite Micro thường có sẵn; model dùng toán tử ngoài danh sách này sẽ bị cảnh báo
TFLM_OPS_OK = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "ADD", "MEAN", "AVERAGE_POOL_2D", "MAX_POOL_2D",
               "RESHAPE", "SOFTMAX", "QUANTIZE", "DEQUANTIZE", "PAD", "MUL", "RELU", "RELU6", "LOGISTIC",
               "STRIDED_SLICE", "CONCATENATION", "EXPAND_DIMS", "SQUEEZE", "TRANSPOSE", "SUB",
               "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"}      # 2 toán tử cuối do conv dilation sinh ra: có hỗ trợ nhưng chậm
# ==============================================================================================

import importlib, sys


def _import_train_kws():
    """Tìm train_kws.py trong TRAIN_KWS_DIRS (và thư mục chứa file này nếu có) rồi import."""
    dirs = list(TRAIN_KWS_DIRS)
    if "__file__" in globals():                          # notebook/Colab không có __file__
        dirs.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for d in dirs:
        if os.path.isfile(os.path.join(d, "train_kws.py")):
            sys.path.insert(0, os.path.abspath(d))
            print(f"  (dùng train_kws.py ở {os.path.abspath(d)})")
            if "train_kws" in sys.modules:               # chạy lại trong cùng kernel: K.build_model đã bị vá ở lần chạy trước
                return importlib.reload(sys.modules["train_kws"])
            return importlib.import_module("train_kws")
    raise SystemExit("Không tìm thấy train_kws.py. Script này dùng lại pipeline trong train_kws.py của dự án.\n"
                     "  - Colab: upload train_kws.py vào /content (panel Files bên trái, hoặc chạy trong một ô:\n"
                     "      from google.colab import files; files.upload()   # chọn train_kws.py)\n"
                     "  - hoặc thêm thư mục chứa nó vào TRAIN_KWS_DIRS ở khối cấu hình đầu file.\n"
                     f"  Đã tìm trong: {dirs}")


K = _import_train_kws()          # dùng lại pipeline dữ liệu / đặc trưng / chuyển int8 / xuất C

try:                             # pip install tflite-micro -> chạy đúng bộ kernel TFLite Micro trên PC để đo arena thật
    from tflite_micro.python.tflite_micro import runtime as _tflm
except Exception:
    _tflm = None

L = tf.keras.layers
_DS_CNN_S_ORIG = K.build_model           # DS-CNN hiện tại của dự án (giữ lại trước khi vá K.build_model)


# ----------------------------- khối dựng chung -----------------------------
def _bn_act(x, act=True):
    x = L.BatchNormalization()(x)
    return L.ReLU()(x) if act else x


def _conv(x, ch, k, s=1, d=1, act=True):
    x = L.Conv2D(ch, k, strides=s, dilation_rate=d, padding="same", use_bias=False)(x)
    return _bn_act(x, act)


def _dw(x, k, s=1, d=1, act=True):
    x = L.DepthwiseConv2D(k, strides=s, dilation_rate=d, padding="same", use_bias=False)(x)
    return _bn_act(x, act)


def _head(x, drop=DROPOUT):
    x = L.GlobalAveragePooling2D()(x)
    if drop:
        x = L.Dropout(drop)(x)
    return L.Dense(len(K.CLASSES))(x)     # logits (softmax khi suy luận)


def _input():
    return L.Input((K.N_FRAMES, K.N_MEL, 1))


def _mel_as_channels(inp, ch, k_time, stride_time=1):
    """Tương đương coi 40 băng mel là KÊNH rồi conv 1D theo thời gian (cách TC-ResNet / MatchboxNet làm):
    (T, F, 1) --Conv2D kernel (k_time, F), 'valid' theo tần số--> (T', 1, ch). Không dùng Reshape vì Keras sinh
    toán tử SHAPE/PACK (reshape động) trong TFLite."""
    pad = (k_time - 1) // 2
    x = L.ZeroPadding2D(((pad, k_time - 1 - pad), (0, 0)))(inp)
    x = L.Conv2D(ch, (k_time, K.N_MEL), strides=(stride_time, 1), padding="valid", use_bias=False)(x)
    return _bn_act(x)


# ----------------------------- các kiến trúc -----------------------------
def build_tiny_conv():
    """Baseline nhỏ nhất kiểu 'micro_speech tiny_conv': 1 conv + 1 Dense. Làm mốc sàn."""
    inp = _input()
    x = L.Conv2D(8, (10, 8), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = _bn_act(x)
    x = L.Dropout(DROPOUT)(x)
    x = L.Conv2D(len(K.CLASSES), tuple(x.shape[1:3]), padding="valid")(x)    # = Flatten + Dense, không sinh SHAPE/PACK
    return tf.keras.Model(inp, L.GlobalAveragePooling2D()(x), name="tiny_conv")


def build_ds_cnn_s():
    """DS-CNN hiện đang dùng trong train_kws.py (32 -> 48/64/64/64)."""
    m = _DS_CNN_S_ORIG()
    m._name = "ds_cnn_s"
    return m


def build_ds_cnn_m():
    """DS-CNN rộng/sâu hơn bản hiện tại (khoảng x4 tham số)."""
    inp = _input()
    x = _conv(inp, 64, (10, 4), s=2)
    for ch, st in [(64, 1), (96, 2), (128, 1), (128, 1)]:
        x = _dw(x, 3, s=st)
        x = _conv(x, ch, 1)
    return tf.keras.Model(inp, _head(x), name="ds_cnn_m")


def _tc_resnet(channels, name):
    """TC-ResNet (Choi et al., 2019): conv theo trục thời gian, kernel 9, mỗi khối stride 2 + shortcut 1x1."""
    inp = _input()
    x = _mel_as_channels(inp, channels[0], 3)
    for ch in channels[1:]:
        sc = _conv(x, ch, 1, s=(2, 1), act=False)
        y = _conv(x, ch, (9, 1), s=(2, 1))
        y = _conv(y, ch, (9, 1), act=False)
        x = L.ReLU()(L.Add()([sc, y]))
    return tf.keras.Model(inp, _head(x), name=name)


def build_tc_resnet8():
    return _tc_resnet([16, 24, 32, 48], "tc_resnet8")


def build_tc_resnet14():
    return _tc_resnet([16, 24, 24, 32, 32, 48, 48], "tc_resnet14")


def _bc_block(x, out_ch, freq_stride, dilation, transition):
    """Khối BC-ResNet (Kim et al., 2021). Trục: (thời gian, tần số). f2 = conv theo tần số; f1 = conv theo thời gian
    trên bản đã lấy trung bình tần số rồi cộng 'broadcast'. Sub-spectral norm được thay bằng BatchNorm thường."""
    inp = x
    if transition:
        x = _conv(x, out_ch, 1)
    f2 = _dw(x, (1, 3), s=(1, freq_stride), act=False)
    w = f2.shape[2]
    a = L.AveragePooling2D((1, w))(f2)                                             # (B, T, 1, C)
    a = _dw(a, (3, 1), d=(dilation, 1), act=True)
    a = L.Conv2D(out_ch, 1)(a)
    a = L.Dropout(0.1)(a)
    y = L.Add()([f2, a])                                                           # broadcast theo trục tần số
    if not transition:
        y = L.Add()([inp, y])
    return L.ReLU()(y)


def _bc_resnet(scale, name, dilate):
    c = [int(8 * scale), int(12 * scale), int(16 * scale), int(20 * scale)]
    n, fstride, dil = [2, 2, 4, 4], [1, 2, 2, 1], [1, 2, 4, 8]
    inp = _input()
    x = L.Conv2D(int(16 * scale), 5, strides=(1, 2), padding="same", use_bias=False)(inp)   # stride theo tần số
    x = _bn_act(x)
    cin = int(16 * scale)
    for s in range(4):
        for b in range(n[s]):
            x = _bc_block(x, c[s], fstride[s] if b == 0 else 1, dil[s] if dilate else 1,
                          transition=(b == 0 and cin != c[s]))
            cin = c[s]
    x = _dw(x, 5)
    x = _conv(x, int(32 * scale), 1)
    return tf.keras.Model(inp, _head(x, drop=0.0), name=name)


def build_bc_resnet_1():
    return _bc_resnet(1, "bc_resnet_1", USE_DILATION)


def build_bc_resnet_3():
    return _bc_resnet(3, "bc_resnet_3", USE_DILATION)


def _matchboxnet(name, dilate):
    """MatchboxNet-3x2x64 (Majumdar & Ginsburg, 2020) viết bằng Conv2D 'rộng 1' để TFLM chạy được."""
    inp = _input()
    x = _mel_as_channels(inp, 64, 11, stride_time=2)                               # prologue
    for k in (13, 15, 17):                                                         # 3 khối, mỗi khối 2 sub-block tách kênh
        sc = _conv(x, 64, 1, act=False)
        y = x
        for r in range(2):
            y = _dw(y, (k, 1), act=False)
            y = L.Conv2D(64, 1, use_bias=False)(y)
            y = _bn_act(y, act=(r == 0))
        x = L.ReLU()(L.Add()([sc, y]))
    x = _dw(x, (29, 1), d=((2 if dilate else 1), 1), act=False)                    # epilogue
    x = _conv(x, 128, 1)
    return tf.keras.Model(inp, _head(x), name=name)


def build_matchboxnet_3x2x64():
    return _matchboxnet("matchboxnet_3x2x64", USE_DILATION)


MODEL_ZOO = {
    "tiny_conv": build_tiny_conv,
    "ds_cnn_s": build_ds_cnn_s,
    "ds_cnn_m": build_ds_cnn_m,
    "tc_resnet8": build_tc_resnet8,
    "tc_resnet14": build_tc_resnet14,
    "bc_resnet_1": build_bc_resnet_1,
    "bc_resnet_3": build_bc_resnet_3,
    "matchboxnet_3x2x64": build_matchboxnet_3x2x64,
    # biến thể có dilation giống bài báo gốc (không nằm trong MODELS mặc định)
    "bc_resnet_1_dil": lambda: _bc_resnet(1, "bc_resnet_1_dil", True),
    "bc_resnet_3_dil": lambda: _bc_resnet(3, "bc_resnet_3_dil", True),
    "matchboxnet_3x2x64_dil": lambda: _matchboxnet("matchboxnet_3x2x64_dil", True),
}


# ----------------------------- đo kích thước / tính toán -----------------------------
def count_macs(model):
    """Số phép nhân-cộng (MACs) của Conv2D / DepthwiseConv2D / Dense (BN, ReLU, Add... bỏ qua vì không đáng kể)."""
    total = 0
    for ly in model.layers:
        if isinstance(ly, L.DepthwiseConv2D):
            _, h, w, _ = ly.output.shape
            kh, kw = ly.kernel_size
            total += h * w * kh * kw * ly.input.shape[-1] * ly.depth_multiplier
        elif isinstance(ly, L.Conv2D):
            _, h, w, co = ly.output.shape
            kh, kw = ly.kernel_size
            total += h * w * kh * kw * ly.input.shape[-1] * co // ly.groups
        elif isinstance(ly, L.Dense):
            total += int(ly.input.shape[-1]) * int(ly.units)
    return int(total)


def tflite_stats(path):
    """(kích thước file KB, trọng số KB, ước lượng arena KB, tập toán tử).
    - file KB gồm cả tên tensor/metadata (model nhiều lớp bị phình), còn trọng số KB chỉ là hằng số thật sự dùng.
    - Arena = cận dưới: tổng kích thước các tensor kích hoạt đang sống cùng lúc (đỉnh); TFLM thực tế cần nhiều hơn
      một chút do căn lề và scratch buffer của kernel."""
    size_kb = os.path.getsize(path) / 1024
    try:
        it = tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=
                                 tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)   # không gắn XNNPACK -> thấy đúng toán tử
        it.allocate_tensors()
        ops = it._get_ops_details()
        td = {t["index"]: t for t in it.get_tensor_details()}
        inp = it.get_input_details()[0]["index"]
        birth, death = {inp: -1}, {}
        for i, op in enumerate(ops):
            for o in op["outputs"]:
                birth[o] = i
            for t in op["inputs"]:
                death[t] = i
        for o in it.get_output_details():
            death[o["index"]] = len(ops)
        nbytes = lambda t: int(np.prod(td[t]["shape"])) * np.dtype(td[t]["dtype"]).itemsize
        peak = 0
        for i in range(len(ops)):
            live = sum(nbytes(t) for t, b in birth.items() if b <= i <= death.get(t, b))
            peak = max(peak, live)
        consts = {t for op in ops for t in op["inputs"] if t >= 0 and t not in birth}      # đầu vào của toán tử mà không do toán tử nào sinh ra
        weights = sum(nbytes(t) for t in consts)
        return size_kb, weights / 1024, peak / 1024, sorted({op["op_name"] for op in ops})
    except Exception as e:
        print(f"  (không đo được arena/toán tử: {e})")
        return size_kb, float("nan"), float("nan"), []


# ----------------------------- đo bằng TFLite Micro thật + ngân sách ESP32 -----------------------------
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
    """Arena nhỏ nhất (KB) mà TFLite Micro THẬT cấp phát và chạy được (tìm nhị phân). NaN nếu không đo được.
    Đây là bộ nhớ tensor_arena thực tế, đã gồm scratch buffer và căn lề; ESP32 dùng kernel khác (ESP-NN) nên
    có thể lệch nhẹ, nhưng sát hơn nhiều so với ước lượng từ tensor kích hoạt."""
    if not (MEASURE_TFLM and _tflm):
        return float("nan")
    data = open(path, "rb").read()

    def ok(n):
        try:
            with _Quiet():
                it = _tflm.Interpreter.from_bytes(data, arena_size=int(n))
                it.set_input(np.zeros(it.get_input_details(0)["shape"], np.int8), 0)
                it.invoke()
            return True
        except Exception:
            return False
    hi = max(64 * 1024, 8 * len(data))
    while not ok(hi):
        hi *= 2
        if hi > 16 * 1024 * 1024:
            return float("nan")
    lo = 1024
    while hi - lo > 256:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi / 1024


def tflm_agreement(path, F, arena_kb):
    """Tỉ lệ mẫu mà TFLM và TFLite cho cùng nhãn (đặc trưng F đã chuẩn hóa, float). ~1.0 = model chạy đúng trên TFLM."""
    if not (MEASURE_TFLM and _tflm) or np.isnan(arena_kb):
        return float("nan")
    F = F[:TFLM_MATCH_SAMPLES]
    ref = K.predict_tflite(path, F)
    it_ref = tf.lite.Interpreter(model_path=path)
    s, z = it_ref.get_input_details()[0]["quantization"]
    with _Quiet():
        it = _tflm.Interpreter.from_bytes(open(path, "rb").read(), arena_size=int(arena_kb * 1024) + 1024)
        out = []
        for f in F:
            it.set_input(np.clip(np.round(f / s + z), -128, 127).astype(np.int8)[None], 0)
            it.invoke()
            out.append(int(np.argmax(it.get_output(0))))
    return float(np.mean(np.array(out) == ref))


def fit_verdict(arena_kb, file_kb):
    """(RAM cần KB, kết luận). RAM cần = arena + bộ đệm âm thanh 2 s int16 + đặc trưng int8 + phần hệ thống OTHER_RAM_KB."""
    audio_kb = K.WIN * 2 / 1024
    feat_kb = K.N_FRAMES * K.N_MEL / 1024
    if USE_PSRAM:
        internal = audio_kb + feat_kb + OTHER_RAM_KB
        need, bad_arena = internal, arena_kb > PSRAM_KB
    else:
        need, bad_arena = arena_kb + audio_kb + feat_kb + OTHER_RAM_KB, False
    notes = []
    if file_kb > FLASH_BUDGET_KB or bad_arena or need > RAM_BUDGET_KB:
        verdict = "KHÔNG VỪA"
    elif need > TIGHT_RATIO * RAM_BUDGET_KB or file_kb > TIGHT_RATIO * FLASH_BUDGET_KB:
        verdict = "SÁT"
    else:
        verdict = "VỪA"
    if not USE_PSRAM and arena_kb > STATIC_DRAM_LIMIT_KB:
        notes.append("arena > giới hạn DRAM tĩnh, phải cấp phát bằng heap")
    return need, verdict + (f" ({'; '.join(notes)})" if notes else "")


def check_files(paths):
    """Chế độ --check: báo cáo bộ nhớ cho file .tflite bất kỳ, không cần dữ liệu hay train."""
    print(f"Ngân sách: RAM {RAM_BUDGET_KB} KB (hệ thống/khác {OTHER_RAM_KB} KB, âm thanh {K.WIN * 2 / 1024:.0f} KB, "
          f"đặc trưng {K.N_FRAMES * K.N_MEL / 1024:.1f} KB), flash {FLASH_BUDGET_KB} KB, PSRAM {'có ' + str(PSRAM_KB) + ' KB' if USE_PSRAM else 'không'}")
    if not _tflm:
        print("(chưa cài tflite-micro: dùng cận dưới từ TFLite. Cài: pip install tflite-micro)")
    for p in paths:
        size_kb, w_kb, lb_kb, ops = tflite_stats(p)
        real = tflm_arena_min(p)
        arena = real if not np.isnan(real) else lb_kb
        need, verdict = fit_verdict(arena, size_kb)
        print(f"\n{os.path.basename(p)}\n  file {size_kb:.0f} KB (trọng số {w_kb:.0f} KB) | arena "
              f"{'TFLM ' + format(real, '.1f') if not np.isnan(real) else 'cận dưới ' + format(lb_kb, '.1f')} KB | "
              f"RAM cần {need:.0f}/{RAM_BUDGET_KB} KB | flash {size_kb:.0f}/{FLASH_BUDGET_KB} KB -> {verdict}\n  toán tử: {' '.join(ops)}")


# ----------------------------- dữ liệu / fold -----------------------------
def resolve_data_dir():
    for d in [DATA_DIR] + DATA_DIR_FALLBACKS:
        if os.path.isdir(d) and glob.glob(os.path.join(d, "dataset*")):
            if d != DATA_DIR:
                print(f"  (DATA_DIR '{DATA_DIR}' không có dataset*, dùng '{d}')")
            return d
    raise SystemExit(f"Không thấy dataset* trong {[DATA_DIR] + DATA_DIR_FALLBACKS}. Sửa DATA_DIR ở khối cấu hình.")


def make_folds(items, noise, noise_cache, extra):
    """Trả về list (tên fold, tập train, tập val, pool noise train, noise wav của val, file noise val)."""
    folds = []
    if SPLIT_MODE == "cv":
        for held in CV_HELD_OUT:
            tr = [it for it in items if it[2] != held]
            te = [it for it in items if it[2] == held]
            if not te:
                print(f"  (bỏ qua fold '{held}': không có dữ liệu)")
                continue
            val_noise_wavs = [noise_cache[p] for p, d in noise if d == held]
            pool = [noise_cache[p] for p, d in noise if d != held] + extra
            folds.append((held, tr, te, pool, val_noise_wavs))
    else:
        tr, te, tr_n, va_n = K.split_holdout(items, noise, K.VAL_FRAC, "speaker", [], K.SPLIT_SEED)
        pool = [noise_cache[p] for p, _ in tr_n] + extra
        folds.append(("speaker", tr, te, pool, [noise_cache[p] for p, _ in va_n]))
    if not folds:
        raise SystemExit("Không dựng được fold nào.")
    return folds


# ----------------------------- huấn luyện + đánh giá một model -----------------------------
def run_one(name, fold, seed, items, noise, wav_cache, epochs, cfg):
    fname, tr, te, pool, val_noise = fold
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    K.build_model = MODEL_ZOO[name]                      # K.train() gọi build_model() ở phạm vi module K
    K.LR = MODEL_LR.get(name, LR)
    K.VAL_EVERY = VAL_EVERY
    val = K.make_val_features(te, noise, wav_cache, val_noise)

    t0 = time.time()
    model = K.train(tr, pool, wav_cache, epochs, rng, cfg, bs=BATCH_SIZE, val=val)
    train_sec = time.time() - t0

    # chuyển int8
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
    K.SAVEDMODEL_DIR = os.path.join(OUT_DIR, "_saved_model")
    rep = K.features(np.stack([K.fit_window(wav_cache[it[0]]) for it in tr[::3]]))
    path = K.convert_int8(model, rep, os.path.join(OUT_DIR, "models", f"{name}_{fname}_s{seed}.tflite"))
    size_kb, weights_kb, arena_kb, ops = tflite_stats(path)
    tflm_kb = tflm_arena_min(path)
    best_arena = tflm_kb if not np.isnan(tflm_kb) else arena_kb
    ram_need, verdict = fit_verdict(best_arena, size_kb)
    bad = sorted(set(ops) - TFLM_OPS_OK)
    if bad:
        print(f"  CẢNH BÁO: toán tử ngoài danh sách TFLM thường có: {bad}")
    if "MUL" in ops:
        print("  CẢNH BÁO: còn toán tử MUL = BatchNorm chưa gộp vào conv (model to/chậm hơn cần thiết)")

    # đánh giá float + int8 trên cùng đặc trưng, quét SNR
    row = {"model": name, "fold": fname, "seed": seed, "epochs": epochs, "params": model.count_params(),
           "macs_M": round(count_macs(model) / 1e6, 3), "tflite_KB": round(size_kb, 1),
           "weights_KB": round(weights_kb, 1), "arena_KB": round(arena_kb, 1),
           "tflm_arena_KB": round(tflm_kb, 1), "ram_need_KB": round(ram_need, 1), "fit": verdict, "ops": " ".join(ops), "train_sec": round(train_sec)}
    for snr in K.SWEEP:
        tag = "" if snr is None else f"_snr{snr}"
        X, y = K.build_eval(te, noise, wav_cache, val_noise, snr_db=snr)
        F = K.features(X)
        fa_, ff_ = K.report(y, model.predict(F, verbose=0).argmax(1), "", verbose=False)
        qa_, qf_ = K.report(y, K.predict_tflite(path, F), "", verbose=False)
        if snr is None:
            row["tflm_match"] = round(tflm_agreement(path, F, tflm_kb), 4)
        row[f"f_acc{tag}"], row[f"f_fa{tag}"] = round(float(fa_), 4), round(float(ff_), 4)
        row[f"q_acc{tag}"], row[f"q_fa{tag}"] = round(float(qa_), 4), round(float(qf_), 4)
    row["avg_q_acc"] = round(float(np.mean([row["q_acc" + ("" if s is None else f"_snr{s}")] for s in K.SWEEP])), 4)
    row["q_drop"] = round(row["f_acc"] - row["q_acc"], 4)
    row["tflite_path"] = path
    print(f"  -> {name}/{fname}/s{seed}: int8 sạch {row['q_acc'] * 100:.1f}% (báo nhầm {row['q_fa'] * 100:.1f}%), "
          f"TB các SNR {row['avg_q_acc'] * 100:.1f}%, {row['params']:,} tham số, {row['macs_M']:.2f}M MACs, "
          f"file {row['tflite_KB']:.0f} KB (trọng số {row['weights_KB']:.0f} KB), arena {'TFLM ' + format(row['tflm_arena_KB'], '.0f') if not np.isnan(tflm_kb) else '>= ' + format(row['arena_KB'], '.0f')} KB, {verdict}, {train_sec / 60:.1f} phút", flush=True)
    return row


# ----------------------------- CSV / tổng kết -----------------------------
def load_rows(path):
    if not (RESUME and os.path.exists(path)):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    """Trung bình theo model qua mọi fold/seed."""
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    out = []
    def num(rs, k):
        v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
        return float(np.mean(v)) if v else float("nan")
    for m, rs in by.items():
        d = {"model": m, "runs": len(rs)}
        for k in ("params", "macs_M", "tflite_KB", "weights_KB", "arena_KB", "tflm_arena_KB", "ram_need_KB", "tflm_match", "q_acc", "q_fa", "q_acc_snr20", "q_acc_snr10",
                  "q_acc_snr5", "q_acc_snr0", "avg_q_acc", "q_drop", "train_sec"):
            d[k] = num(rs, k)
        d["arena_best"] = d["tflm_arena_KB"] if not np.isnan(d["tflm_arena_KB"]) else d["arena_KB"]
        d["ram_need"], d["fit"] = fit_verdict(d["arena_best"], d["tflite_KB"])
        if np.isnan(d["tflm_arena_KB"]):
            d["fit"] += " [arena chưa đo thật]"
        out.append(d)
    key = SORT_BY if SORT_BY in out[0] else "avg_q_acc"
    out = [d for d in out if not np.isnan(d[key])] or out
    return sorted(out, key=lambda d: -d[key])


def render_table(agg, markdown=False):
    measured = all(not np.isnan(d["tflm_arena_KB"]) for d in agg)
    hdr = ["model", "tham số", "MACs(M)", "file KB", "trọng số KB", "arena KB" if measured else "arena>= KB", "RAM cần KB", "int8 sạch", "báo nhầm", "SNR20", "SNR10", "SNR5",
           "SNR0", "TB SNR", "float→int8", "vừa ESP32?"]
    if ESP32_MACS_PER_MS:
        hdr.append("~ms ESP32")
    rows = []
    for d in agg:
        r = [d["model"], f"{d['params']:,.0f}", f"{d['macs_M']:.2f}", f"{d['tflite_KB']:.0f}", f"{d['weights_KB']:.0f}", f"{d['arena_best']:.0f}", f"{d['ram_need']:.0f}",
             f"{d['q_acc'] * 100:.1f}%", f"{d['q_fa'] * 100:.1f}%", f"{d['q_acc_snr20'] * 100:.1f}%",
             f"{d['q_acc_snr10'] * 100:.1f}%", f"{d['q_acc_snr5'] * 100:.1f}%", f"{d['q_acc_snr0'] * 100:.1f}%",
             f"{d['avg_q_acc'] * 100:.1f}%", f"{d['q_drop'] * 100:+.1f}%", d["fit"]]
        if ESP32_MACS_PER_MS:
            r.append(f"{d['macs_M'] * 1e6 / ESP32_MACS_PER_MS:.0f}")
        rows.append(r)
    if markdown:
        return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows])
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    line = lambda r: " | ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(hdr), "-+-".join("-" * x for x in w)] + [line(r) for r in rows])


def make_plot(agg, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(bỏ qua biểu đồ: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, key, xl in ((ax[0], "macs_M", "MACs mỗi lần suy luận (triệu)"), (ax[1], "weights_KB", "trọng số int8 (KB)")):
        for d in agg:
            a.scatter(d[key], d["avg_q_acc"] * 100, s=60)
            a.annotate(d["model"], (d[key], d["avg_q_acc"] * 100), textcoords="offset points", xytext=(5, 4), fontsize=8)
        a.set_xscale("log"); a.set_xlabel(xl); a.set_ylabel("độ chính xác 6 lệnh, TB các SNR, int8 (%)"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  đã ghi {path}")


def remeasure():
    """Đo lại arena bằng TFLM thật cho các dòng results.csv chưa có (dùng file .tflite đã lưu), cập nhật cột RAM/kết luận."""
    if not _tflm:
        raise SystemExit("Chưa cài tflite-micro. Chạy: pip install tflite-micro")
    csv_path = os.path.join(OUT_DIR, "results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Không thấy {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = 0
    for r in rows:
        if str(r.get("tflm_arena_KB", "")).strip().lower() in ("", "nan") and os.path.exists(r.get("tflite_path", "")):
            kb = tflm_arena_min(r["tflite_path"])
            if not np.isnan(kb):
                r["tflm_arena_KB"] = round(kb, 1)
                r["ram_need_KB"], r["fit"] = (round(v, 1) if isinstance(v, float) else v
                                              for v in fit_verdict(kb, float(r["tflite_KB"])))
                n += 1
                print(f"  {r['model']}/{r['fold']}: arena TFLM {kb:.1f} KB (trước đó chỉ là cận dưới {r['arena_KB']} KB)")
    save_rows(csv_path, rows)
    agg = aggregate(rows)
    print(f"\nĐã đo lại {n} dòng.\n\n" + render_table(agg))
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# So sánh model KWS (int8)\n\n" + render_table(agg, markdown=True) + "\n")


# ----------------------------- main -----------------------------
def main():
    global MODELS, EPOCHS, SMOKE_TEST
    ap = argparse.ArgumentParser()                       # chỉ ghi đè mặc định ở khối cấu hình; không truyền gì vẫn chạy
    ap.add_argument("--models", nargs="+", default=None, help=f"chọn trong: {', '.join(MODEL_ZOO)}")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", default=SMOKE_TEST)
    ap.add_argument("--check", nargs="+", default=None, help="chỉ kiểm tra bộ nhớ của các file .tflite này (không train)")
    ap.add_argument("--remeasure", action="store_true", help="điền lại arena TFLM vào results.csv cho các dòng chưa có, rồi in bảng (không train)")
    a, _ = ap.parse_known_args()
    if a.check:
        check_files(a.check)
        return
    if a.remeasure:
        remeasure()
        return
    MODELS = a.models or MODELS
    EPOCHS = a.epochs or EPOCHS
    SMOKE_TEST = a.smoke
    unknown = [m for m in MODELS if m not in MODEL_ZOO]
    if unknown:
        raise SystemExit(f"Model không có trong MODEL_ZOO: {unknown}\nHiện có: {list(MODEL_ZOO)}")
    epochs_default = 2 if SMOKE_TEST else EPOCHS
    seeds = SEEDS[:1] if SMOKE_TEST else SEEDS
    os.makedirs(OUT_DIR, exist_ok=True)

    data_dir = resolve_data_dir()
    items, noise = K.collect(data_dir)
    wav_cache = {p: K.read_wav(p) for p, *_ in items}
    noise_cache = {p: K.read_wav(p) for p, _ in noise}
    extra = K.load_extra_noise(EXTRA_NOISE_DIRS)
    folds = make_folds(items, noise, noise_cache, extra)
    cfg = types.SimpleNamespace(noise_mix=USE_NOISE_MIX, p_noise=P_NOISE, snr_min=SNR_MIN, snr_max=SNR_MAX,
                                babble_prob=BABBLE_PROB)

    csv_path = os.path.join(OUT_DIR, "results_smoke.csv" if SMOKE_TEST else "results.csv")
    rows = load_rows(csv_path)
    done = {(r["model"], r["fold"], str(r["seed"])) for r in rows}
    print(f"\nSo sánh {len(MODELS)} model x {len(folds)} fold x {len(seeds)} seed; "
          f"{'SMOKE TEST (2 epoch)' if SMOKE_TEST else f'{epochs_default} epoch'}; đã có {len(rows)} kết quả trong {csv_path}")

    if MEASURE_TFLM and not _tflm:
        print("\n  CẢNH BÁO: chưa cài tflite-micro -> cột arena chỉ là CẬN DƯỚI (thường thấp hơn thật 5-60%), "
              "kết luận 'vừa ESP32?' có thể sai. Cài: pip install tflite-micro (rồi chạy lại hoặc dùng --check).")
    seen = {}
    for name in MODELS:
        m = MODEL_ZOO[name]()
        sig = (m.count_params(), count_macs(m))
        if sig in seen:
            print(f"\n  CẢNH BÁO: '{name}' và '{seen[sig]}' có cùng số tham số/MACs {sig} -> nhiều khả năng trùng kiến trúc, "
                  f"kết quả của chúng KHÔNG thể so sánh với nhau (khởi động lại runtime rồi chạy lại).")
        seen[sig] = name
        print(f"\n######## {name}: {m.count_params():,} tham số, {count_macs(m) / 1e6:.2f}M MACs ########")
        for fold in folds:
            for seed in seeds:
                if (name, fold[0], str(seed)) in done:
                    print(f"  (bỏ qua {name}/{fold[0]}/s{seed}: đã có kết quả)")
                    continue
                ep = 2 if SMOKE_TEST else MODEL_EPOCHS.get(name, epochs_default)
                print(f"\n--- {name} | val = {fold[0]} | seed {seed} | {ep} epoch ---")
                try:
                    rows.append(run_one(name, fold, seed, items, noise, wav_cache, ep, cfg))
                    save_rows(csv_path, rows)
                except Exception as e:                   # một model lỗi không làm hỏng cả đợt chạy
                    print(f"  LỖI ở {name}/{fold[0]}/s{seed}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("Không có kết quả nào.")
    agg = aggregate(rows)
    table = render_table(agg)
    print("\n######## TỔNG KẾT (trung bình qua fold/seed; xếp theo " + SORT_BY + ") ########\n" + table)
    print("\nGhi chú: file KB gồm cả tên tensor/metadata, trọng số KB là phần hằng số thật; arena KB lấy từ TFLite Micro thật "
          "(nếu đã cài tflite-micro) hoặc là cận dưới; 'vừa ESP32?' dựa trên các ngân sách GIẢ ĐỊNH ở khối cấu hình, "
          "phải đối chiếu bằng số đo trên board.")
    with open(os.path.join(OUT_DIR, "summary_smoke.md" if SMOKE_TEST else "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# So sánh model KWS (int8)\n\nSPLIT_MODE={SPLIT_MODE}, fold={[x[0] for x in folds]}, seeds={seeds}, "
                f"epochs={epochs_default}\n\n" + render_table(agg, markdown=True) + "\n")
    if MAKE_PLOT and not SMOKE_TEST:
        make_plot(agg, os.path.join(OUT_DIR, "compare.png"))

    if EXPORT_BEST and not SMOKE_TEST:
        want = [agg[0]["model"]] + [m for m in EXPORT_MODELS if m != agg[0]["model"]]
        for m in want:
            r = next((r for r in rows if r["model"] == m), None)
            if r and os.path.exists(r["tflite_path"]):
                d = os.path.join(OUT_DIR, "export", m)
                os.makedirs(d, exist_ok=True)
                K.export_c(r["tflite_path"], d)
                print(f"  đã xuất C cho '{m}' vào {d}")
        print("Sau khi chọn model, đặt nó vào train_kws.py (build_model) rồi chạy MODE='final' để train trên toàn bộ dữ liệu.")


if __name__ == "__main__":
    main()