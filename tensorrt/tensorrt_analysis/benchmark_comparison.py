"""TensorRT 加速前后速度对比基准测试

对 4 个模型分别运行原始推理 和 TRT 推理，对比速度差异。

用法:
    conda activate py38
    cd /home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/tensorrt/tensorrt_analysis
    python3 benchmark_comparison.py 2>&1 | tee comparison_results.txt
"""

import os, sys, time, ctypes, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, '/home/ssd/code/vh3/src/py_algorithm')

# 预加载 torch 库
CONDA_LIB = '/home/ssd/anaconda3/envs/py38/lib'
for _lib in ['libopenblas.so.0', 'libgomp.so.1']:
    try:
        ctypes.CDLL(f'{CONDA_LIB}/{_lib}', mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

import cv2
import torch
from trt_engine import TrtEngine

PARENT = os.path.join(HERE, '..', '..')
WEIGHTS = os.path.join(PARENT, 'data', 'weights')
ENGINES = os.path.join(WEIGHTS, 'engines')

WARMUP = 10
ITERS = 50
LOAD_ORIGINAL = True  # 设为 False 可跳过原始模型加载（如内存不足）


def bench_original_yolo(model, dummy_img):
    """原始 YOLOv8m (ultralytics PyTorch)"""
    times = []
    for _ in range(ITERS):
        t0 = time.time()
        results = model(dummy_img, classes=[0], conf=0.7, verbose=False)
        # 强制同步 GPU
        _ = results[0].boxes
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return np.array(times)


def bench_trt(engine, inputs):
    """TRT engine 推理"""
    times = []
    for _ in range(ITERS):
        t0 = time.time()
        engine.infer(inputs)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return np.array(times)


def bench_original_sface(recognizer, aligned):
    """原始 SFace (OpenCV FaceRecognizerSF)"""
    times = []
    for _ in range(ITERS):
        t0 = time.time()
        recognizer.feature(aligned)
        times.append((time.time() - t0) * 1000)
    return np.array(times)


def run_comparison(name, original_setup_fn, original_bench_fn,
                   engine_name, dummy_input_fn, engine_input_name):
    """运行单个模型的对比测试"""
    print(f"\n{'=' * 60}")
    print(f"[{name}]")
    print(f"{'=' * 60}")

    result = {'name': name}

    # === 原始推理 ===
    if LOAD_ORIGINAL:
        print("  加载原始模型...")
        try:
            original_model = original_setup_fn()
            # 预热
            original_bench_fn(original_model)  # reuse in fn
            # 测试
            times = original_bench_fn(original_model)
            result['original_mean_ms'] = float(times.mean())
            result['original_min_ms'] = float(times.min())
            result['original_max_ms'] = float(times.max())
            result['original_std_ms'] = float(times.std())
            result['original_p50_ms'] = float(np.median(times))
            result['original_fps'] = 1000.0 / times.mean()
            print(f"  原始耗时 (ms): mean={times.mean():.2f}, min={times.min():.2f}, "
                  f"max={times.max():.2f}, p50={np.median(times):.2f}")
            del original_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  原始模型加载失败: {e}")
            result['original_error'] = str(e)

    # === TRT 推理 ===
    engine_path = os.path.join(ENGINES, engine_name)
    if not os.path.exists(engine_path):
        print(f"  TRT engine 不存在: {engine_path}")
        result['trt_error'] = 'engine_not_found'
        return result

    print(f"  加载 TRT engine...")
    try:
        engine = TrtEngine(engine_path)
        # 准备输入
        if engine_input_name is not None:
            inputs_raw = dummy_input_fn()
            inputs = {engine_input_name: inputs_raw}
        else:
            inputs = dummy_input_fn()

        # 预热
        bench_trt(engine, inputs)
        # 测试
        times = bench_trt(engine, inputs)
        result['trt_mean_ms'] = float(times.mean())
        result['trt_min_ms'] = float(times.min())
        result['trt_max_ms'] = float(times.max())
        result['trt_std_ms'] = float(times.std())
        result['trt_p50_ms'] = float(np.median(times))
        result['trt_fps'] = 1000.0 / times.mean()
        print(f"  TRT 耗时 (ms):   mean={times.mean():.2f}, min={times.min():.2f}, "
              f"max={times.max():.2f}, p50={np.median(times):.2f}")

        del engine
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TRT engine 加载失败: {e}")
        result['trt_error'] = str(e)

    # === 加速比 ===
    if 'original_mean_ms' in result and 'trt_mean_ms' in result:
        speedup = result['original_mean_ms'] / result['trt_mean_ms']
        result['speedup'] = float(speedup)
        time_saved = result['original_mean_ms'] - result['trt_mean_ms']
        result['time_saved_ms'] = float(time_saved)
        print(f"  >>> 加速比: {speedup:.2f}x, 每帧节省: {time_saved:.1f} ms")

    return result


# ===== YOLOv8m =====
print("\n" + "=" * 80)
print("模型 1/4: YOLOv8m 人体检测")
print("=" * 80)

yolo_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
_yolo_model = None

if LOAD_ORIGINAL:
    print("  加载 ultralytics YOLOv8m...")
    from ultralytics import YOLO
    _yolo_model = YOLO(os.path.join(WEIGHTS, 'yolov8m.pt'))
    _yolo_model.to('cuda')

    # 预热
    for _ in range(WARMUP):
        _r = _yolo_model(yolo_img, classes=[0], conf=0.7, verbose=False)
        _ = _r[0].boxes
    torch.cuda.synchronize()

    # 测试
    yolo_orig_times = []
    for _ in range(ITERS):
        t0 = time.time()
        _r = _yolo_model(yolo_img, classes=[0], conf=0.7, verbose=False)
        _ = _r[0].boxes
        torch.cuda.synchronize()
        yolo_orig_times.append((time.time() - t0) * 1000)
    yolo_orig_times = np.array(yolo_orig_times)
    print(f"  原始 (PyTorch FP32): mean={yolo_orig_times.mean():.2f}ms, "
          f"min={yolo_orig_times.min():.2f}ms, p50={np.median(yolo_orig_times):.2f}ms, "
          f"FPS={1000/yolo_orig_times.mean():.1f}")

# TRT
yolo_trt = TrtEngine(os.path.join(ENGINES, 'yolov8m_fp16.engine'))
yolo_trt_input = {'images': np.random.rand(1, 3, 640, 640).astype(np.float32)}
for _ in range(WARMUP):
    yolo_trt.infer(yolo_trt_input)
torch.cuda.synchronize()

yolo_trt_times = []
for _ in range(ITERS):
    t0 = time.time()
    yolo_trt.infer(yolo_trt_input)
    torch.cuda.synchronize()
    yolo_trt_times.append((time.time() - t0) * 1000)
yolo_trt_times = np.array(yolo_trt_times)
print(f"  TRT    (FP16):        mean={yolo_trt_times.mean():.2f}ms, "
      f"min={yolo_trt_times.min():.2f}ms, p50={np.median(yolo_trt_times):.2f}ms, "
      f"FPS={1000/yolo_trt_times.mean():.1f}")

if LOAD_ORIGINAL:
    yolo_speedup = yolo_orig_times.mean() / yolo_trt_times.mean()
    print(f"  >>> 加速比: {yolo_speedup:.2f}x, "
          f"节省: {yolo_orig_times.mean() - yolo_trt_times.mean():.1f}ms")

del yolo_trt
if _yolo_model is not None:
    del _yolo_model
torch.cuda.empty_cache()


# ===== SixDRepNet =====
print("\n" + "=" * 80)
print("模型 2/4: SixDRepNet 头部姿态估计")
print("=" * 80)

if LOAD_ORIGINAL:
    print("  加载 SixDRepNet PyTorch...")
    from sixdrepnet.model import SixDRepNet
    from torchvision import transforms
    from PIL import Image

    euler_model = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='',
                              deploy=True, pretrained=False)
    ckpt = torch.load(os.path.join(WEIGHTS, '6DRepNet_300W_LP_AFLW2000.pth'), map_location='cpu')
    euler_model.load_state_dict(ckpt)
    euler_model.cuda().eval()

    euler_transforms = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 预热
    dummy_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        pil_img = Image.fromarray(dummy_crop).convert('RGB')
        inp = euler_transforms(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            _ = euler_model(inp)
    torch.cuda.synchronize()

    # 测试
    euler_orig_times = []
    for _ in range(ITERS):
        t0 = time.time()
        pil_img = Image.fromarray(dummy_crop).convert('RGB')
        inp = euler_transforms(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            _ = euler_model(inp)
        torch.cuda.synchronize()
        euler_orig_times.append((time.time() - t0) * 1000)
    euler_orig_times = np.array(euler_orig_times)
    print(f"  原始 (PyTorch FP32,含预处理): mean={euler_orig_times.mean():.2f}ms, "
          f"min={euler_orig_times.min():.2f}ms, FPS={1000/euler_orig_times.mean():.1f}")

    del euler_model
    torch.cuda.empty_cache()

# TRT
euler_trt = TrtEngine(os.path.join(ENGINES, 'sixdrepnet_fp16.engine'))
euler_trt_input = {'input': (np.random.rand(1, 3, 192, 192).astype(np.float32) - 0.45) / 0.23}
for _ in range(WARMUP):
    euler_trt.infer(euler_trt_input)
torch.cuda.synchronize()

euler_trt_times = []
for _ in range(ITERS):
    t0 = time.time()
    euler_trt.infer(euler_trt_input)
    torch.cuda.synchronize()
    euler_trt_times.append((time.time() - t0) * 1000)
euler_trt_times = np.array(euler_trt_times)
print(f"  TRT    (FP16,仅推理):          mean={euler_trt_times.mean():.2f}ms, "
      f"min={euler_trt_times.min():.2f}ms, FPS={1000/euler_trt_times.mean():.1f}")

if LOAD_ORIGINAL:
    euler_speedup = euler_orig_times.mean() / euler_trt_times.mean()
    print(f"  >>> 加速比 (含预处理): {euler_speedup:.2f}x")
    # 纯推理对比 (排除预处理)
    print(f"  注: 原始测量含 PIL→tensor 预处理, TRT 仅测纯推理")

del euler_trt
torch.cuda.empty_cache()


# ===== SFace =====
print("\n" + "=" * 80)
print("模型 3/4: SFace 人脸识别")
print("=" * 80)

sface_aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

if LOAD_ORIGINAL:
    print("  加载 OpenCV FaceRecognizerSF...")
    sface_orig = cv2.FaceRecognizerSF.create(
        os.path.join(WEIGHTS, 'face_recognition_sface_2021dec.onnx'), '')
    # 预热
    for _ in range(WARMUP):
        sface_orig.feature(sface_aligned)
    # 测试
    sface_orig_times = []
    for _ in range(ITERS):
        t0 = time.time()
        sface_orig.feature(sface_aligned)
        sface_orig_times.append((time.time() - t0) * 1000)
    sface_orig_times = np.array(sface_orig_times)
    print(f"  原始 (OpenCV ONNX): mean={sface_orig_times.mean():.2f}ms, "
          f"min={sface_orig_times.min():.2f}ms, FPS={1000/sface_orig_times.mean():.1f}")

# TRT
sface_trt = TrtEngine(os.path.join(ENGINES, 'sface_fp16.engine'))
sface_trt_input = {'data': np.random.rand(1, 3, 112, 112).astype(np.float32) * 2 - 1}
for _ in range(WARMUP):
    sface_trt.infer(sface_trt_input)
torch.cuda.synchronize()

sface_trt_times = []
for _ in range(ITERS):
    t0 = time.time()
    sface_trt.infer(sface_trt_input)
    torch.cuda.synchronize()
    sface_trt_times.append((time.time() - t0) * 1000)
sface_trt_times = np.array(sface_trt_times)
print(f"  TRT    (FP16):        mean={sface_trt_times.mean():.2f}ms, "
      f"min={sface_trt_times.min():.2f}ms, FPS={1000/sface_trt_times.mean():.1f}")

if LOAD_ORIGINAL:
    sface_speedup = sface_orig_times.mean() / sface_trt_times.mean()
    print(f"  >>> 加速比: {sface_speedup:.2f}x, "
          f"节省: {sface_orig_times.mean() - sface_trt_times.mean():.1f}ms")

del sface_trt
torch.cuda.empty_cache()


# ===== RepViT =====
print("\n" + "=" * 80)
print("模型 4/4: UpperbodyRepViT 衣物分类")
print("=" * 80)

if LOAD_ORIGINAL:
    print("  加载 RepViT PyTorch...")
    from repvit import *
    from torchvision import transforms
    from PIL import Image

    class SmartResize:
        def __init__(self, size, mode='pad'):
            self.size = size; self.mode = mode
        def __call__(self, img):
            w, h = img.size
            target_w, target_h = self.size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            new_img = Image.new('RGB', (target_w, target_h), (128, 128, 128))
            new_img.paste(img, (int((target_w - new_w) / 2), int((target_h - new_h) / 2)))
            return new_img

    class UpperbodyDualBranchRepViT(torch.nn.Module):
        def __init__(self, num_categories, num_attributes, model_name='repvit_m1_5'):
            super().__init__()
            model_configs = {'repvit_m1_5': repvit_m1_5}
            self.backbone = model_configs[model_name](distillation=True)
            feature_dim = 512
            self.feature_enhance = torch.nn.Sequential(
                torch.nn.LayerNorm(feature_dim), torch.nn.Dropout(0.2),
                torch.nn.Linear(feature_dim, feature_dim // 2), torch.nn.GELU(),
            )
            self.category_branch = torch.nn.Sequential(
                torch.nn.Linear(feature_dim // 2, 256), torch.nn.GELU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(128, num_categories)
            )
            self.category_attention = torch.nn.Sequential(
                torch.nn.Linear(feature_dim // 2, feature_dim // 2 // 4), torch.nn.GELU(),
                torch.nn.Linear(feature_dim // 2 // 4, feature_dim // 2), torch.nn.Sigmoid()
            )
        def forward(self, x):
            for module in self.backbone.features:
                x = module(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            enhanced = self.feature_enhance(x)
            cat_attn = self.category_attention(enhanced)
            return self.category_branch(enhanced * cat_attn)

    cloth_transform = transforms.Compose([
        SmartResize((224, 224), mode='pad'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cloth_model = UpperbodyDualBranchRepViT(6, 35, 'repvit_m1_5')
    ckpt = torch.load(os.path.join(WEIGHTS, 'two_branch_repvit15_finetuning_b128_e30.pth'),
                      map_location='cpu')
    cloth_model.load_state_dict(ckpt['model_state_dict'])
    cloth_model.cuda().eval()

    # 预热
    dummy_frame = np.random.randint(0, 255, (480, 360, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        pil_img = Image.fromarray(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
        inp = cloth_transform(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            _ = cloth_model(inp)
    torch.cuda.synchronize()

    # 测试
    cloth_orig_times = []
    for _ in range(ITERS):
        t0 = time.time()
        pil_img = Image.fromarray(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
        inp = cloth_transform(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            _ = cloth_model(inp)
        torch.cuda.synchronize()
        cloth_orig_times.append((time.time() - t0) * 1000)
    cloth_orig_times = np.array(cloth_orig_times)
    print(f"  原始 (PyTorch FP32,含预处理): mean={cloth_orig_times.mean():.2f}ms, "
          f"min={cloth_orig_times.min():.2f}ms, FPS={1000/cloth_orig_times.mean():.1f}")

    del cloth_model
    torch.cuda.empty_cache()

# TRT
cloth_trt = TrtEngine(os.path.join(ENGINES, 'repvit_fp16.engine'))
cloth_trt_input = {'input': (np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.45) / 0.23}
for _ in range(WARMUP):
    cloth_trt.infer(cloth_trt_input)
torch.cuda.synchronize()

cloth_trt_times = []
for _ in range(ITERS):
    t0 = time.time()
    cloth_trt.infer(cloth_trt_input)
    torch.cuda.synchronize()
    cloth_trt_times.append((time.time() - t0) * 1000)
cloth_trt_times = np.array(cloth_trt_times)
print(f"  TRT    (FP16,仅推理):          mean={cloth_trt_times.mean():.2f}ms, "
      f"min={cloth_trt_times.min():.2f}ms, FPS={1000/cloth_trt_times.mean():.1f}")

if LOAD_ORIGINAL:
    cloth_speedup = cloth_orig_times.mean() / cloth_trt_times.mean()
    print(f"  >>> 加速比 (含预处理): {cloth_speedup:.2f}x")

del cloth_trt
torch.cuda.empty_cache()


# ===== 汇总 =====
print("\n" + "=" * 80)
print("加速对比汇总")
print("=" * 80)
if LOAD_ORIGINAL:
    print(f"{'模型':<18} {'原始(ms)':>9} {'TRT(ms)':>9} {'加速比':>7} {'每帧节省':>9} {'原始FPS':>8} {'TRT_FPS':>8}")
    print("-" * 75)
    fmt = "{:<18} {:>9.1f} {:>9.1f} {:>6.2f}x {:>8.1f}ms {:>8.0f} {:>8.0f}"
    print(fmt.format('YOLOv8m', yolo_orig_times.mean(), yolo_trt_times.mean(),
                      yolo_speedup, yolo_orig_times.mean()-yolo_trt_times.mean(),
                      1000/yolo_orig_times.mean(), 1000/yolo_trt_times.mean()))
    print(fmt.format('SixDRepNet', euler_orig_times.mean(), euler_trt_times.mean(),
                      euler_speedup, euler_orig_times.mean()-euler_trt_times.mean(),
                      1000/euler_orig_times.mean(), 1000/euler_trt_times.mean()))
    print(fmt.format('SFace', sface_orig_times.mean(), sface_trt_times.mean(),
                      sface_speedup, sface_orig_times.mean()-sface_trt_times.mean(),
                      1000/sface_orig_times.mean(), 1000/sface_trt_times.mean()))
    print(fmt.format('RepViT', cloth_orig_times.mean(), cloth_trt_times.mean(),
                      cloth_speedup, cloth_orig_times.mean()-cloth_trt_times.mean(),
                      1000/cloth_orig_times.mean(), 1000/cloth_trt_times.mean()))
else:
    print(f"{'模型':<18} {'TRT(ms)':>9} {'FPS':>8}")
    print("-" * 40)
    print(f"{'YOLOv8m':<18} {yolo_trt_times.mean():>9.1f} {1000/yolo_trt_times.mean():>8.0f}")
    print(f"{'SixDRepNet':<18} {euler_trt_times.mean():>9.1f} {1000/euler_trt_times.mean():>8.0f}")
    print(f"{'SFace':<18} {sface_trt_times.mean():>9.1f} {1000/sface_trt_times.mean():>8.0f}")
    print(f"{'RepViT':<18} {cloth_trt_times.mean():>9.1f} {1000/cloth_trt_times.mean():>8.0f}")

# 端到端帧率估算
total_orig = (yolo_orig_times.mean() + euler_orig_times.mean() +
              sface_orig_times.mean() + cloth_orig_times.mean()) if LOAD_ORIGINAL else 0
total_trt = (yolo_trt_times.mean() + euler_trt_times.mean() +
             sface_trt_times.mean() + cloth_trt_times.mean())
print(f"\n4 模型纯推理总耗时: 原始={total_orig:.1f}ms, TRT={total_trt:.1f}ms")
if LOAD_ORIGINAL:
    print(f"综合加速比: {total_orig/total_trt:.2f}x, 总节省: {total_orig-total_trt:.1f}ms")
print(f"TRT 纯推理 FPS: {1000/total_trt:.0f}")

print("\n对比测试完成")
