"""实时摄像头全模型推理验证 (py38 conda 环境)

用法:
    conda activate py38
    LD_PRELOAD=<conda_lib>/libopenblas.so.0:<conda_lib>/libgomp.so.1 \
    DISPLAY=:1 python3 test_camera_trt.py

测试 7 个模型:
  4 TRT:  YOLOv8m, SixDRepNet, SFace, RepViT
  3 其他: FaceDetectorYN(OpenCV), Hand(RTMDet+RTMPose, onnxruntime), MiVOLO(PyTorch)
"""

import sys, os, time, ctypes

# ---- 环境设置 ----
# 预加载 torch 需要的库 (LD_PRELOAD 备选方案)
CONDA_LIB = '/home/ssd/anaconda3/envs/py38/lib'
for _lib in ['libopenblas.so.0', 'libgomp.so.1']:
    try:
        ctypes.CDLL(f'{CONDA_LIB}/{_lib}', mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass
sys.path.insert(0, '/home/ssd/code/vh3/src/py_algorithm')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
from PIL import Image
import torch
from trt_engine import TrtEngine

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- 内联工具函数 ----
def _iou(rect1, rect2):
    xl = max(rect1[0], rect2[0]); yt = max(rect1[1], rect2[1])
    xr = min(rect1[2], rect2[2]); yb = min(rect1[3], rect2[3])
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _find_max_iou_face(last_face, current_faces, threshold=0.3):
    if current_faces is None or len(current_faces) == 0:
        return None
    if last_face is None:
        return max(current_faces, key=lambda f: f[2] * f[3])
    lx, ly, lw, lh = last_face[:4]
    last_rect = [lx, ly, lx + lw, ly + lh]
    best, best_iou = None, threshold
    for face in current_faces:
        fx, fy, fw, fh = face[:4]
        iou = _iou(last_rect, [fx, fy, fx + fw, fy + fh])
        if iou > best_iou:
            best_iou = iou; best = face
    if best is None:
        return max(current_faces, key=lambda f: f[2] * f[3])
    return best


def face_detection(frame, last_face, detector):
    h, w = frame.shape[:2]
    input_size = 320
    det_frame = cv2.resize(frame, (input_size, input_size))
    detector.setInputSize((input_size, input_size))
    _, faces = detector.detect(det_frame)
    faces = [] if faces is None else faces
    if len(faces) > 0:
        sx, sy = w / input_size, h / input_size
        for f in faces:
            f[0] *= sx; f[1] *= sy; f[2] *= sx; f[3] *= sy
            for i in range(4, 14, 2):
                f[i] *= sx; f[i+1] *= sy
    best = _find_max_iou_face(last_face, faces)
    if best is not None:
        return [best], best
    return [], None


def _rotation_matrix_to_euler(R):
    R = R.reshape(-1, 3, 3)
    sy = np.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    singular = sy < 1e-6
    pitch = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    yaw = np.arctan2(-R[:, 2, 0], sy)
    roll = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    pitch_s = np.arctan2(-R[:, 1, 2], R[:, 1, 1])
    yaw_s = np.arctan2(-R[:, 2, 0], sy)
    roll_s = np.zeros_like(R[:, 1, 0])
    pitch = np.where(singular, pitch_s, pitch)
    yaw = np.where(singular, yaw_s, yaw)
    roll = np.where(singular, roll_s, roll)
    return np.degrees(pitch[0]), -np.degrees(yaw[0]), np.degrees(roll[0])


def face_align_crop(frame, face):
    dst_pts = np.array([
        [35.819, 44.808], [76.181, 44.808], [56.000, 60.944],
        [41.885, 76.415], [70.115, 76.415]
    ], dtype=np.float32)
    src_pts = np.array([
        [face[4], face[5]], [face[6], face[7]], [face[8], face[9]],
        [face[10], face[11]], [face[12], face[13]]
    ], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts[:3], dst_pts[:3])
    return cv2.warpAffine(frame, M, (112, 112))


# ===== 加载全部模型 =====
print("=" * 60)
print("加载 7 个模型 ...")
WEIGHTS = os.path.join(HERE, '..', 'data', 'weights')
ENGINES = os.path.join(WEIGHTS, 'engines')

# 1. YOLOv8m TRT — 人体检测
driver_trt = TrtEngine(f'{ENGINES}/yolov8m_fp16.engine')
print("  [1/7] YOLOv8m TRT         — 人体检测")

# 2. SixDRepNet TRT — 头部姿态
euler_trt = TrtEngine(f'{ENGINES}/sixdrepnet_fp16.engine')
print("  [2/7] SixDRepNet TRT       — 头部姿态")

# 3. SFace TRT — 人脸识别
sface_trt = TrtEngine(f'{ENGINES}/sface_fp16.engine')
print("  [3/7] SFace TRT            — 人脸识别")

# 4. RepViT TRT — 衣物分类
cloth_trt = TrtEngine(f'{ENGINES}/repvit_fp16.engine')
print("  [4/7] RepViT TRT           — 衣物分类")

# 5. FaceDetectorYN OpenCV — 人脸检测
face_detector = cv2.FaceDetectorYN.create(
    f'{WEIGHTS}/face_detection_yunet_2023mar.onnx', "", (320, 320), 0.8, 0.4
)
print("  [5/7] FaceDetectorYN       — 人脸检测 (OpenCV)")

# 6. Hand RTMDet+RTMPose — 手部检测+关键点 (onnxruntime)
from rtmlib import Hand
hand_model = Hand(
    det=f'{WEIGHTS}/rtmdet_hand.onnx', det_input_size=(320, 320),
    pose=f'{WEIGHTS}/rtmpose_hand.onnx', pose_input_size=(256, 256),
    backend='onnxruntime', device='cpu'
)
print("  [6/7] Hand RTMDet+RTMPose  — 手部检测+关键点 (onnxruntime)")

# 7. MiVOLO PyTorch — 年龄性别
from mivolo.model.mi_volo import MiVOLO
mivolo_model = MiVOLO(
    ckpt_path=f'{WEIGHTS}/model_imdb_age_gender_4.22.pth.tar',
    device='cuda', half=True,
    use_persons=False, disable_faces=False, verbose=False
)
print("  [7/7] MiVOLO PyTorch       — 年龄性别")

print("=" * 60)

# ===== 打开摄像头 =====
cap = cv2.VideoCapture(8)
if not cap.isOpened():
    print("错误: 无法打开摄像头 /dev/video8!")
    sys.exit(1)

print("按 'q' 退出, 画面中显示 7 模型推理结果")

last_face = None
fps_times, frame_count = [], 0
sface_label = cloth_label = age_gender_label = "--"
hand_keypoints = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    frame_count += 1

    # 预处理
    frame = np.rot90(frame, k=1)[::-1, :, :]
    frame_h, frame_w = 640, int(640 * frame.shape[1] / frame.shape[0])
    frame = cv2.resize(frame, (frame_w, frame_h))
    h, w = frame.shape[:2]

    # ==== [1] YOLOv8m TRT: 人体检测 ====
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = min(640 / h, 640 / w)
    new_w, new_h = int(w * r), int(h * r)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (640 - new_w) / 2; dh = (640 - new_h) / 2
    img_letterbox = cv2.copyMakeBorder(
        img_resized, int(dh), int(np.ceil(dh)), int(dw), int(np.ceil(dw)),
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_in = np.ascontiguousarray(
        (img_letterbox.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...])
    yolo_out = driver_trt.infer({'images': img_in})
    preds = yolo_out['output0'][0].transpose(1, 0)
    person_scores = preds[:, 4]
    person_mask = person_scores > 0.5
    person_scores, bboxes_cxcywh = person_scores[person_mask], preds[:, :4][person_mask]

    has_person = False
    if len(person_scores) > 0:
        cx, cy, bw, bh = bboxes_cxcywh[:, 0], bboxes_cxcywh[:, 1], bboxes_cxcywh[:, 2], bboxes_cxcywh[:, 3]
        x1 = (cx - bw/2)*640; y1 = (cy - bh/2)*640
        x2 = (cx + bw/2)*640; y2 = (cy + bh/2)*640
        x1, y1 = (x1-dw)/r, (y1-dh)/r
        x2, y2 = (x2-dw)/r, (y2-dh)/r
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), person_scores.tolist(), 0.5, 0.45)
        if len(keep) > 0:
            has_person = True
            for k in keep.flatten():
                bx1, by1, bx2, by2 = map(int, boxes[k])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f'person {person_scores[k]:.2f}',
                            (bx1, by1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ==== [5] FaceDetectorYN: 人脸检测 ====
    faces = []
    if has_person:
        faces, last_face = face_detection(frame, last_face, face_detector)

    if faces:
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = map(int, face[:4])
        det_score = face[-1]
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
        cv2.putText(frame, f'face {det_score:.2f}', (fx, fy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # ==== [2] SixDRepNet TRT: 头部姿态 ====
        fx_min = max(0, fx-int(0.2*fh)); fy_min = max(0, fy-int(0.2*fw))
        fx_max = fx+fw+int(0.2*fh); fy_max = fy+fh+int(0.2*fw)
        crop = frame[fy_min:fy_max, fx_min:fx_max]
        if crop.size > 0:
            img_f = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_f = cv2.resize(img_f, (224, 224))[16:208, 16:208]
            img_f = img_f.astype(np.float32)/255.0
            img_f = (img_f-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
            img_f = np.ascontiguousarray(img_f.transpose(2,0,1)[np.newaxis,...])
            R = euler_trt.infer({'input': img_f})['ortho6d'].reshape(1,3,3)
            pitch, yawn, roll = _rotation_matrix_to_euler(R)

        # ==== [3] SFace TRT: 人脸识别 (每5帧) ====
        if frame_count % 5 == 0:
            aligned = face_align_crop(frame, face)
            img_s = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_s = (img_s-127.5)/128.0
            img_s = np.ascontiguousarray(img_s.transpose(2,0,1)[np.newaxis,...])
            feat = sface_trt.infer({'data': img_s})['fc1'].reshape(-1)
            sface_label = f'feat:[{feat[0]:.2f},{feat[1]:.2f}...]'

        # ==== [7] MiVOLO PyTorch: 年龄性别 (每5帧) ====
        if frame_count % 5 == 0:
            fx_c, fy_c = max(0,fx), max(0,fy)
            face_roi = frame[fy_c:fy+fh, fx_c:fx+fw]
            if face_roi.size > 0:
                fc_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                fc_rgb = cv2.resize(fc_rgb, (224,224)).astype(np.float32)/255.0
                fc_rgb = (fc_rgb-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
                fc_t = torch.from_numpy(np.ascontiguousarray(
                    fc_rgb.transpose(2,0,1))).unsqueeze(0).cuda()
                with torch.no_grad():
                    out = mivolo_model.inference(fc_t)
                age = round(out[:,2].item()*(95-1)+48, 1)
                gd = "M" if out[:,:2].softmax(-1).topk(1)[1][0].item()==0 else "F"
                age_gender_label = f'{gd} {age}y'

        # ==== [4] RepViT TRT: 衣物分类 (每10帧) ====
        if frame_count % 10 == 0:
            ci = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ci = Image.fromarray(ci)
            cw, ch = ci.size
            s = min(224/cw, 224/ch)
            ci = ci.resize((int(cw*s), int(ch*s)), Image.BILINEAR)
            canvas = Image.new('RGB', (224,224), (128,128,128))
            canvas.paste(ci, ((224-int(cw*s))//2, (224-int(ch*s))//2))
            ca = np.array(canvas).astype(np.float32)/255.0
            ca = (ca-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
            ca = np.ascontiguousarray(ca.transpose(2,0,1)[np.newaxis,...])
            logits = cloth_trt.infer({'input': ca})['category'][0]
            names = ['shirt','t-shirt','sweater','jacket','coat','dress']
            cloth_label = names[np.argmax(logits)]

        # ==== [6] Hand 手部检测+关键点 (每10帧) ====
        if frame_count % 10 == 0:
            bboxes = hand_model.det_model(frame)  # 先检测手部区域
            kpts, scores = hand_model.pose_model(frame, bboxes=bboxes)
            hand_keypoints = []
            if bboxes.shape[0] > 0:
                for i in range(min(bboxes.shape[0], 2)):
                    bx1, by1, bx2, by2 = map(int, bboxes[i][:4])
                    sc = float(scores[i].mean()) if len(scores)>i else 0
                    pts = kpts[i] if kpts is not None and len(kpts)>i else []
                    hand_keypoints.append((bx1,by1,bx2,by2,sc,pts))
        for (bx1,by1,bx2,by2,sc,pts) in hand_keypoints:
            cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,255), 2)
            cv2.putText(frame, f'hand {sc:.2f}', (bx1,by1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            for kp in pts:
                px, py = int(kp[0]), int(kp[1])
                if px>0 and py>0:
                    cv2.circle(frame, (px,py), 2, (0,200,255), -1)

        # 绘制面部关键点
        lm = [(int(face[4]),int(face[5])),(int(face[6]),int(face[7])),
              (int(face[8]),int(face[9])),(int(face[10]),int(face[11])),
              (int(face[12]),int(face[13]))]
        for lp in lm:
            cv2.circle(frame, lp, 3, (0,155,255), -1)

        # 信息面板
        cv2.putText(frame, f'pitch:{pitch:.1f} yaw:{yawn:.1f} roll:{roll:.1f}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.putText(frame, f'SFace: {sface_label}', (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.putText(frame, f'AgeGen: {age_gender_label}', (10,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
        cv2.putText(frame, f'Cloth: {cloth_label}', (10,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    else:
        sface_label = cloth_label = age_gender_label = "--"
        cv2.putText(frame, 'NO FACE', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)

    # FPS
    fps_times.append(time.time()-t0)
    if len(fps_times) > 30:
        fps_times.pop(0)
    fps = 1.0/(sum(fps_times)/len(fps_times))
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)

    cv2.imshow('7 Models TRT Test - Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("测试结束")
