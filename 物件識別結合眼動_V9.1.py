import sys
import os
import cv2
import time
import threading
import pyttsx3
import csv
import smtplib
import numpy as np
import requests
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from collections import defaultdict, Counter

import supervision as sv
import matplotlib.pyplot as plt  # for heatmap
from flask import Flask, Response, request

# 設定路徑以導入 Ganzin SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.streaming.gaze_stream import GazeData
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient
from ganzin.sol_sdk.requests import AddTagRequest, TagColor
from ganzin.sol_sdk.utils import find_nearest_timestamp_match, get_timestamp_ms
import firebase_admin
from firebase_admin import credentials, db

# Flask app
app = Flask(__name__)

# 全域停止旗標
stop_flag = False

# Global variable for latest frame
latest_frame = None

# 全域 annotated 變數
annotated = None

# MJPEG 串流產生器
def generate_mjpeg():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # 約 30fps

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_server():
    global stop_flag
    stop_flag = True
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return 'Not running with the Werkzeug Server', 500
    func()
    return 'Server shutting down...', 200

# 啟動 Flask Thread
def flask_thread():
    app.run(host='0.0.0.0', port=8090)

# Firebase 初始化
cred = credentials.Certificate("eye-tracking.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://eye-tracking-9c85c-default-rtdb.firebaseio.com/'
})

# 初始化語音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Roboflow API 設定
API_KEY = "mx6cAQDaGAVyXhc80rEh"
MODEL_ID = "axial-mri/1"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"

# 建立標註器
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.8)

# 狀態變數
is_inferencing = False
annotated_expire_time = 0
last_announced_class = None
inference_log = []

# 眼動軌跡
gaze_trajectory = []  # 顯示最新 N 點
recent_N_points = 30
all_gaze_points = []  # 全程記錄

# 推論統計
class_counter = defaultdict(int)
confidence_sum = defaultdict(float)

# 建立圖片儲存資料夾
SAVE_DIR = "saved_inference_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 疊加即時眼動軌跡
def draw_gaze_trajectory(img, trajectory):
    for i, pt in enumerate(trajectory):
        cv2.circle(img, pt, 8, (0, 0, 255), -1)
        if i > 0:
            cv2.line(img, trajectory[i-1], pt, (255, 255, 0), 2)
    return img

# Roboflow 推論
def roboflow_infer(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
        response.raise_for_status()
        return response.json()

# Email 發送函式
def send_email_with_csv_and_images():
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "4A930099@stust.edu.tw"
    sender_password = "trwvoyligttxqcjy"
    receiver_email = "book901006@gmail.com"

    subject = " Roboflow + Gaze 推論結果報告"
    body = "您好，附件包含辨識記錄 CSV 與所有推論時儲存的圖片，請查收。"

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject
    msg.set_content(body)

    # 附加 CSV
    with open("inference_log.csv", "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename="inference_log.csv")
    # 附加圖片
    for img_path in Path(SAVE_DIR).glob("*.jpg"):
        with open(img_path, "rb") as img_file:
            msg.add_attachment(
                img_file.read(), maintype="image", subtype="jpeg", filename=img_path.name
            )
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email 已成功寄出，共附加圖片 {len(list(Path(SAVE_DIR).glob('*.jpg')))} 張。")
    except Exception as e:
        print(" 寄送 Email 發生錯誤：", e)

# 非同步推論
def run_inference_async(image):
    global annotated, is_inferencing, annotated_expire_time, last_announced_class, inference_log
    is_inferencing = True

    try:
        temp_img_path = "webcam_capture.jpg"
        cv2.imwrite(temp_img_path, image)

        result = roboflow_infer(temp_img_path)
        print(result)

        filtered_predictions = [p for p in result.get("predictions", []) if p.get("confidence", 0) > 0.5]
        if not filtered_predictions:
            print(" 無高信心度預測，略過語音與儲存。")
            annotated = None
            is_inferencing = False
            return

        filtered_result = dict(result)
        filtered_result["predictions"] = filtered_predictions
        detections = sv.Detections.from_inference(filtered_result)
        labels = [f'{p["class"]} ({p["confidence"] * 100:.1f}%)' for p in filtered_predictions]

        first_class = filtered_predictions[0]["class"]
        conf = filtered_predictions[0]["confidence"]

        # 推論統計
        class_counter[first_class] += 1
        confidence_sum[first_class] += conf

        if first_class != last_announced_class:
            print(f" 唸出辨識結果：{first_class}")
            engine.say(first_class)
            engine.runAndWait()
            last_announced_class = first_class
        else:
            print(f" 與上次相同（{first_class}），不再唸")

        timestamp = time.time()
        inference_log.append({
            "class": first_class,
            "confidence": conf,
            "timestamp": timestamp
        })

        temp = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_img = label_annotator.annotate(scene=temp, detections=detections, labels=labels)
        combo_img = draw_gaze_trajectory(annotated_img, gaze_trajectory)
        annotated = combo_img

        filename = f"{first_class}_{int(timestamp)}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, combo_img)
        print(f" 圖片已儲存：{save_path}")
        cv2.imwrite(r"D:\examples\eye.jpg", combo_img)
        print(" 已複製圖片到 D:\\examples\\eye.jpg")
        ref = db.reference("/roboflow_results")
        ref.set({
            "class": first_class,
            "confidence": float(conf),
            "timestamp": timestamp
        })
        print("☁️ 已上傳推論結果到 Firebase Realtime Database")
        # 統計即時顯示
        print("【即時推論統計】")
        for cls in class_counter:
            avg_conf = confidence_sum[cls] / class_counter[cls]
            print(f"  {cls}: {class_counter[cls]} 次，平均信心度 {avg_conf:.2f}")

        annotated_expire_time = time.time() + 4

    except Exception as e:
        print("推論：", e)
    is_inferencing = False

def main():
    global stop_flag, latest_frame, annotated
    address, port = get_ip_and_port()
    sc = SyncClient(address, port)
    sc.begin_record()
    result = sc.run_time_sync(50)
    th = sc.create_streaming_thread(StreamingMode.GAZE_SCENE)
    th.start()

    frame_count = 0
    last_frame_for_gaze_map = None

    while not stop_flag:
        try:
            frames = sc.get_scene_frames_from_streaming(timeout=5.0)
            frame = frames[-1].get_buffer()
            last_frame_for_gaze_map = frame.copy()
            gazes = sc.get_gazes_from_streaming(timeout=5.0)
            gaze = find_nearest_timestamp_match(frames[-1].get_timestamp(), gazes)
            pt = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))
            cv2.circle(frame, pt, 30, (255, 255, 0), 5)
            gaze_trajectory.append(pt)
            all_gaze_points.append(pt)
            if len(gaze_trajectory) > recent_N_points:
                gaze_trajectory.pop(0)

            frame_count += 1
            if frame_count % 5 == 0 and not is_inferencing:
                threading.Thread(target=run_inference_async, args=(frame.copy(),)).start()

            if annotated is not None and time.time() > annotated_expire_time:
                annotated = None

            display = cv2.resize(annotated if annotated is not None else frame, (500, 500))
            latest_frame = display.copy()
            cv2.imshow(" Roboflow + Gaze Inference", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as err:
            print("串流異常：", err)
            break

    th.cancel()
    th.join()
    sc.end_record()
    cv2.destroyAllWindows()

    with open("inference_log.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["class","confidence","timestamp"])
        writer.writeheader()
        writer.writerows(inference_log)
    send_email_with_csv_and_images()
         # 匯出全程 gaze CSV
    csv_path = os.path.join(SAVE_DIR, "all_gaze_points.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "x", "y"])
        for idx, pt in enumerate(all_gaze_points):
            writer.writerow([idx, pt[0], pt[1]])
        print(f" 已匯出全程 gaze CSV：{csv_path}")

    # 全程 gaze 熱區與熱力圖
    if all_gaze_points:
        def round_pt(pt, grid=10):
            return (int(round(pt[0]/grid)*grid), int(round(pt[1]/grid)*grid))
        rounded_points = [round_pt(pt) for pt in all_gaze_points]
        hotspot, maxcount = Counter(rounded_points).most_common(1)[0]
        print(f" 最大熱區: {hotspot}, 注視次數 {maxcount}")
        gaze_map = last_frame_for_gaze_map.copy() if last_frame_for_gaze_map is not None else np.ones((800,800,3), dtype=np.uint8)*255
        for i, pt in enumerate(all_gaze_points):
            cv2.circle(gaze_map, pt, 6, (0,0,255), -1)
            if i>0:
                cv2.line(gaze_map, all_gaze_points[i-1], pt, (255,255,0), 2)
        save_map = os.path.join(SAVE_DIR, "gaze_trajectory_all_hotspot.jpg")
        cv2.imwrite(save_map, gaze_map)
        print(f" gaze 迴路圖已存: {save_map}")

        h, w = gaze_map.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        for x,y in all_gaze_points:
            if 0<=x<w and 0<=y<h:
                heatmap[y,x] += 1
        heatmap = cv2.GaussianBlur(heatmap, (0,0), sigmaX=25, sigmaY=25, borderType=cv2.BORDER_REPLICATE)
        cap = np.percentile(heatmap, 99)
        heatmap = np.minimum(heatmap, cap)
        norm = heatmap / heatmap.max() if heatmap.max()>0 else heatmap
        cmap = plt.get_cmap('jet')
        heat_rgb = (cmap(norm)*255).astype(np.uint8)[..., :3]
        overlay = cv2.addWeighted(gaze_map, 0.5, heat_rgb, 0.7, 0)
        save_heat = os.path.join(SAVE_DIR, "gaze_heatmap.jpg")
        cv2.imwrite(save_heat, overlay)
        print(f"熱力圖已存: {save_heat}")

if __name__ == '__main__':
    flask_t = threading.Thread(target=flask_thread, daemon=True)
    flask_t.start()
    print("Flask MJPEG server 啟動：http://<你的IP>:8090/video_feed")
    print(" 若要停止，請 POST 到 http://<你的IP>:8090/stop")
    main()
