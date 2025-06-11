
首先進行系統初始化，包括 YOLO 目標檢測模型初始化、眼動儀同步初始化，並啟動 pyttsx3 語音引擎，完成核心功能模組準備。接續階段啟動眼動儀串流，進行即時眼動數據擷取，系統同步擷取前端影像畫面（Frame）與對應眼動點座標，並持續更新 gaze_trajectory（當前注視軌跡）與 all_gaze_points（全程眼動座標紀錄）。系統於主迴圈中定期判斷是否需觸發 YOLO 目標檢測推論處理，推論頻率可設定以平衡即時性與系統負載。當 YOLO 模型完成推論後，系統判斷是否存在腫瘤區域，如檢測出腫瘤異常區塊，將立即執行語音提示提醒使用者，並將該推論結果以 bounding box 標註於影像畫面中，同步加疊 gaze_trajectory，呈現當前眼動軌跡，提供即時視覺輔助資訊，同時將辨識結果將上傳至雲端資料庫。系統持續迴圈執行，並於每次畫面更新後判斷使用者是否有透過按鍵操作進行「結束」流程指令。當使用者觸發結束流程，系統進入結束流程處理階段，包含停止眼動儀串流 Thread、匯出 inference_log.csv（推論紀錄）、匯出 all_gaze_points.csv（全程眼動資料）、生成 gaze_trajectory_all_hotspot.jpg（全程注視熱區圖）、生成 gaze_heatmap.jpg（注視熱力圖），並將以上結果整合發送至設定電子郵件信箱，完成全流程資料保存與通報。




firebase資料庫的JOSN檔案需自行下載及建置資料庫

cred = credentials.Certificate("eye.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://eye-tracking-9c85c-default-rtdb.firebaseio.com/'
})

