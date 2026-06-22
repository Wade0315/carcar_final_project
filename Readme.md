# Raspberry Pi YOLO Tracking Car

這個專案是以 Raspberry Pi Camera、YOLO NCNN 模型與 Arduino 控制板組成的追蹤小車系統。Raspberry Pi 負責擷取影像與目標偵測，計算目標相對於畫面中心的水平誤差，再透過 serial 將控制狀態送給 Arduino。

目前主要版本位於 `YOLO_in_rpi/`。相機擷取與 YOLO 推論已拆成 latest-frame 架構：相機 thread 持續更新最新畫面，推論 thread 永遠拿最新 frame 處理，不會排隊消化舊畫面，因此可以降低控制延遲。

## 功能特色

- 使用 Picamera2 擷取 Raspberry Pi Camera 畫面。
- 使用 NCNN runtime 載入 YOLO 匯出模型。
- 支援全畫面掃描與 ROI local tracking，追蹤穩定後可優先在目標附近推論。
- 依據偵測結果輸出追蹤、找不到、接近、NOHEAD 等控制狀態。
- 自動搜尋 Arduino serial port，未連線時仍可執行影像偵測流程。
- 將每次推論效能寫入 CSV，方便觀察 FPS、推論耗時與 frame 延遲。
- 提供 OpenCV debug view，可顯示 bbox、ROI、目標點與 error。

## 專案結構

| 路徑 | 說明 |
| --- | --- |
| `YOLO_in_rpi/main.py` | Raspberry Pi 正式入口，執行 YOLO 追蹤與 Arduino 控制。 |
| `YOLO_in_rpi/camera_YOLO.py` | Picamera2 擷取、NCNN 推論、ROI tracking 與效能紀錄。 |
| `YOLO_in_rpi/camera_base.py` | 相機基底、latest-frame thread、目標選擇與追蹤鎖定邏輯。 |
| `YOLO_in_rpi/cameraUI.py` | OpenCV debug view，會畫出候選框、目標、ROI 與 error。 |
| `YOLO_in_rpi/arduino.py` | 掃描 serial port 並傳送控制訊息給 Arduino。 |
| `YOLO_in_rpi/performance_logger.py` | 將推論效能寫入 CSV 並定期輸出摘要。 |
| `YOLO_in_rpi/ncnn_image_test.py` | 使用單張圖片測試 NCNN 模型。 |
| `YOLO/` | 模型訓練、測試與匯出 NCNN 的輔助腳本。 |
| `image_recognition/` | 早期 OpenCV/影像辨識版本。 |
| `opencv_test/` | 桌面端與 Raspberry Pi 的 OpenCV 測試腳本。 |

## 執行環境

建議使用 Raspberry Pi OS 與 Python virtual environment。Picamera2 通常由系統套件提供，因此 venv 建議加上 `--system-site-packages`。

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv

cd /home/waryt/carcar_final_project
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Python 套件至少需要：

```bash
pip install numpy pyserial pyyaml ncnn
```

如果要訓練、測試或匯出模型，桌面/訓練環境還需要：

```bash
pip install ultralytics
```

確認 Raspberry Pi 上 Picamera2 與 NCNN 可正常匯入：

```bash
python3 -c "from picamera2 import Picamera2; import ncnn; print('Picamera2 and NCNN are ready')"
```

## 模型設定

`YOLO_in_rpi/camera_YOLO.py` 目前預設會載入：

```text
/home/waryt/YOLO/best_ncnn_model_v5nu_ver2_256
```

local ROI tracking 預設會載入較小尺寸模型：

```text
/home/waryt/YOLO/best_ncnn_model_v5nu_ver2_192
```

每個 NCNN 模型目錄至少需要包含：

```text
*.param
*.bin
```

如果目錄內有 `metadata.yaml`，程式會讀取類別名稱；也可以用 `YOLO_NAMES` 手動指定。

臨時指定模型：

```bash
YOLO_MODEL=/path/to/best_ncnn_model python3 YOLO_in_rpi/main.py
```

同時指定 global/local 模型：

```bash
YOLO_MODEL=/path/to/model_256 \
YOLO_LOCAL_MODEL=/path/to/model_192 \
python3 YOLO_in_rpi/main.py
```

## 啟動方式

正式執行小車控制：

```bash
cd /home/waryt/carcar_final_project/YOLO_in_rpi
python3 main.py
```

目前 `main.py` 預設使用 `camera_YOLO.Camera`，也就是不開 OpenCV 預覽視窗的版本。執行時可按 `q` 結束；如果 stdin 不是 terminal，請用 `Ctrl+C`。

只跑 YOLO camera loop：

```bash
python3 camera_YOLO.py
```

開啟 debug view：

```bash
python3 cameraUI.py
```

單張圖片測試 NCNN 模型：

```bash
python3 ncnn_image_test.py --source /path/to/image_or_folder --output /tmp/ncnn_test
```

## 控制狀態

`main.py` 會根據 YOLO 偵測結果送出以下狀態給 Arduino：

| 數值 | 名稱 | 說明 |
| ---: | --- | --- |
| `0` | `TRACK` | 找到目標，後面會附上水平誤差，例如 `0 -18`。 |
| `1` | `NOT_FOUND` | 找不到目標，或啟動 warmup 尚未完成。 |
| `2` | `CLOSE_ENOUGH` | 目標已經足夠接近。 |
| `3` | `OUT_OF_BOUND` | 保留狀態，目前主流程較少使用。 |
| `4` | `INIT` | 初始化/停止狀態，按 `q` 離開時會送出。 |
| `5` | `NOHEAD` | 連續多幀只看到球身、沒看到球頭，且面積達門檻。 |
| `6` | `IDLE` | 暫停動作。 |

啟動後馬達不會立刻動作，程式會先等待 YOLO 穩定：

```text
至少等待 YOLO_WARMUP_SECONDS 秒
並且連續 YOLO_WARMUP_STABLE_FRAMES 次推論低於 YOLO_MAX_INFERENCE_MS
```

如果已啟用控制後連續多次推論過慢，程式會暫停馬達輸出，避免小車根據延遲太久的影像動作。

## Latest-Frame 架構

相機與推論速度通常不同。例如相機是 30 FPS，約每 33 ms 產生一張影像；但 NCNN 推論可能需要數百 ms。如果每一張 frame 都排隊處理，控制會越來越慢。

目前架構分成兩個 thread：

```text
Camera thread:
capture_array() -> 修正方向 -> 覆蓋 latest_frame -> 繼續抓下一張

Inference thread:
取得比上次更新的 latest_frame -> NCNN 推論 -> 寫入 CSV -> 產生控制狀態
```

程式只保留最新畫面，不建立 frame queue。假設推論期間相機取得：

```text
frame 100 -> 101 -> 102 -> 103 -> 104
```

下一次推論會直接使用 `frame 104`，中間 frame 會被略過。CSV 中 `frame_index` 跳號是正常現象，代表程式正在避免處理舊畫面。

## ROI Tracking

`camera_YOLO.Camera` 預設啟用 ROI tracking：

```text
YOLO_ROI_TRACKING=true
```

流程概念：

1. 沒有目標時，先跑 global scan。
2. 找到目標後，根據 bbox 建立 ROI。
3. 後續優先對 ROI 做 local inference。
4. 每隔 `YOLO_GLOBAL_SCAN_INTERVAL` 幀會再做一次 global scan。
5. local miss 後可回到 global scan 重新取得目標。

debug view 中橘色框代表目前 ROI。

## 常用環境變數

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | 本專案 logging 等級。 |
| `THIRD_PARTY_LOG_LEVEL` | `WARNING` | `picamera2`、`libcamera` 等第三方 logger 等級。 |
| `SYSTEM_LOG` | `YOLO_in_rpi/logs/system_時間.log` | 主流程系統 log 路徑。 |
| `YOLO_MODEL` | `/home/waryt/YOLO/best_ncnn_model_v5nu_ver2_256` | global NCNN 模型。 |
| `YOLO_LOCAL_MODEL` | `/home/waryt/YOLO/best_ncnn_model_v5nu_ver2_192` | local ROI NCNN 模型。 |
| `YOLO_IMGSZ` | `256` | global 模型輸入尺寸。 |
| `YOLO_LOCAL_IMGSZ` | `192` | local 模型輸入尺寸。 |
| `YOLO_CONF` | `0.25` | YOLO confidence 門檻。 |
| `YOLO_IOU` | `0.45` | NMS IoU 門檻。 |
| `YOLO_CLASS` | 空字串 | 只保留類別名稱包含此文字的結果。 |
| `YOLO_NAMES` | 空字串 | 手動指定逗號分隔類別名稱。 |
| `YOLO_NCNN_THREADS` | `3` | NCNN CPU thread 數量。 |
| `YOLO_CAMERA_FPS` | `30` | Picamera2 FPS。 |
| `YOLO_EXPOSURE_TIME_US` | `5000` | 鎖定後的曝光時間，單位為微秒。 |
| `YOLO_PERF_LOG` | `YOLO_in_rpi/logs/yolo_performance_時間.csv` | 效能 CSV 路徑；設為空字串可停用。 |
| `YOLO_PERF_SUMMARY_INTERVAL` | `30` | 每幾筆推論輸出一次效能摘要。 |
| `YOLO_ROI_TRACKING` | `true` | 是否啟用 ROI tracking。 |
| `YOLO_GLOBAL_SCAN_INTERVAL` | `5` | 每幾幀執行一次 global scan。 |
| `YOLO_LOCAL_MIN_ROI_SIDE` | `120` | local ROI 最小邊長。 |
| `YOLO_LOCAL_MAX_ROI_SIDE` | `min(width, height)` | local ROI 最大邊長。 |
| `YOLO_LOCAL_ROI_SMALL_SCALE` | `3.0` | 目標較小時 ROI 放大倍率。 |
| `YOLO_LOCAL_ROI_LARGE_SCALE` | `1.25` | 目標較大時 ROI 放大倍率。 |
| `YOLO_LOCAL_MISS_REACQUIRE_FRAMES` | `1` | local miss 幾次後回到 global scan。 |
| `YOLO_HEAD_CLOSE_AREA` | `40000` | head 類候選被視為接近的面積門檻。 |
| `YOLO_BALL_CLOSE_AREA` | `65000` | ball/body 類候選被視為接近的面積門檻。 |
| `YOLO_NOHEAD_AREA` | `3000` | 只看到球身時觸發 NOHEAD 的最小面積。 |
| `YOLO_NOHEAD_MAX_AREA` | `30000` | 只看到球身時觸發 NOHEAD 的最大面積。 |
| `YOLO_NOHEAD_TOLERANCE` | `3` | 需連續幾幀符合 NOHEAD 條件。 |
| `YOLO_NOHEAD_SLEEP_TIME` | `1.0` | 送出 NOHEAD 後暫停秒數。 |
| `YOLO_WARMUP_SECONDS` | `2` | 啟動後至少等待幾秒才允許馬達動作。 |
| `YOLO_WARMUP_STABLE_FRAMES` | `5` | 需連續幾次穩定推論才允許馬達動作。 |
| `YOLO_MAX_INFERENCE_MS` | `800` | 穩定推論的最大允許耗時。 |
| `YOLO_SLOW_INFERENCE_TOLERANCE` | `3` | 控制啟用後可容忍連續幾次慢推論。 |
| `CAMERA_LOCK_FRAME_DURATION` | `1` | 是否固定 frame duration。 |
| `CAMERA_FRAME_TIMEOUT_SECONDS` | `5` | 等待新 frame 的逾時秒數。 |
| `CAMERA_MAX_FRAME_TIMEOUTS` | `3` | 連續逾時幾次後停止。 |

範例：

```bash
LOG_LEVEL=DEBUG \
YOLO_CONF=0.35 \
YOLO_NCNN_THREADS=4 \
YOLO_IMGSZ=256 \
YOLO_LOCAL_IMGSZ=192 \
python3 main.py
```

停用 ROI tracking：

```bash
YOLO_ROI_TRACKING=0 python3 main.py
```

## 效能 CSV

預設輸出位置：

```text
YOLO_in_rpi/logs/yolo_performance_YYYYMMDD_HHMMSS.csv
```

查看最新 CSV：

```bash
cd /home/waryt/carcar_final_project/YOLO_in_rpi
ls -lt logs/
tail -n 5 "$(ls -t logs/yolo_performance_*.csv | head -n 1)"
```

重要欄位：

| 欄位 | 說明 |
| --- | --- |
| `frame_index` | 相機 thread 的 frame 流水號；跳號是正常現象。 |
| `capture_ms` | 單次 `capture_array()` 與方向修正耗時。 |
| `capture_gap_ms` | 兩張相機 frame 完成擷取的間隔。 |
| `frame_age_ms` | 推論開始使用該 frame 時，frame 已經存在多久。 |
| `preprocess_ms` | resize、padding 等前處理耗時。 |
| `inference_ms` | NCNN 推論耗時。 |
| `postprocess_ms` | decode、NMS、候選目標建立耗時。 |
| `processing_ms` | 單次完整偵測耗時。 |
| `processed_gap_ms` | 兩次完成推論並寫入 CSV 的時間差。 |
| `effective_fps` | 實際推論 FPS，約為 `1000 / processed_gap_ms`。 |
| `detect_mode` | 本次使用 global、local 或 fallback 偵測。 |
| `roi` | 本次 ROI 範圍。 |
| `detections` | NMS 後偵測數量。 |
| `candidates` | 類別與位置篩選後的候選數量。 |
| `find_ball` | 是否找到或短暫延續追蹤目標。 |
| `error` | 目標相對畫面中心的水平誤差。 |

## 模型訓練與匯出

訓練腳本位於 `YOLO/model1.py`，目前使用 Ultralytics YOLO：

```bash
python3 YOLO/model1.py
```

匯出 NCNN：

```bash
python3 YOLO/model1_export.py
```

測試模型預測：

```bash
python3 YOLO/model_test.py
```

這些腳本目前含有本機路徑，換環境時請先確認資料集與權重路徑是否存在。

## 常見問題

### 找不到 Arduino

程式仍可以執行 YOLO，但不會送出 serial 指令。請確認 USB 裝置與權限：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

必要時將目前使用者加入 `dialout`：

```bash
sudo usermod -aG dialout "$USER"
```

登出後重新登入才會生效。

### UI 沒有出現

`cameraUI.py` 使用 `cv2.imshow()`，需要桌面環境或 X forwarding。如果 Raspberry Pi 在純 SSH terminal 執行，請使用 `main.py` 或 `camera_YOLO.py` 的 headless 流程。

### CSV 只有表頭

CSV 建立後會先寫入表頭，第一輪推論完成後才會寫入資料。若結束時 log 顯示 `samples=0`，代表沒有任何一次推論完成，請查看 NCNN、Picamera2 或相機錯誤。

### `frame_index` 不連續

這是 latest-frame 架構的預期行為。相機可能以 30 FPS 抓圖，但推論只有數 FPS；程式會跳過舊畫面，避免控制延遲累積。

### 相機 thread 逾時

如果出現：

```text
timed out waiting for camera frame
```

請確認相機排線、Picamera2 安裝、相機是否被其他程序占用，以及 `CAMERA_FRAME_TIMEOUT_SECONDS` 是否設定過短。

## 開發驗證

修改 Python 程式後，可先執行語法檢查：

```bash
cd /home/waryt/carcar_final_project/YOLO_in_rpi
python3 -m py_compile \
  performance_logger.py \
  camera_base.py \
  camera_YOLO.py \
  cameraUI.py \
  arduino.py \
  main.py
```

latest-frame、Picamera2、Arduino serial 都需要在 Raspberry Pi 實機驗證。建議每次調整後確認：

```text
1. camera capture thread started
2. first camera frame captured
3. performance first sample written
4. CSV 持續增加資料列
5. frame_index 可跳號，但不應長時間停止
6. 按 q 後 Arduino 收到 INIT，log 顯示 serial closed
```
