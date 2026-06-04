# Raspberry Pi YOLO Tracking Car

這個專案使用 Raspberry Pi Camera、NCNN 模型與 Arduino 控制小車。程式會持續偵測目標球，計算目標相對於畫面中心的水平誤差，並將控制狀態傳送給 Arduino。

目前 Raspberry Pi 版本位於 `YOLO_in_rpi/`。相機擷取與 YOLO 推論已拆成不同 thread：相機持續更新最新畫面，推論永遠取目前最新的一張，不會排隊處理舊 frame。

## 目錄

| 檔案 | 用途 |
| --- | --- |
| `YOLO_in_rpi/main.py` | 正式入口。執行 YOLO、UI 預覽與 Arduino 控制。 |
| `YOLO_in_rpi/camera_YOLO.py` | Picamera2、NCNN 推論、latest-frame thread、效能記錄。 |
| `YOLO_in_rpi/cameraUI.py` | 在 `camera_YOLO.Camera` 上加入 OpenCV 畫框與預覽視窗。 |
| `YOLO_in_rpi/camera_base.py` | 地板遮罩、目標選擇、短暫遺失目標時的追蹤策略。 |
| `YOLO_in_rpi/arduino.py` | 尋找 serial port，向 Arduino 傳送控制狀態。 |
| `YOLO_in_rpi/performance_logger.py` | 將每次推論的耗時記錄到 CSV。 |
| `YOLO_in_rpi/ncnn_image_test.py` | 使用單張圖片測試 NCNN 模型。 |

## 執行環境

建議在 Raspberry Pi OS 使用 Python virtual environment：

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv

cd ~/USER/YOLO/
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Python 套件至少需要：

```bash
pip install numpy pyserial pyyaml ncnn
```

相機需要 Raspberry Pi OS 提供的 `picamera2`。NCNN Python runtime 的安裝方式取決於 Raspberry Pi 架構與目前使用的 build，請確認以下命令可以成功：

```bash
python3 -c "from picamera2 import Picamera2; import ncnn; print('Picamera2 and NCNN are ready')"
```

## 模型

預設模型位置：

```text
/home/waryt/YOLO/best_ncnn_model_v5nu
```

模型目錄至少需要：

```text
model.param
model.bin
```

如果目錄中有 `metadata.yaml`，程式會讀取類別名稱。也可以使用 `YOLO_MODEL` 指定其他模型：

```bash
YOLO_MODEL=/home/waryt/YOLO/my_ncnn_model python3 main.py
```

## 啟動方式

進入 Raspberry Pi 程式目錄：

```bash
cd ~/USER/YOLO/YOLO_in_rpi/
```

正式執行小車控制與 UI 預覽：

```bash
python3 main.py
```


只執行 YOLO，不開 UI：

```bash
python3 camera_YOLO.py
```

單獨執行 UI 除錯：

```bash
python3 cameraUI.py
```

## Latest-Frame 相機架構

### 為什麼需要背景 thread

NCNN 推論速度比相機 FPS 慢。例如：

```text
相機：30 FPS，約每 33 ms 產生一張影像
推論：約 265 ms 才能完成一次
```

如果相機抓圖與推論都放在同一個 loop，推論期間 Python 不會主動取得新 frame。下一次推論可能讀到較舊的 buffer，反應延遲也會增加。

目前程式將工作拆成兩個 thread：

```text
Camera thread:
capture_array() -> 修正方向 -> 覆蓋 latest_frame -> 繼續抓下一張

Inference thread:
取得比上次更新的 latest_frame -> NCNN 推論 -> 記錄 CSV -> 回傳控制結果
```

### 相機 thread

初始化完成後，`camera_YOLO.Camera.start_frame_capture()` 會建立背景 thread：

```python
self.capture_thread = threading.Thread(
    target=self.capture_latest_frames,
    name="camera-capture",
    daemon=True,
)
```

背景 thread 持續執行：

```python
frame = self.picam2.capture_array()
frame = self.fix_orientation(frame)

with self.latest_frame_lock:
    self.latest_frame = frame
    self.latest_capture_ms = capture_ms
    self.latest_frame_index = frame_index

self.latest_frame_ready.set()
frame_index += 1
```

它只保留最新畫面，不會建立 queue。假設推論期間相機取得：

```text
frame 100 -> 101 -> 102 -> 103 -> 104
```

下一次推論直接使用 `frame 104`。`101` 到 `103` 會被略過，不會堆積。

### Lock 的用途

`latest_frame_lock` 只會在更新或讀取以下三個欄位時短暫鎖定：

```text
latest_frame
latest_capture_ms
latest_frame_index
```

lock 可以避免推論 thread 讀到混合狀態，例如新影像搭配舊 frame index。

以下操作不在 lock 內：

```text
capture_array()
fix_orientation()
NCNN 推論
OpenCV 畫框
CSV 寫入
```

因此推論約耗時數百毫秒，也不會阻塞相機 thread。

### `get_latest_frame()`

推論 loop 不再直接呼叫 `capture_array()`，而是執行：

```python
raw_frame, capture_ms, frame_index = self.get_latest_frame(last_frame_index)
last_frame_index = frame_index
```

`after_frame_index` 參數用來保證不重複處理同一張畫面。如果相機還沒有產生更新的 frame，推論 thread 會等待 `latest_frame_ready` Event。

時間軸範例：

```text
Camera thread:    10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16
Inference thread: 10 -------- 推論中 --------> 16
```

推論 FPS 不一定會提高很多，因為 NCNN 仍然是主要瓶頸。但每次推論會盡量使用當下最新的畫面，降低控制延遲。

## 控制流程

`main.py` 會使用有 UI 的 `cameraUI.Camera`：

```text
main.py
  -> cameraUI.Camera.streaming()
  -> get_latest_frame()
  -> detect_frame()
  -> record_performance()
  -> visualize_frame()
  -> yield find_ball, error, target
  -> Arduino.send()
```

啟動時會先等待 YOLO 推論穩定：

```text
至少等待 YOLO_WARMUP_SECONDS
連續 YOLO_WARMUP_STABLE_FRAMES 次推論低於 YOLO_MAX_INFERENCE_MS
```

穩定前馬達保持停止。

控制狀態：

| 數值 | 名稱 | 說明 |
| ---: | --- | --- |
| `0` | `TRACK` | 已找到目標，需要持續追蹤。 |
| `1` | `NOT_FOUND` | 找不到目標。 |
| `2` | `CLOSE_ENOUGH` | 目標已經足夠接近。 |
| `3` | `OUT_OF_BOUND` | 目標超出允許範圍。 |
| `4` | `IDLE` | 暫停動作。 |

## 環境變數

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Python logging 等級，例如 `DEBUG`。 |
| `YOLO_MODEL` | `/home/waryt/YOLO/best_ncnn_model_v5nu` | NCNN 模型目錄。 |
| `YOLO_CAMERA_FPS` | `30` | 相機設定 FPS。 |
| `YOLO_EXPOSURE_TIME_US` | `5000` | 鎖定後的曝光時間，單位為微秒。 |
| `YOLO_IMGSZ` | `256` | NCNN 輸入圖片尺寸。 |
| `YOLO_CONF` | `0.25` | YOLO 信心門檻。 |
| `YOLO_IOU` | `0.45` | NMS IoU 門檻。 |
| `YOLO_CLASS` | 空字串 | 只保留類別名稱包含此文字的偵測結果。 |
| `YOLO_NAMES` | 空字串 | 手動指定逗號分隔類別名稱。 |
| `YOLO_PERF_LOG` | `YOLO_in_rpi/logs/yolo_performance_時間.csv` | 效能 CSV 路徑。設為空字串可停用。 |
| `YOLO_PERF_SUMMARY_INTERVAL` | `30` | 每幾筆推論在終端機輸出一次摘要。 |
| `YOLO_WARMUP_SECONDS` | `2` | 啟動後至少等待幾秒才允許馬達動作。 |
| `YOLO_WARMUP_STABLE_FRAMES` | `5` | 需要連續幾次穩定推論才允許馬達動作。 |
| `YOLO_MAX_INFERENCE_MS` | `800` | 穩定推論的最大允許耗時。 |

例如：

```bash
LOG_LEVEL=DEBUG \
YOLO_CONF=0.35 \
YOLO_IMGSZ=256 \
python3 main.py
```

## 效能 CSV

預設輸出位置：

```text
YOLO_in_rpi/logs/yolo_performance_YYYYMMDD_HHMMSS.csv
```

查看最新 CSV：

```bash
ls -lt logs/
tail -n 5 "$(ls -t logs/yolo_performance_*.csv | head -n 1)"
```

重要欄位：

| 欄位 | 用途 |
| --- | --- |
| `frame_index` | 相機 thread 的 frame 流水號。跳號代表推論期間略過了舊畫面，屬於預期行為。 |
| `capture_ms` | 背景 thread 執行一次 `capture_array()` 與方向修正的耗時。 |
| `preprocess_ms` | resize、padding 等 NCNN 前處理耗時。 |
| `inference_ms` | NCNN 推論耗時。 |
| `postprocess_ms` | 解碼、NMS、建立候選目標的耗時。 |
| `processing_ms` | 單次完整偵測耗時。 |
| `processed_gap_ms` | 兩次完成推論並寫入 CSV 的時間差。 |
| `effective_fps` | `1000 / processed_gap_ms`，代表實際推論 FPS。 |
| `detections` | NMS 後的偵測數量。 |
| `candidates` | 類別與位置篩選後的候選數量。 |
| `find_ball` | 是否找到或短暫延續追蹤目標。 |
| `error` | 目標相對畫面中心的水平誤差。 |

使用 latest-frame thread 後，正常情況下：

```text
processed_gap_ms 約等於 processing_ms 加上少量主迴圈開銷
frame_index 可能跳號
```

這代表程式略過舊畫面，持續使用最新 frame。

## 常見問題

### CSV 只有表頭

CSV 建立後會先寫入表頭。第一輪推論完成後才會寫入資料。

正常執行時終端機會出現：

```text
performance log=/.../logs/yolo_performance_....csv
performance first sample written path=/.../logs/yolo_performance_....csv
```

按 `q` 結束時會出現：

```text
performance log closed path=... samples=123
```

如果 `samples=0`，代表沒有任何一次推論完成。請查看終端機是否有 NCNN、Picamera2 或 OpenCV 錯誤。

### `frame_index` 不連續

這是正常現象。相機 thread 可能以 30 FPS 持續抓圖，但推論只能處理約 3 到 4 FPS。程式刻意只處理最新 frame，避免因 queue 累積造成延遲。

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

`cameraUI.py` 使用 `cv2.imshow()`，需要桌面環境或 X forwarding。如果 Raspberry Pi 在純 SSH terminal 執行，請改用：

```bash
python3 camera_YOLO.py
```

### 相機 thread 逾時

如果終端機出現：

```text
timed out waiting for camera frame
```

請確認相機排線、Picamera2 安裝與相機是否被其他程序占用。

## 開發驗證

修改 Python 程式後，可先執行語法檢查：

```bash
python3 -m py_compile \
  performance_logger.py \
  camera_base.py \
  camera_YOLO.py \
  cameraUI.py \
  arduino.py \
  main.py
```

latest-frame thread 涉及 Picamera2，需要在 Raspberry Pi 實機驗證。建議每次調整後至少確認：

```text
1. camera capture thread started
2. performance first sample written
3. CSV 持續增加資料列
4. frame_index 允許跳號，但不應長時間停止
5. 按 q 後顯示 performance log closed
```
