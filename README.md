# 八相機3D人體姿態估計與運動捕捉系統

## 項目概述

這是一個基於八個相機的3D人體姿態估計和運動捕捉系統，使用MediaPipe進行姿態檢測，結合多視角三角測量和GOM(Geometric Optimization Method)優化算法，實現高精度的3D人體運動重建。

## 主要特性

- **多相機標定系統**：支持八相機同步標定
- **ArUco標記系統**：建立統一的世界坐標系
- **MediaPipe姿態檢測**：33個關鍵點的實時檢測
- **多視角三角測量**：穩健的3D重建算法
- **GOM優化**：同時考慮重投影精度、時間平滑性和骨架長度約束
- **3D可視化**：交互式動畫顯示和視頻導出

## 系統架構

```
八相機輸入 → 相機標定 → ArUco坐標系 → MediaPipe檢測 → 三角測量 → GOM優化 → 3D可視化
```

### GOM優化目標函數

```
Total_Cost = λ_smooth × Smoothness + λ_reproj × Reprojection_Error
Subject to: Limb_Length_Constraints = 0 (硬約束)
```

其中：
- **重投影誤差**：提高多視角一致性
- **時間平滑性**：減少運動跳躍
- **骨架長度約束**：保證人體結構合理性

## 安装依赖

```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install matplotlib
pip install scipy
pip install pandas
pip install pyvista
pip install pyvistaqt
pip install seaborn
```

可選依賴（用於LSTM優化）：
```bash
pip install tensorflow
```

## 使用方法

### 1. 配置文件設置

編輯 `config.py` 文件，設置：
- 相機參數文件路徑
- 視頻文件路徑
- MediaPipe模型路徑
- 優化參數

### 2. 相機標定（可選）

如果需要重新標定相機：
```python
from camera_8visibility import calibrate_eight_cameras

# 設置標定圖片文件夾路徑
calibrate_eight_cameras(
    left_1_folder, left_2_folder, left_3_folder,
    center_folder,
    right_1_folder, right_2_folder, right_3_folder, right_4_folder
)
```

### 3. 運行主程序

```python
python camera_8visibility.py
```

或者：
```python
from camera_8visibility import main
main()
```

### 4. 批量處理

```python
from camera_8visibility import batch_process_and_save

video_paths_list = [...]  # 多組視頻路徑
batch_process_and_save(video_paths_list, "output_directory")
```

## 文件結構

```
camera_8visibility/
├── camera_8visibility.py      # 主程序文件
├── config.py                  # 配置文件
├── Progress.md               # 項目進展記錄
├── README.md                 # 說明文檔
├── .gitignore               # Git忽略文件
└── camera_8visibility/      # 子目錄
    ├── eight_camera_calibration.npz  # 相機標定參數
    ├── pose_landmarker_full.task     # MediaPipe模型
    └── output/              # 輸出文件夾
```

## 配置參數

### GOM優化參數
- `lambda_smooth = 0.1`：時間平滑性權重
- `lambda_reproj = 1.0`：重投影誤差權重
- `max_iterations = 1000`：最大迭代次數
- `tolerance = 1e-6`：收斂容差

### MediaPipe參數
- `min_pose_detection_confidence = 0.8`
- `min_pose_presence_confidence = 0.8`
- `min_tracking_confidence = 0.8`

## 輸出格式

### 3D點雲數據
- 格式：NPZ文件
- 內容：`points_3d` (frames, 33, 3)
- 坐標系：ArUco世界坐標系

### 可視化輸出
- 交互式3D動畫
- 多視角影片導出（前視、側視、俯視）
- 重投影誤差分析圖表

## 系統要求

- Python 3.8+
- OpenCV 4.5+
- MediaPipe 0.10+
- 8GB+ RAM（推薦16GB）
- NVIDIA GPU（可選，用於加速）

## 性能指標

- **重投影誤差**：通常 < 2 pixels
- **處理速度**：約10-15 FPS（取決於硬件）
- **3D重建精度**：亞厘米級別

## 故障排除

### 常見問題

1. **找不到MediaPipe模型**
   - 確保 `pose_landmarker_full.task` 文件存在
   - 檢查 `config.py` 中的路徑設置

2. **相機標定失敗**
   - 檢查棋盤格圖片質量
   - 確保圖片中包含完整的棋盤格角點

3. **優化收斂問題**
   - 調整 `lambda_smooth` 和 `lambda_reproj` 權重
   - 增加 `max_iterations` 值

## 貢獻指南

歡迎提交Issues和Pull Requests！

## 許可證

本項目採用BSD-style許可證。

## 作者

- Zhang-wen feng
-  游承皓

## 更新日誌

### v2.0.0 (2024-12-24)
- ✅ 實現完整GOM優化（重投影精度 + 時間平滑性 + 骨架約束）
- ✅ 修復函數名衝突問題
- ✅ 添加配置文件管理
- ✅ 改進3D可視化系統

### v1.0.0 (2024-12-22)
- ✅ 基礎八相機系統
- ✅ MediaPipe姿態檢測
- ✅ 三角測量算法
- ✅ 基礎優化功能

## 引用

如果您在研究中使用了此項目，請引用：

```bibtex
@software{eight_camera_motion_capture,
  title={Eight-Camera 3D Human Pose Estimation and Motion Capture System},
  author={Zhang-wen feng and Yao-Chung Chang},
  year={2024},
  url={https://github.com/Yknighter424/eight-camera-motion-capture}
}
``` 