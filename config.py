"""
配置文件：存儲所有外部參數
"""

import os

# 文件路徑配置
CAMERA_PARAMS_PATH = r"C:\Users\godli\OneDrive\Desktop\camera_8visibility(1)\camera_8visibility\eight_camera_calibration.npz"
MODEL_ASSET_PATH = r"C:\Users\godli\OneDrive\Desktop\camera_8visibility(1)\camera_8visibility\pose_landmarker_full.task"

# 視頻路徑配置
VIDEO_PATHS = {
    'l1': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\685\685-02282025150351-0000.avi",
    'l2': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\684\684-02282025150353-0000.avi", 
    'l3': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\688\688-02282025150359-0000.avi",
    'c':  r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\1034\1034-02282025150358-0000.avi",
    'r1': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\686\686-02282025150352-0000.avi",
    'r2': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\725\725-02282025150355-0000.avi",
    'r3': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\724\724-02282025150356-0000.avi",
    'r4': r"C:\Users\godli\Dropbox\Camera_passion changes lives\8camera_0228\826\826-02282025150354-0000.avi"
}

# ArUco 參數配置
ARUCO_CONFIG = {
    'board_length': 160,  # 標記板邊長
    'board_gap': 30,      # 標記板間距
}

# MediaPipe 配置
MEDIAPIPE_CONFIG = {
    'min_pose_detection_confidence': 0.8,
    'min_pose_presence_confidence': 0.8,
    'min_tracking_confidence': 0.8
}

# GOM優化配置
GOM_CONFIG = {
    'lambda_smooth': 0.1,    # 時序平滑項權重
    'lambda_reproj': 1.0,    # 重投影誤差權重
    'max_iterations': 1000,  # 最大迭代次數
    'tolerance': 1e-6       # 收斂容差
}

# 相機順序配置
CAM_ORDER = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']

# 相機名稱映射
CAM_NAMES = {
    'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
    'c': "中心相機",
    'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
}

# 檢查配置的文件路徑是否存在
def validate_paths():
    """驗證所有配置的文件路徑"""
    if not os.path.exists(CAMERA_PARAMS_PATH):
        raise FileNotFoundError(f"相機參數文件不存在: {CAMERA_PARAMS_PATH}")
    
    if not os.path.exists(MODEL_ASSET_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_ASSET_PATH}")
    
    for cam_id, path in VIDEO_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{CAM_NAMES[cam_id]}的視頻文件不存在: {path}") 