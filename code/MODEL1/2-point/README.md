# 2-point: 足球检测与跟踪模块

## 功能说明
该模块负责从帧图像中检测足球位置，并进行跟踪关联，输出 2D 像素轨迹。

## 输入
- 帧图像序列：`src/img/frames/`
- 帧元数据：`output/trajectory/2d/frame_metadata.csv`

## 输出
- 2D 像素轨迹：`output/trajectory/2d/trajectory_2d_raw.csv`
- 检测可视化：`src/img/vis/detection/`

## 输出格式

### trajectory_2d_raw.csv
```csv
frame,t,u,v,score,bbox_x,bbox_y,bbox_w,bbox_h
1,0.0000,523.4,312.8,0.92,510,300,27,25
2,0.0333,541.2,298.5,0.88,528,286,26,25
...
```

字段说明：
- `frame`: 帧序号
- `t`: 时间戳 (秒)
- `u`, `v`: 足球中心像素坐标
- `score`: 检测置信度
- `bbox_*`: 检测框信息 (可选)

## 支持的检测方法

### 1. YOLO 目标检测（推荐）
```python
python detect_yolo.py --input src/img/frames/ --output output/trajectory/2d/
```

### 2. Hough 圆检测
```python
python detect_hough.py --input src/img/frames/ --output output/trajectory/2d/
```

### 3. 颜色分割检测
```python
python detect_color.py --input src/img/frames/ --output output/trajectory/2d/
```

## 跟踪算法
- Kalman 滤波 + 匈牙利匹配
- SORT / ByteTrack（可选）

## 配置参数
参见 `config/config.yaml` 中的 `detection` 和 `tracking` 部分。

## 注意事项
1. 对于运动模糊帧，置信度可能较低
2. 遮挡情况下使用跟踪预测补齐
3. 输出所有检测结果，清洗工作留给下一模块
