# 3-dataclean: 轨迹数据清洗模块

## 功能说明
该模块负责对原始 2D 轨迹进行清洗处理，包括：
- 异常点检测与剔除
- 缺失点插值补齐
- 轨迹平滑滤波

## 输入
- 原始 2D 轨迹：`output/trajectory/2d/trajectory_2d_raw.csv`

## 输出
- 清洗后轨迹：`output/trajectory/2d/trajectory_2d_clean.csv`
- 清洗报告：`output/trajectory/2d/clean_report.json`

## 输出格式

### trajectory_2d_clean.csv
```csv
frame,t,u,v,score,is_interpolated,is_smoothed
1,0.0000,523.4,312.8,0.92,0,1
2,0.0333,541.2,298.5,0.88,0,1
3,0.0667,558.9,284.2,0.00,1,1
...
```

### clean_report.json
```json
{
  "total_frames": 150,
  "valid_detections": 142,
  "outliers_removed": 3,
  "interpolated_frames": 5,
  "smoothing_method": "savgol",
  "processing_time": 1.23
}
```

## 处理流程

### 1. 异常点检测
支持方法：
- Z-Score 检测
- IQR 四分位距检测
- 速度阈值检测

### 2. 缺失点插值
支持方法：
- 线性插值
- 三次样条插值
- Cubic 插值

### 3. 轨迹平滑
支持方法：
- Savitzky-Golay 滤波
- 高斯滤波
- Kalman 平滑

## 使用方法
```python
python clean_trajectory.py --input output/trajectory/2d/trajectory_2d_raw.csv --output output/trajectory/2d/trajectory_2d_clean.csv
```

## 配置参数
参见 `config/config.yaml` 中的 `dataclean` 部分。

## 注意事项
1. 插值帧数不宜过多（建议 ≤5 帧）
2. 平滑窗口需根据轨迹密度调整
3. 保留原始数据用于误差分析
