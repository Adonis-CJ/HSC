# 3-dataclean: 轨迹数据清洗模块

## 功能说明
该模块负责对手动标注的 2D 轨迹进行清洗处理，包括：
- 异常点检测与剔除
- 缺失点插值补齐
- 轨迹平滑滤波

## 输入
- 手动标注轨迹：`output/trajectory/2d/trajectory_2d_manual.csv`

## 输出
- 清洗后轨迹：`output/trajectory/2d/trajectory_2d_clean.csv`
- 清洗报告：`output/trajectory/2d/clean_report.json`
- 对比可视化：`output/trajectory/2d/clean_comparison.png`

## 实际处理结果

### 输入数据
| 参数 | 值 |
|------|-----|
| 原始记录数 | 35 条 |
| 缺失帧 | 第 20 帧 |
| 时间范围 | 0.000 - 1.167 秒 |

### 清洗结果
| 指标 | 值 |
|------|-----|
| 清洗后记录数 | 36 条 |
| 异常点剔除 | 0 个 |
| 插值补齐 | 1 帧 |
| 平滑方法 | Savitzky-Golay |
| 处理耗时 | 0.119 秒 |

## 输出格式

### trajectory_2d_clean.csv
```csv
frame,t,u,v,score,is_interpolated,is_smoothed
1,0.0000,459.0000,656.0000,1.0,0,1
2,0.0333,513.0000,628.0000,1.0,0,1
20,0.6333,1215.0000,441.0000,0.0,1,1
...
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame` | int | 帧序号 |
| `t` | float | 时间戳（秒） |
| `u` | float | 水平像素坐标（平滑后） |
| `v` | float | 垂直像素坐标（平滑后） |
| `score` | float | 置信度（插值帧为 0） |
| `is_interpolated` | int | 是否为插值点 (0/1) |
| `is_smoothed` | int | 是否经过平滑 (0/1) |

### clean_report.json
```json
{
  "total_frames": 36,
  "valid_detections": 35,
  "outliers_removed": 0,
  "interpolated_frames": 1,
  "smoothing_method": "savgol",
  "processing_time": 0.119
}
```

## 处理流程

### 1. 异常点检测
采用 Z-Score 方法，阈值设为 3.0：

$$z = \frac{x - \mu}{\sigma}$$

若 $|z| > 3$，则判定为异常点。本次处理未检测到异常点。

### 2. 缺失点插值
采用三次插值（Cubic）补齐缺失的第 20 帧：

$$f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3$$

插值仅在已有数据范围内进行，不做外推。

### 3. 轨迹平滑
采用 Savitzky-Golay 滤波器：
- 窗口大小：5
- 多项式阶数：2

该滤波器在平滑的同时保留轨迹的高频特征（如转折点）。

## 使用方法

### 基本用法
```bash
cd code/MODEL1/3-dataclean
python clean_trajectory.py
```

### 完整参数
```bash
python clean_trajectory.py \
  --input ../../../output/trajectory/2d/trajectory_2d_manual.csv \
  --output ../../../output/trajectory/2d/trajectory_2d_clean.csv \
  --report ../../../output/trajectory/2d/clean_report.json \
  --config ../../../config/config.yaml \
  --total-frames 36 \
  --fps 30.0
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `trajectory_2d_manual.csv` | 输入轨迹文件 |
| `--output` | `trajectory_2d_clean.csv` | 输出轨迹文件 |
| `--report` | `clean_report.json` | 清洗报告文件 |
| `--config` | `config.yaml` | 配置文件路径 |
| `--total-frames` | 36 | 总帧数（用于插值） |
| `--fps` | 30.0 | 视频帧率 |

## 配置参数

参见 `config/config.yaml` 中的 `dataclean` 部分：

```yaml
dataclean:
  smoothing:
    method: "savgol"    # 可选: savgol, gaussian, kalman
    window_size: 5
    poly_order: 2
  interpolation:
    method: "cubic"     # 可选: linear, cubic, spline
    max_gap: 5
  outlier:
    method: "zscore"    # 可选: zscore, iqr, velocity
    threshold: 3.0
```

## 注意事项
1. 插值帧数不宜过多（建议 ≤5 帧连续缺失）
2. 平滑窗口需根据轨迹密度调整（建议为奇数）
3. 保留原始数据 `trajectory_2d_manual.csv` 用于误差分析
4. 手动标注数据质量高，通常无需剔除异常点
