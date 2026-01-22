# 轨迹数据输出目录

该目录存放足球轨迹数据文件。

## 子目录结构

- `2d/` - 2D 像素轨迹数据
- `3d/` - 3D 场地坐标轨迹数据

## 文件说明

### 2D 目录
- `frame_metadata.csv` - 帧序号与时间戳映射
- `trajectory_2d_raw.csv` - 原始检测轨迹
- `trajectory_2d_clean.csv` - 清洗后轨迹
- `clean_report.json` - 数据清洗报告

### 3D 目录
- `trajectory_ground.csv` - 地面投影坐标
- `trajectory_3d.csv` - 完整 3D 轨迹
