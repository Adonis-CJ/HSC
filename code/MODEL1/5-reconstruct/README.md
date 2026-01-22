# 5-reconstruct: 轨迹重建模块

## 功能说明
该模块负责将 2D 像素轨迹映射到场地坐标系，并在可能的情况下估计 3D 轨迹。

## 输入
- 清洗后 2D 轨迹：`output/trajectory/2d/trajectory_2d_clean.csv`
- 单应性矩阵：`output/calibration/homography.json`

## 输出
- 地面坐标轨迹：`output/trajectory/3d/trajectory_ground.csv`
- 3D 轨迹（含高度）：`output/trajectory/3d/trajectory_3d.csv`
- 重建可视化：`src/img/vis/reconstruction/`

## 输出格式

### trajectory_ground.csv
```csv
t,X,Y,u,v
0.0000,25.3,45.2,523.4,312.8
0.0333,26.1,44.8,541.2,298.5
...
```

### trajectory_3d.csv
```csv
t,X,Y,Z,method,confidence
0.0000,25.3,45.2,0.22,projectile,0.85
0.0333,26.1,44.8,0.58,projectile,0.82
...
```

## 重建方法

### 方法 1: 仅地面投影（Homography Only）
- 将像素点通过单应性映射到地面平面
- 假设足球始终在地面或地面附近
- 适用于低平轨迹

### 方法 2: 单应性 + 抛体约束（推荐）
- 地面坐标 (X, Y) 通过单应性获得
- 高度 Z 通过抛体运动约束估计
- 结合初速度、发射角等参数

### 方法 3: 多视角三角化
- 需要多个同步相机
- 完整 3D 重建
- 本项目可能不适用

## 使用方法

### 地面投影
```python
python project_to_ground.py --trajectory output/trajectory/2d/trajectory_2d_clean.csv --homography output/calibration/homography.json --output output/trajectory/3d/trajectory_ground.csv
```

### 3D 估计
```python
python estimate_3d.py --ground output/trajectory/3d/trajectory_ground.csv --method projectile --output output/trajectory/3d/trajectory_3d.csv
```

## 配置参数
参见 `config/config.yaml` 中的 `reconstruct` 部分。

## 注意事项
1. 单应性仅对地面平面有效
2. 3D 估计需要物理假设（如抛体运动）
3. 应评估并报告重建误差
4. 球门高度 (2.44m) 可作为约束条件
