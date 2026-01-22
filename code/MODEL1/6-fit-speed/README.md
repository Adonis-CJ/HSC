# 6-fit-speed: 曲线拟合与初速度估计模块

## 功能说明
该模块负责：
1. 对轨迹进行曲线拟合
2. 估计足球初速度及相关物理量
3. 不确定性分析

## 输入
- 3D 轨迹数据：`output/trajectory/3d/trajectory_3d.csv`
- 地面轨迹数据：`output/trajectory/3d/trajectory_ground.csv`

## 输出
- 拟合参数：`output/fitting/fit_params.json`
- 速度估计：`output/velocity/v0.json`
- 拟合可视化：`src/img/vis/fitting/`

## 输出格式

### fit_params.json
```json
{
  "ground_curve": {
    "method": "polynomial",
    "degree": 3,
    "coefficients_x": [a0, a1, a2, a3],
    "coefficients_y": [b0, b1, b2, b3],
    "r_squared": 0.995
  },
  "height_curve": {
    "method": "parabola",
    "a": -4.905,
    "b": 12.3,
    "c": 0.22,
    "r_squared": 0.988
  }
}
```

### v0.json
```json
{
  "initial_velocity": {
    "magnitude": 28.5,
    "unit": "m/s",
    "components": {
      "vx": 15.2,
      "vy": 22.1,
      "vz": 10.8
    }
  },
  "launch_angle": {
    "horizontal": 55.4,
    "vertical": 22.3,
    "unit": "degrees"
  },
  "uncertainty": {
    "v_magnitude": 1.2,
    "v_components": [0.8, 1.0, 0.6],
    "angle_horizontal": 2.1,
    "angle_vertical": 1.5
  },
  "method": "linear_regression",
  "frames_used": 5,
  "timestamp": "2026-01-22T10:00:00"
}
```

## 拟合方法

### 1. 地面投影曲线拟合
- 多项式拟合：$X(t) = \sum a_i t^i$, $Y(t) = \sum b_i t^i$
- 样条拟合：B-Spline / Cubic Spline
- Bezier 曲线

### 2. 高度曲线拟合
- 抛物线：$Z(t) = -\frac{1}{2}gt^2 + v_{z0}t + z_0$
- 带阻力模型：数值积分拟合

### 3. 物理模型拟合（高级）
- 考虑空气阻力
- 考虑 Magnus 效应（旋转）
- 参数反演

## 初速度估计方法

### 方法 1: 差分法
$$\mathbf{v}_0 \approx \frac{\mathbf{x}_1 - \mathbf{x}_0}{t_1 - t_0}$$

### 方法 2: 线性回归（推荐）
对前 N 帧做线性回归：
$$\mathbf{x}(t) = \mathbf{x}_0 + \mathbf{v}_0 t$$

### 方法 3: 物理模型拟合
拟合完整物理模型，提取初速度参数。

## 使用方法

### 曲线拟合
```python
python fit_trajectory.py --input output/trajectory/3d/trajectory_3d.csv --output output/fitting/fit_params.json
```

### 速度估计
```python
python estimate_velocity.py --input output/trajectory/3d/trajectory_3d.csv --output output/velocity/v0.json --method linear_regression --frames 5
```

## 配置参数
参见 `config/config.yaml` 中的 `fitting` 和 `velocity` 部分。

## 注意事项
1. 初速度估计应使用轨迹开始的若干帧
2. 报告置信区间，而非仅点估计
3. 拟合质量指标 (R², RMSE) 应一并输出
4. 物理可行性检查（速度应在合理范围内，如 15-40 m/s）
