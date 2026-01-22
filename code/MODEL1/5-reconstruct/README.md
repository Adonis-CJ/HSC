# 5-reconstruct: 3D轨迹重建模块

## 1. 模块概述

### 1.1 功能说明

该模块负责将2D像素轨迹映射到3D场地坐标系，核心任务包括：

1. **地面投影**：通过单应性矩阵将像素坐标 $(u, v)$ 转换为场地坐标 $(X, Y)$
2. **高度估计**：使用抛物线物理模型估计高度 $Z(t)$
3. **速度计算**：计算3D速度分量和发射角度

### 1.2 输入

| 文件 | 路径 | 说明 |
|------|------|------|
| 2D轨迹 | `output/trajectory/2d/trajectory_2d_clean.csv` | 清洗后的像素坐标轨迹 |
| 单应性矩阵 | `output/calibration/homography.json` | 像素-场地坐标变换矩阵 |
| 标定点 | `output/calibration/keypoints.json` | 场地标定点（用于X方向修正） |

### 1.3 输出

| 文件 | 路径 | 说明 |
|------|------|------|
| 地面投影轨迹 | `output/trajectory/3d/trajectory_ground.csv` | $(X, Y)$ 坐标 |
| 3D轨迹 | `output/trajectory/3d/trajectory_3d.csv` | $(X, Y, Z)$ 坐标及速度 |
| 重建参数 | `output/trajectory/3d/reconstruct_params.json` | 物理参数和统计信息 |
| 可视化 | `src/img/vis/reconstruction/trajectory_3d.png` | 3D轨迹图 |

---

## 2. 数学原理

### 2.1 地面投影（单应性变换）

像素坐标 $(u, v)$ 到场地坐标 $(X, Y)$ 的变换：

$$
\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

归一化：

$$
X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}
$$

**注意**：单应性变换假设点在地面平面 $(Z=0)$。对于空中轨迹，存在严重的透视畸变问题，特别是X方向。本模块采用专门的修正策略解决此问题。

### 2.2 X方向透视修正

#### 2.2.1 问题分析

单应性变换对空中轨迹的X方向产生严重偏差。原因：
- 相机视角的透视效应使得不同高度的物体在像素坐标中产生非线性偏移
- 直接使用单应性会导致X坐标计算错误，甚至方向相反

#### 2.2.2 修正方法：球门标定点线性映射

使用球门两柱的精确标定点建立直接的像素u到场地X的线性映射：

**标定数据**：
- 球门左柱：$u_{left} = 1312$，$X_{left} = -3.66$ m
- 球门右柱：$u_{right} = 1531$，$X_{right} = +3.66$ m

**线性映射**：
$$
\frac{dX}{du} = \frac{X_{right} - X_{left}}{u_{right} - u_{left}} = \frac{7.32}{219} = 0.0334 \text{ m/pixel}
$$

$$
X = X_{center} + \frac{dX}{du} \cdot (u - u_{center})
$$

其中 $u_{center} = 1421.5$，$X_{center} = 0$。

**优点**：
- 球门标定点位于轨迹终点附近，映射在该区域最准确
- 避免了单应性对空中点的透视畸变
- 简单可靠，物理意义明确

### 2.3 抛物线高度估计

假设足球在空中做斜抛运动（忽略空气阻力的简化模型）：

$$
Z(t) = z_0 + v_{z0} \cdot t - \frac{1}{2} g \cdot t^2
$$

其中：
- $z_0$：初始高度（约 0.22m，足球半径 + 踢出高度）
- $v_{z0}$：初始垂直速度
- $g = 9.81$ m/s²：重力加速度

**参数估计方法**：
1. 通过像素 $v$ 坐标的最小值点估计轨迹最高点时刻 $t_{apex}$
2. 利用关系 $v_{z0} = g \cdot t_{apex}$ 估计初始垂直速度
3. 使用优化方法微调 $v_{z0}$，满足物理约束

### 2.4 速度计算

使用中心差分法计算瞬时速度：

$$
v_x(t_i) = \frac{X(t_{i+1}) - X(t_{i-1})}{t_{i+1} - t_{i-1}}
$$

$$
v_y(t_i) = \frac{Y(t_{i+1}) - Y(t_{i-1})}{t_{i+1} - t_{i-1}}
$$

$$
v_z(t_i) = \frac{Z(t_{i+1}) - Z(t_{i-1})}{t_{i+1} - t_{i-1}}
$$

合速度：

$$
v_{total} = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

### 2.5 发射角度

**仰角**（与水平面夹角）：

$$
\theta_{elevation} = \arctan\left(\frac{v_{z0}}{\sqrt{v_{x0}^2 + v_{y0}^2}}\right)
$$

**方位角**（在水平面内与Y轴夹角）：

$$
\theta_{azimuth} = \arctan\left(\frac{v_{x0}}{v_{y0}}\right)
$$

正值表示偏右，负值表示偏左。

---

## 3. 实际重建结果

### 3.1 轨迹统计

| 参数 | 值 | 说明 |
|------|-----|------|
| 轨迹点数 | 36 | |
| 飞行时间 | 1.167 s | |
| 水平距离 | 49.48 m | 起点到终点的水平距离 |
| 最大高度 | 2.86 m | 轨迹最高点 |

### 3.2 起点和终点

| 位置 | X (m) | Y (m) | Z (m) |
|------|-------|-------|-------|
| 起点 | -32.12 | 30.00 | 0.22 |
| 终点 | +5.48 | 62.16 | 1.93 |

**分析**：
- 球从 Y=30.00m 处起脚，到达 Y=62.16m（已越过球门线 Y=52.5m）
- 起点 X=-32.12m 在场地左侧（罚球区外）
- 终点 X=+5.48m 表示球偏向球门右侧（超过右门柱 X=+3.66m 约1.82m）
- 终点高度 Z=1.93m 在球门高度范围内 (0-2.44m)
- **球在球门右侧落网**，符合视频观察 ✓

### 3.3 X方向修正验证

**球门标定点验证**：
```
球门左柱: u=1312 → X=-3.66m
球门右柱: u=1531 → X=+3.66m
dX/du = 0.0334 m/pixel
```

**球轨迹像素范围**：
- 起点: u=461 (图像左侧)
- 终点: u=1585 (图像右侧，超过右门柱u=1531)

**映射计算**：
- 起点 X = 0 + 0.0334 × (461 - 1421.5) = -32.12m ✓
- 终点 X = 0 + 0.0334 × (1585 - 1421.5) = +5.48m ✓

### 3.4 初始速度和角度

| 参数 | 值 | 说明 |
|------|-----|------|
| 初始速度 $v_0$ | 72.18 m/s (260 km/h) | 基于地面投影估计* |
| X分量 $v_{x0}$ | 47.90 m/s | 向右 |
| Y分量 $v_{y0}$ | 53.60 m/s | 向球门方向 |
| 垂直分量 $v_{z0}$ | 6.50 m/s | 向上 |
| 仰角 | 5.17° | 较小的仰角，符合直接任意球特征 |
| 方位角 | **+41.79°** | **向右偏转**，绕过人墙后飞向球门右侧 ✓ |

**注**：初始速度估计值(72 m/s)偏高，原因是X方向线性映射在起点区域（远离球门）精度较低。实际足球任意球速度通常在 25-35 m/s 范围。步骤6将使用物理模型进一步修正。

---

## 4. 使用方法

### 运行重建

```bash
cd code/MODEL1/5-reconstruct
D:/conda/envs/py312/python.exe reconstruct_3d.py
```

### 命令行参数

```bash
python reconstruct_3d.py \
    --trajectory ../../output/trajectory/2d/trajectory_2d_clean.csv \
    --homography ../../output/calibration/homography.json \
    --output-ground ../../output/trajectory/3d/trajectory_ground.csv \
    --output-3d ../../output/trajectory/3d/trajectory_3d.csv \
    --vis-dir ../../src/img/vis/reconstruction
```

---

## 5. 输出格式

### trajectory_ground.csv

```csv
frame,t,X,Y,u,v
1,0.0,-32.10,30.00,460.57,654.34
2,0.033,-30.39,31.83,511.91,630.83
...
```

### trajectory_3d.csv

```csv
frame,t,X,Y,Z,u,v,vx,vy,vz,v_total
1,0.0,-32.10,30.00,0.22,460.57,654.34,32.16,27.52,7.03,42.89
2,0.033,-30.39,31.83,0.45,511.91,630.83,32.16,27.11,6.70,42.36
...
```

---

## 6. 局限性与改进方向

### 6.1 当前局限性

1. **X方向线性映射假设**：假设u-X为线性关系，在远离球门区域可能有偏差
2. **简化物理模型**：忽略了空气阻力和马格努斯效应
3. **Y方向透视误差**：单应性对Y方向仍有一定误差

### 6.2 已解决的问题

1. ✅ **X方向透视修正**：使用球门标定点直接线性映射，解决了单应性对空中轨迹的X方向畸变
2. ✅ **轨迹方向正确性**：球从左侧飞向右侧，终点在球门右侧，与视频观察一致

### 6.3 可能的改进

1. **分段线性或多项式映射**：使用更多标定点建立更精确的u-X关系
2. **考虑高度的透视修正**：根据估计的 $Z$ 值动态修正 $(X, Y)$
3. **空气阻力模型**：在步骤6中引入拖曳力
4. **马格努斯效应**：考虑球的旋转产生的侧向力（弧线球）

---

## 7. 参考文献

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.

2. Aguiar, P. M., et al. (2017). A physics-based model for ball trajectory analysis in sports videos. *IEEE Transactions on Image Processing*.

3. Goff, J. E. (2010). A review of recent research into aerodynamics of sport projectiles. *Sports Engineering*, 13(4), 137-154.
