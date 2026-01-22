# 4-field-calib: 场地标定模块

## 1. 模块概述

### 1.1 功能说明

该模块负责建立**像素坐标系**到**场地物理坐标系**的映射关系，核心任务包括：

1. **关键点标注**：在图像上手动标注已知物理坐标的场地特征点
2. **单应性估计**：基于点对应关系计算 3×3 单应性矩阵 $\mathbf{H}$
3. **坐标变换**：实现像素坐标 $(u, v)$ 到场地坐标 $(x, y)$ 的双向变换

### 1.2 输入

| 文件 | 路径 | 说明 |
|------|------|------|
| 参考帧图像 | `src/img/frames/000001.jpg` | 用于标注的清晰帧 |
| 坐标系参数 | `code/MODEL1/4-field-calib/坐标系参数.md` | 预定义的场地关键点物理坐标 |
| 2D轨迹 | `output/trajectory/2d/trajectory_2d_clean.csv` | 待转换的像素坐标轨迹 |

### 1.3 输出

| 文件 | 路径 | 说明 |
|------|------|------|
| 标注关键点 | `output/calibration/keypoints.json` | 图像像素坐标与场地物理坐标的对应关系 |
| 单应性矩阵 | `output/calibration/homography.json` | 3×3变换矩阵及其逆矩阵 |
| 标定可视化 | `src/img/vis/calibration/calibration_result.png` | 重投影验证图 |

## 2. 场地坐标系定义

根据国际足联（FIFA）标准场地规格，建立如下右手坐标系（原点在场地中心）：

```
                         Y = 52.5m (球门线)
    ────────────────────┬────────┬────────────────────
    │                   │  球门  │                   │
    │   ┌───────────────┴────────┴───────────────┐   │  Y = 47m (小禁区线)
    │   │              小禁区                     │   │
    │   └─────────────────────────────────────────┘   │
    │   ┌─────────────────────────────────────────┐   │  Y = 36m (大禁区线)
    │   │                                         │   │
    │   │              大禁区                      │   │
    │   │                 · 罚球点 (Y=41.5m)      │   │
    │   │                                         │   │
    │   └─────────────────────────────────────────┘   │
    │                       │                         │
    │                       │                         │
    │                       │                         │
    │  -X ←─────────────────O─────────────────→ +X   │  Y = 0 (中线)
    │                       │                         │
    │                       ↓ Y (负方向)              │
    │                       │                         │
    │   ┌─────────────────────────────────────────┐   │  Y = -36m
    │   │              大禁区                      │   │
    │   │                 · 罚球点                 │   │
    │   └─────────────────────────────────────────┘   │
    │   ┌─────────────────────────────────────────┐   │  Y = -47m
    │   │              小禁区                     │   │
    │   └───────────────┬────────┬───────────────┘   │
    │                   │  球门  │                   │
    ────────────────────┴────────┴────────────────────  Y = -52.5m
```

**坐标系约定**：
- **原点 O**：场地几何中心
- **X轴**：平行于球门线，**向右为正**
- **Y轴**：垂直于球门线，向进攻方向球门为正
- **Z轴**：竖直向上（右手系）
- **单位**：米 (m)

### 2.1 标准场地尺寸参考

| 要素 | 尺寸 (m) | 坐标范围 |
|------|----------|----------|
| 场地长度 | 105 | Y: [-52.5, +52.5] |
| 场地宽度 | 68 | X: [-34, +34] |
| 球门宽度 | 7.32 | X: [-3.66, +3.66] |
| 球门高度 | 2.44 | Z: [0, 2.44] |
| 大禁区深度 | 16.5 | Y: [36, 52.5] |
| 大禁区宽度 | 40.32 | X: [-20.16, +20.16] |
| 小禁区深度 | 5.5 | Y: [47, 52.5] |
| 小禁区宽度 | 18.32 | X: [-9.16, +9.16] |
| 罚球点距离 | 11.0 | Y = 41.5 |

---

## 3. 数学原理

### 3.1 单应性变换 (Homography)

单应性变换描述了两个平面之间的**透视映射关系**。对于场地标定问题，我们建立像素平面到地面平面（$z=0$）的映射。

给定像素坐标 $(u, v)$ 和对应的场地坐标 $(x, y)$，其齐次坐标表示下的变换关系为：

$$
\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

归一化后得到欧氏坐标：

$$
x = \frac{x'}{w} = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad y = \frac{y'}{w} = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}
$$

**单应性矩阵性质**：
- 尺寸：$3 \times 3$
- 自由度：8（9个元素减去1个尺度因子，通常固定 $h_{33} = 1$）
- 最少点数：4对非共线点

### 3.2 直接线性变换 (DLT) 算法

给定 $n$ 对点 $(u_i, v_i) \leftrightarrow (x_i, y_i)$，将变换方程重写为齐次线性方程组：

对于每对点，有约束：

$$
\begin{cases}
x_i(h_{31}u_i + h_{32}v_i + h_{33}) = h_{11}u_i + h_{12}v_i + h_{13} \\
y_i(h_{31}u_i + h_{32}v_i + h_{33}) = h_{21}u_i + h_{22}v_i + h_{23}
\end{cases}
$$

整理为 $\mathbf{A}\mathbf{h} = \mathbf{0}$：

$$
\begin{bmatrix}
u_1 & v_1 & 1 & 0 & 0 & 0 & -x_1 u_1 & -x_1 v_1 & -x_1 \\
0 & 0 & 0 & u_1 & v_1 & 1 & -y_1 u_1 & -y_1 v_1 & -y_1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
u_n & v_n & 1 & 0 & 0 & 0 & -x_n u_n & -x_n v_n & -x_n \\
0 & 0 & 0 & u_n & v_n & 1 & -y_n u_n & -y_n v_n & -y_n
\end{bmatrix}
\begin{bmatrix} h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32} \\ h_{33} \end{bmatrix} = \mathbf{0}
$$

**求解方法**：对矩阵 $\mathbf{A}$ 进行奇异值分解 (SVD)：

$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
$$

最优解 $\mathbf{h}^*$ 为 $\mathbf{V}$ 的最后一列（对应最小奇异值）。

### 3.3 RANSAC 鲁棒估计

当存在标注误差或噪声时，使用 RANSAC (Random Sample Consensus) 进行鲁棒估计：

**算法流程**：
1. 从 $n$ 对点中随机采样最小集（4对点）
2. 用 DLT 估计单应性矩阵 $\mathbf{H}$
3. 计算所有点的重投影误差，统计内点数量
4. 重复步骤1-3，选择内点最多的模型
5. 用所有内点重新估计 $\mathbf{H}$

**OpenCV 实现**：
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```
其中阈值 5.0 表示重投影误差容忍度（像素）。

### 3.4 重投影误差

评估标定质量的核心指标。对于第 $i$ 个标定点：

$$
\text{err}_i = \left\| \mathbf{p}_i^{\text{real}} - \mathbf{p}_i^{\text{proj}} \right\|_2
$$

其中：
- $\mathbf{p}_i^{\text{real}} = (x_i, y_i)$ 是真实场地坐标
- $\mathbf{p}_i^{\text{proj}} = f(\mathbf{H}, u_i, v_i)$ 是通过单应性矩阵变换得到的投影坐标

**平均重投影误差**：

$$
\bar{e} = \frac{1}{n} \sum_{i=1}^{n} \text{err}_i
$$

---

## 4. 实际标定结果

### 4.1 标注关键点

本项目使用 8 个场地特征点进行标定：

| 序号 | 名称 | 像素坐标 (u, v) | 场地坐标 (x, y, z) | 说明 |
|------|------|-----------------|-------------------|------|
| 1 | goal_left_bottom | (1312, 523) | (-3.66, 52.5, 0) | 球门左柱底部（X负方向）|
| 2 | goal_right_bottom | (1531, 570) | (3.66, 52.5, 0) | 球门右柱底部（X正方向）|
| 3 | goal_left_top | (1315, 408) | (-3.66, 52.5, 2.44) | 球门左柱顶部* |
| 4 | goal_right_top | (1537, 444) | (3.66, 52.5, 2.44) | 球门右柱顶部* |
| 5 | field_corner_left | (689, 403) | (-34, 52.5, 0) | 场地左顶点（X负方向）|
| 6 | penalty_spot | (947, 584) | (0, 41.5, 0) | 罚球点（X=0）|
| 7 | penalty_area_left | (305, 490) | (-20.16, 36, 0) | 大禁区左下角（X负方向）|
| 8 | penalty_area_right | (1306, 789) | (20.16, 36, 0) | 大禁区右下角（X正方向）|

> **注意**：标记 * 的点（球门顶部）$z \neq 0$，不在地面平面上，被 RANSAC 自动识别为离群点排除。

### 4.2 单应性矩阵

**像素 → 场地** 变换矩阵 $\mathbf{H}$：

$$
\mathbf{H} = \begin{bmatrix}
-0.5644 & -7.1969 & 4728.24 \\
-1.0999 & -0.3085 & -1344.68 \\
-0.00157 & -0.1050 & 1.0
\end{bmatrix}
$$

**场地 → 像素** 逆变换矩阵 $\mathbf{H}^{-1}$：

$$
\mathbf{H}^{-1} = \begin{bmatrix}
-0.2356 & -0.8147 & 18.54 \\
0.00535 & 0.01142 & -9.92 \\
0.000192 & -0.0000799 & -0.0129
\end{bmatrix}
$$

### 4.3 重投影精度

| 关键点 | 投影坐标 | 真实坐标 | 误差 (m) | 内点 |
|--------|----------|----------|----------|------|
| goal_left_bottom | (-3.53, 52.35) | (-3.66, 52.5) | 0.20 | ✓ |
| goal_right_bottom | (3.69, 52.53) | (3.66, 52.5) | 0.05 | ✓ |
| goal_left_top | - | (-3.66, 52.5) | - | ✗ |
| goal_right_top | - | (3.66, 52.5) | - | ✗ |
| field_corner_left | (-33.98, 52.69) | (-34, 52.5) | 0.19 | ✓ |
| penalty_spot | (0.03, 41.51) | (0, 41.5) | 0.15 | ✓ |
| penalty_area_left | (-20.21, 35.95) | (-20.16, 36) | 0.07 | ✓ |
| penalty_area_right | (20.11, 36.04) | (20.16, 36) | 0.07 | ✓ |

**地面点平均重投影误差：约 0.12 m**（优秀，< 0.5 m 阈值）

---

## 5. 使用方法

### 步骤1：标注关键点

```bash
cd code/MODEL1/4-field-calib
D:/conda/envs/py312/python.exe annotate_keypoints.py
```

**交互操作**：
- **左键点击**：在图像上标注对应场地特征点
- **Z 键**：撤销上一个标注
- **S 键**：保存并退出
- **Q 键**：不保存退出

### 步骤2：计算单应性矩阵

```bash
D:/conda/envs/py312/python.exe compute_homography.py
```

输出内容：
- 单应性矩阵 $\mathbf{H}$ 及其逆矩阵 $\mathbf{H}^{-1}$
- 每个点的重投影误差
- 平均重投影误差
- 可视化验证图

### 步骤3：验证标定结果

检查 `src/img/vis/calibration/calibration_result.png`：
- 左图：原图上的标注点（绿色）和重投影点（红色）应重合
- 右图：场地俯视图上的标定点分布

---

## 6. 输出格式

### keypoints.json
```json
{
  "image_path": "src/img/frames/000001.jpg",
  "image_size": [1920, 1080],
  "coordinate_system": {
    "origin": "field_center",
    "x_axis": "parallel to goal line, right positive",
    "y_axis": "perpendicular to goal line, toward goal positive",
    "z_axis": "vertical up",
    "unit": "meters"
  },
  "keypoints": [
    {"name": "goal_left_bottom", "pixel": [1312, 523], "field": [-3.66, 52.5, 0]},
    ...
  ],
  "timestamp": "2026-01-22T20:25:17"
}
```

### homography.json
```json
{
  "H": [[-0.5644, -7.1969, 4728.24], [-1.0999, -0.3085, -1344.68], [-0.00157, -0.1050, 1.0]],
  "H_inv": [[-0.2356, -0.8147, 18.54], [0.00535, 0.01142, -9.92], [0.000192, -0.0000799, -0.0129]],
  "source_points_pixel": [[1312, 523], [1531, 570], ...],
  "target_points_field": [[-3.66, 52.5], [3.66, 52.5], ...],
  "reprojection_error_m": 0.12,
  "num_points": 8,
  "inlier_mask": [1, 1, 0, 0, 1, 1, 1, 1],
  "method": "cv2.findHomography (RANSAC)",
  "timestamp": "2026-01-22T20:25:17"
}
```

---

## 7. 配置参数

参见 `config/config.yaml` 中的 `calibration` 部分：

```yaml
calibration:
  field:
    length: 105.0      # 场地长度 (m)
    width: 68.0        # 场地宽度 (m)
  goal:
    width: 7.32        # 球门宽度 (m)
    height: 2.44       # 球门高度 (m)
  penalty_area:
    width: 40.32       # 大禁区宽度 (m)
    depth: 16.5        # 大禁区深度 (m)
  method: "homography"
```

---

## 8. 局限性与注意事项

### 8.1 单应性变换的局限性

单应性变换假设所有点位于**同一平面**（地面 $z=0$），因此：

| 场景 | 准确性 | 说明 |
|------|--------|------|
| 地面轨迹 | ✅ 准确 | 球在地面滚动或刚踢出时 |
| 空中轨迹 | ⚠️ 需补偿 | 需要额外的高度信息 |

对于足球飞行轨迹，本模块仅提供地面投影坐标 $(x, y)$，3D 重建（含高度 $z$）在步骤5完成。

### 8.2 标注建议

1. **选择清晰帧**：确保场地线条清晰可见
2. **点分布均匀**：覆盖足球运动区域，避免集中在一侧
3. **优先选择硬边缘**：球门柱、禁区角点等精确位置
4. **最少4点**：理论最低要求，建议6-8点以提高精度
5. **检查重投影误差**：地面点平均误差应小于 0.5m

### 8.3 球门顶部点的特殊处理

球门顶部点 $z = 2.44$ m，不在地面平面上，不能用于地面单应性估计。但可用于：
- 相机内参估计（如需要）
- 3D-2D 对应约束（PnP 问题）

本模块仅使用地面点（$z = 0$）计算单应性矩阵，球门顶部点自动被 RANSAC 排除。

---

## 9. 参考文献

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press. (Chapter 4: Estimation - 2D Projective Transformations)

2. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381-395.

3. OpenCV Documentation: [cv2.findHomography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)
