# 标定数据输出目录

该目录存放场地标定相关文件。

## 文件说明

| 文件 | 说明 | 状态 |
|------|------|------|
| `keypoints.json` | 8个场地关键点的像素坐标与物理坐标对应关系 | ✅ 已生成 |
| `homography.json` | 3×3单应性矩阵及其逆矩阵 | ✅ 已生成 |

## 标定结果摘要

- **标定日期**: 2026-01-22
- **使用关键点**: 8个（球门柱4个 + 场地角点1个 + 罚球点1个 + 禁区角点2个）
- **有效内点**: 6个（球门顶部2点被排除，因 $z \neq 0$）
- **地面点平均重投影误差**: ~0.12m（优秀）
- **可视化**: `src/img/vis/calibration/calibration_result.png`

## 坐标变换示例

```python
import json
import numpy as np

# 加载单应性矩阵
with open('homography.json') as f:
    data = json.load(f)
H = np.array(data['H'])

# 像素坐标 → 场地坐标
def pixel_to_field(u, v):
    pt = np.array([u, v, 1.0])
    result = H @ pt
    return result[0] / result[2], result[1] / result[2]

# 示例：球的像素位置 (1000, 600) 对应场地坐标
x, y = pixel_to_field(1000, 600)
print(f"场地坐标: ({x:.2f}, {y:.2f}) 米")
```

