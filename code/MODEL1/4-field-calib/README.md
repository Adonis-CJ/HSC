# 4-field-calib: 场地标定模块

## 功能说明
该模块负责建立像素坐标到场地物理坐标的映射关系，包括：
- 场地线/角点/球门检测
- 单应性矩阵估计
- 尺度标定

## 输入
- 参考帧图像：`src/img/frames/` 中选取的关键帧
- 手动标注点（可选）：`output/calibration/keypoints.json`

## 输出
- 单应性矩阵：`output/calibration/homography.json`
- 相机参数（可选）：`output/calibration/camera_params.json`
- 标定可视化：`src/img/vis/calibration/`

## 场地坐标系定义

根据题目要求，建立如下坐标系：

```
      Y (边线方向，指向对方球门)
      ↑
      │
      │
      O ─────→ X (底线方向)
   (原点：进攻方向底线与边线交点)
   
   Z 轴：竖直向上
```

## 标准场地尺寸参考

| 要素 | 尺寸 (m) |
|------|----------|
| 场地 | 105 × 68 |
| 球门 | 7.32 × 2.44 |
| 罚球点距球门线 | 11 |
| 罚球弧/中圈半径 | 9.15 |
| 大禁区 | 40.32 × 16.5 |
| 小禁区 | 18.32 × 5.5 |
| 角球弧半径 | 1 |

## 输出格式

### homography.json
```json
{
  "H": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]],
  "source_points_pixel": [[u1,v1], [u2,v2], ...],
  "target_points_field": [[x1,y1], [x2,y2], ...],
  "reprojection_error": 2.34,
  "timestamp": "2026-01-22T10:00:00"
}
```

## 标定方法

### 1. 自动检测场地线
```python
python detect_field_lines.py --input src/img/frames/000001.jpg --output output/calibration/
```

### 2. 手动标注关键点
```python
python annotate_keypoints.py --input src/img/frames/000001.jpg --output output/calibration/keypoints.json
```

### 3. 计算单应性
```python
python compute_homography.py --keypoints output/calibration/keypoints.json --output output/calibration/homography.json
```

## 配置参数
参见 `config/config.yaml` 中的 `calibration` 部分。

## 注意事项
1. 选取足球运动范围内可见的场地特征
2. 尽量使用分布均匀的标定点（≥4 点）
3. 球门/禁区线是重要的几何约束
4. 保存原始标注以便复查和调整
