#!/usr/bin/env python3
"""
场地标定模块 - 计算单应性矩阵
根据标注的对应点计算像素坐标到场地坐标的映射
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime


def compute_homography(pixel_points: np.ndarray, field_points: np.ndarray):
    """
    计算单应性矩阵
    
    使用直接线性变换 (DLT) 算法求解：
    对于每对对应点 (u, v) <-> (x, y)，有约束：
    
    | -u  -v  -1   0   0   0   u*x  v*x  x |   |h1|
    |  0   0   0  -u  -v  -1   u*y  v*y  y | * |h2| = 0
                                              |..|
                                              |h9|
    
    通过 SVD 分解求解最小二乘解
    
    Args:
        pixel_points: Nx2 像素坐标数组 [[u1,v1], [u2,v2], ...]
        field_points: Nx2 场地坐标数组 [[x1,y1], [x2,y2], ...]
    
    Returns:
        H: 3x3 单应性矩阵 (像素 -> 场地)
        H_inv: 3x3 逆单应性矩阵 (场地 -> 像素)
        error: 重投影误差
    """
    assert len(pixel_points) >= 4, "至少需要4对对应点"
    assert len(pixel_points) == len(field_points), "点数必须一致"
    
    # 使用 OpenCV 的 findHomography，采用 RANSAC 提高鲁棒性
    H, mask = cv2.findHomography(pixel_points, field_points, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("无法计算单应性矩阵")
    
    # 计算逆矩阵
    H_inv = np.linalg.inv(H)
    
    # 计算重投影误差
    error = compute_reprojection_error(pixel_points, field_points, H)
    
    return H, H_inv, error, mask


def compute_reprojection_error(pixel_points: np.ndarray, field_points: np.ndarray, H: np.ndarray):
    """
    计算重投影误差
    
    误差定义：将像素点通过H变换到场地坐标，与真实场地坐标的欧氏距离
    
    Args:
        pixel_points: Nx2 像素坐标
        field_points: Nx2 场地坐标
        H: 3x3 单应性矩阵
    
    Returns:
        mean_error: 平均重投影误差 (米)
    """
    n = len(pixel_points)
    errors = []
    
    for i in range(n):
        u, v = pixel_points[i]
        x_true, y_true = field_points[i]
        
        # 单应性变换
        p = H @ np.array([u, v, 1])
        x_pred, y_pred = p[0] / p[2], p[1] / p[2]
        
        # 欧氏距离误差
        error = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
        errors.append(error)
    
    return np.mean(errors)


def transform_pixel_to_field(pixel_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    将像素坐标转换为场地坐标
    
    Args:
        pixel_points: Nx2 像素坐标 [[u, v], ...]
        H: 3x3 单应性矩阵
    
    Returns:
        field_points: Nx2 场地坐标 [[x, y], ...]
    """
    pixel_points = np.asarray(pixel_points)
    if pixel_points.ndim == 1:
        pixel_points = pixel_points.reshape(1, -1)
    
    n = len(pixel_points)
    field_points = np.zeros((n, 2))
    
    for i in range(n):
        u, v = pixel_points[i]
        p = H @ np.array([u, v, 1])
        field_points[i] = [p[0] / p[2], p[1] / p[2]]
    
    return field_points


def transform_field_to_pixel(field_points: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    将场地坐标转换为像素坐标
    
    Args:
        field_points: Nx2 场地坐标 [[x, y], ...]
        H_inv: 3x3 逆单应性矩阵
    
    Returns:
        pixel_points: Nx2 像素坐标 [[u, v], ...]
    """
    field_points = np.asarray(field_points)
    if field_points.ndim == 1:
        field_points = field_points.reshape(1, -1)
    
    n = len(field_points)
    pixel_points = np.zeros((n, 2))
    
    for i in range(n):
        x, y = field_points[i]
        p = H_inv @ np.array([x, y, 1])
        pixel_points[i] = [p[0] / p[2], p[1] / p[2]]
    
    return pixel_points


def visualize_calibration(image_path: str, keypoints: list, H: np.ndarray, 
                          H_inv: np.ndarray, output_path: str):
    """
    可视化标定结果
    """
    import matplotlib.pyplot as plt
    
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：原图 + 标定点 + 重投影点
    ax1 = axes[0]
    ax1.imshow(img)
    
    for kp in keypoints:
        u, v = kp['pixel']
        x, y = kp['field'][:2]
        name = kp['name']
        
        # 原始标注点 (绿色)
        ax1.scatter(u, v, c='lime', s=100, marker='o', edgecolors='white', linewidths=2, zorder=10)
        ax1.annotate(name.replace('_', '\n'), (u, v), textcoords='offset points', 
                    xytext=(10, -10), fontsize=8, color='yellow')
        
        # 重投影点 (红色)
        reproj = transform_field_to_pixel(np.array([[x, y]]), H_inv)[0]
        ax1.scatter(reproj[0], reproj[1], c='red', s=50, marker='x', linewidths=2, zorder=11)
    
    ax1.set_title('Image with Keypoints\n(green=annotated, red=reprojected)', fontsize=12)
    ax1.axis('off')
    
    # 右图：场地俯视图
    ax2 = axes[1]
    
    # 绘制场地线
    # 场地边界
    field_x = [-34, 34, 34, -34, -34]
    field_y = [52.5, 52.5, 0, 0, 52.5]  # 只画半场
    ax2.plot(field_x, field_y, 'w-', linewidth=2)
    
    # 大禁区
    penalty_x = [-20.16, 20.16, 20.16, -20.16, -20.16]
    penalty_y = [52.5, 52.5, 36, 36, 52.5]
    ax2.plot(penalty_x, penalty_y, 'w-', linewidth=2)
    
    # 小禁区
    goal_area_x = [-9.16, 9.16, 9.16, -9.16, -9.16]
    goal_area_y = [52.5, 52.5, 47, 47, 52.5]
    ax2.plot(goal_area_x, goal_area_y, 'w-', linewidth=2)
    
    # 球门
    goal_x = [-3.66, 3.66]
    goal_y = [52.5, 52.5]
    ax2.plot(goal_x, goal_y, 'yellow', linewidth=4)
    
    # 罚球点
    ax2.scatter(0, 41.5, c='white', s=50, marker='o')
    
    # 标定点
    for kp in keypoints:
        x, y = kp['field'][:2]
        ax2.scatter(x, y, c='lime', s=100, marker='o', edgecolors='white', linewidths=2)
        ax2.annotate(kp['name'].replace('_', '\n'), (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='cyan')
    
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(30, 55)
    ax2.set_aspect('equal')
    ax2.set_facecolor('#2E7D32')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Field View (top-down)', fontsize=12)
    ax2.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='计算单应性矩阵')
    parser.add_argument('--keypoints', type=str,
                        default='../../../output/calibration/keypoints.json',
                        help='标注点文件')
    parser.add_argument('--output', type=str,
                        default='../../../output/calibration/homography.json',
                        help='输出单应性矩阵文件')
    parser.add_argument('--vis', type=str,
                        default='../../../src/img/vis/calibration/calibration_result.png',
                        help='可视化输出路径')
    parser.add_argument('--no-vis', action='store_true',
                        help='禁用可视化')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    keypoints_path = (script_dir / args.keypoints).resolve()
    output_path = (script_dir / args.output).resolve()
    vis_path = (script_dir / args.vis).resolve()
    
    print("=" * 60)
    print("场地标定 - 计算单应性矩阵")
    print("=" * 60)
    
    # 读取标注点
    with open(keypoints_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    keypoints = data['keypoints']
    print(f"加载 {len(keypoints)} 个标定点")
    
    # 提取像素和场地坐标 (只使用 x, y，忽略 z)
    pixel_points = np.array([kp['pixel'] for kp in keypoints], dtype=np.float64)
    field_points = np.array([kp['field'][:2] for kp in keypoints], dtype=np.float64)
    
    print("\n标定点对应关系:")
    print("-" * 60)
    for kp in keypoints:
        print(f"  {kp['name']:25s} pixel{kp['pixel']} -> field{kp['field'][:2]}")
    print("-" * 60)
    
    # 计算单应性矩阵
    H, H_inv, error, mask = compute_homography(pixel_points, field_points)
    
    print(f"\n单应性矩阵 H (像素 -> 场地):")
    print(H)
    
    print(f"\n重投影误差: {error:.4f} m")
    
    # 验证：打印每个点的变换结果
    print("\n变换验证:")
    print("-" * 60)
    for i, kp in enumerate(keypoints):
        u, v = kp['pixel']
        x_true, y_true = kp['field'][:2]
        
        pred = transform_pixel_to_field(np.array([[u, v]]), H)[0]
        err = np.sqrt((pred[0] - x_true)**2 + (pred[1] - y_true)**2)
        
        status = "✓" if mask[i] else "✗ (outlier)"
        print(f"  {kp['name']:25s} ({u:6.1f}, {v:6.1f}) -> "
              f"({pred[0]:7.2f}, {pred[1]:7.2f}) vs ({x_true:7.2f}, {y_true:7.2f}) "
              f"err={err:.3f}m {status}")
    print("-" * 60)
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'H': H.tolist(),
        'H_inv': H_inv.tolist(),
        'source_points_pixel': pixel_points.tolist(),
        'target_points_field': field_points.tolist(),
        'reprojection_error_m': float(error),
        'num_points': len(keypoints),
        'inlier_mask': mask.flatten().tolist() if mask is not None else None,
        'method': 'cv2.findHomography (RANSAC)',
        'coordinate_system': data.get('coordinate_system', {}),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n单应性矩阵已保存: {output_path}")
    
    # 可视化
    if not args.no_vis:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = data.get('image_path', str(script_dir / '../../../src/img/frames/000001.jpg'))
        visualize_calibration(image_path, keypoints, H, H_inv, str(vis_path))


if __name__ == '__main__':
    main()
