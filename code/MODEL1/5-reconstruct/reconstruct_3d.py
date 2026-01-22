#!/usr/bin/env python3
"""
3D轨迹重建模块

功能：
1. 将2D像素轨迹通过单应性矩阵转换为地面坐标 (X, Y)
2. 使用抛物线物理模型估计高度 Z
3. 输出完整的3D轨迹 (X, Y, Z, t)

数学原理：
- 地面坐标：通过单应性变换 H 将像素 (u, v) 映射到场地 (X, Y)
- 高度估计：假设足球做斜抛运动，结合初速度和重力加速度估计 Z(t)

坐标系：
- 原点：场地中心
- X轴：平行于球门线，向右为正
- Y轴：垂直于球门线，向进攻方向球门为正
- Z轴：竖直向上
- 单位：米
"""

import numpy as np
import pandas as pd
import json
import yaml
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
from datetime import datetime

# 物理常数
G = 9.81  # 重力加速度 m/s²


def load_homography(homography_path: Path) -> np.ndarray:
    """加载单应性矩阵"""
    with open(homography_path) as f:
        data = json.load(f)
    return np.array(data['H'])


def pixel_to_field(u: float, v: float, H: np.ndarray) -> tuple:
    """
    像素坐标转场地坐标（地面投影）
    
    Args:
        u, v: 像素坐标
        H: 单应性矩阵
    
    Returns:
        (X, Y): 场地坐标 (米)
    """
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def load_keypoints(keypoints_path: Path) -> dict:
    """加载标定关键点"""
    with open(keypoints_path) as f:
        return json.load(f)


def project_trajectory_to_ground(trajectory_df: pd.DataFrame, H: np.ndarray, 
                                  keypoints: dict = None) -> pd.DataFrame:
    """
    将整条轨迹投影到地面坐标系
    
    策略：
    1. Y方向使用单应性结果（透视导致的误差较小）
    2. X方向使用球门标定点建立直接线性映射（最可靠）
    
    球门标定点分析：
    - goal_left_bottom: u=1312 → X=-3.66m
    - goal_right_bottom: u=1531 → X=+3.66m
    - 这给出了精确的 dX/du = 7.32/(1531-1312) = 0.0334 m/pixel
    """
    result = trajectory_df.copy()
    
    # 计算单应性变换结果
    X_raw, Y_raw = [], []
    for _, row in trajectory_df.iterrows():
        u, v = row['u'], row['v']
        X, Y = pixel_to_field(u, v, H)
        X_raw.append(X)
        Y_raw.append(Y)
    
    result['X_raw'] = X_raw
    result['Y_raw'] = Y_raw
    
    u_vals = trajectory_df['u'].values
    t_vals = trajectory_df['t'].values
    
    # Y方向：使用单应性结果
    Y_corrected = np.array(Y_raw)
    
    # X方向：使用球门标定点的精确线性映射
    X_corrected = None
    
    if keypoints:
        # 提取球门底部的标定点（最可靠的参考）
        goal_left = None
        goal_right = None
        
        for kp in keypoints.get('keypoints', []):
            if kp['name'] == 'goal_left_bottom':
                goal_left = {'u': kp['pixel'][0], 'X': kp['field'][0]}
            elif kp['name'] == 'goal_right_bottom':
                goal_right = {'u': kp['pixel'][0], 'X': kp['field'][0]}
        
        if goal_left and goal_right:
            # 球门线性映射
            u_left, X_left = goal_left['u'], goal_left['X']
            u_right, X_right = goal_right['u'], goal_right['X']
            
            # dX/du = (X_right - X_left) / (u_right - u_left)
            dX_du = (X_right - X_left) / (u_right - u_left)
            
            print(f"  - 球门标定: 左柱(u={u_left}, X={X_left}), 右柱(u={u_right}, X={X_right})")
            print(f"  - 球门区域 dX/du = {dX_du:.5f} m/pixel")
            
            # 使用球门中心作为参考点
            u_center = (u_left + u_right) / 2
            X_center = (X_left + X_right) / 2  # 应该是0
            
            print(f"  - 球门中心: u={u_center:.0f}, X={X_center:.2f}")
            
            # 计算球的像素轨迹对应的X
            # X = X_center + dX_du * (u - u_center)
            X_corrected = X_center + dX_du * (u_vals - u_center)
            
            u_start, u_end = u_vals[0], u_vals[-1]
            X_start, X_end = X_corrected[0], X_corrected[-1]
            
            print(f"  - 球轨迹: u从{u_start:.0f}到{u_end:.0f}")
            print(f"  - 对应X: 从{X_start:.2f}m 到 {X_end:.2f}m")
            
            # 验证方向：u增加时X应该增加（因为dX_du > 0）
            if dX_du > 0:
                print(f"  - ✓ 方向正确: 图像右侧 = 场地右侧")
            else:
                print(f"  - ✗ 方向异常: 图像右侧 = 场地左侧")
    
    if X_corrected is None:
        print("  - 警告：无球门标定点，使用原始单应性结果")
        X_corrected = np.array(X_raw)
    
    result['X'] = X_corrected
    result['Y'] = Y_corrected
    
    return result


def estimate_initial_velocity(X: np.ndarray, Y: np.ndarray, t: np.ndarray) -> dict:
    """
    估计初始水平速度分量
    
    对于足球自由球，假设水平方向（X, Y 分量）近似匀速运动（忽略空气阻力）
    
    Returns:
        dict: {'vx0': float, 'vy0': float, 'v_horizontal': float}
    """
    # 使用多项式拟合求导数估计速度
    # X(t) ≈ X0 + vx0 * t
    # Y(t) ≈ Y0 + vy0 * t
    
    # 线性拟合
    coeffs_x = np.polyfit(t, X, 1)  # [vx0, X0]
    coeffs_y = np.polyfit(t, Y, 1)  # [vy0, Y0]
    
    vx0 = coeffs_x[0]
    vy0 = coeffs_y[0]
    v_horizontal = np.sqrt(vx0**2 + vy0**2)
    
    return {
        'vx0': vx0,
        'vy0': vy0,
        'v_horizontal': v_horizontal,
        'X0': coeffs_x[1],
        'Y0': coeffs_y[1]
    }


def estimate_height_projectile(t: np.ndarray, z0: float, vz0: float, g: float = G) -> np.ndarray:
    """
    抛物线运动模型计算高度
    
    Z(t) = z0 + vz0 * t - 0.5 * g * t²
    
    Args:
        t: 时间序列
        z0: 初始高度 (m)
        vz0: 初始垂直速度 (m/s)
        g: 重力加速度 (m/s²)
    
    Returns:
        Z: 高度序列
    """
    return z0 + vz0 * t - 0.5 * g * t**2


def fit_projectile_model(t: np.ndarray, pixel_v: np.ndarray, 
                          field_Y: np.ndarray, H: np.ndarray,
                          goal_y: float = 52.5, goal_height: float = 2.44) -> dict:
    """
    拟合抛物线模型，估计初始垂直速度
    
    核心思想：
    1. 图像中球的垂直像素坐标 v 与球的真实高度 Z 相关
    2. 球越高，v 越小（图像坐标系 v 向下为正）
    3. 结合场地 Y 坐标和物理约束，反推高度
    
    约束条件：
    - 球从脚踢出，初始高度 z0 ≈ 0.22m（足球半径）
    - 飞向球门时，终点高度应在 0 ~ 2.44m 之间
    
    Args:
        t: 时间序列
        pixel_v: 像素 v 坐标序列
        field_Y: 场地 Y 坐标序列
        H: 单应性矩阵
        goal_y: 球门线 Y 坐标
        goal_height: 球门高度
    
    Returns:
        dict: 拟合参数和高度序列
    """
    # 方法：基于视觉透视约束估计高度
    # 
    # 观察：当球飞高时，它在图像中的位置会"上移"（v减小）
    # 但由于透视效应，远处的高度变化在图像中的位移更小
    #
    # 简化模型：假设相机水平观察，球的像素 v 坐标与高度近似线性关系
    # v = v_ground - k * Z
    # 其中 v_ground 是球在地面时的 v 坐标，k 是比例系数
    
    # 使用球门柱作为标定参考
    # 球门左底部 (1312, 523)，球门左顶部 (1315, 408)
    # 高度差 2.44m 对应 v 差 523 - 408 = 115 像素
    # 比例系数 k ≈ 115 / 2.44 ≈ 47.1 像素/米
    
    v_goal_bottom = 523  # 球门底部 v 坐标
    v_goal_top = 408     # 球门顶部 v 坐标
    goal_height_m = 2.44
    
    k_pixels_per_meter = (v_goal_bottom - v_goal_top) / goal_height_m
    
    # 估计每个点的地面投影 v 坐标
    # 如果球在地面，它的 v 应该等于地面投影的 v
    # 实际 v 比地面投影小，说明球在空中
    
    # 首先需要估计"如果球在地面"时的 v 坐标
    # 这需要逆向使用单应性：给定 (X, Y, Z=0)，求像素 (u, v)
    H_inv = np.linalg.inv(H)
    
    # 但是这里有个问题：我们已经用 H 从 (u,v) 得到了 (X,Y)
    # 这个 (X,Y) 是假设 Z=0 的结果
    # 如果球实际在空中，真实的 (X,Y) 会略有偏移
    
    # 简化方法：使用抛物线约束
    # 假设 Z(t) = z0 + vz0*t - 0.5*g*t²
    # 边界条件：
    # 1. Z(0) = z0 ≈ 0.22m（球从地面踢出）
    # 2. Z(t) >= 0（球不能在地下）
    # 3. 抛物线形状
    
    # 使用轨迹的 v 坐标变化来约束 vz0
    # 球在最高点时 v 最小，此时 dZ/dt = 0，即 vz0 = g*t_max
    
    # 找到 v 的最小值点（轨迹最高点）
    v_min_idx = np.argmin(pixel_v)
    t_apex = t[v_min_idx]
    
    # 初始高度（足球半径 + 脚踢高度）
    z0 = 0.22  # 米
    
    # 如果最高点在轨迹中间，可以估计 vz0
    if 0 < v_min_idx < len(t) - 1:
        # 最高点时刻 t_apex，此时 vz0 - g*t_apex = 0
        # vz0 = g * t_apex
        vz0_estimate = G * t_apex
    else:
        # 如果最高点在端点，使用其他约束
        # 假设球最终到达球门位置，高度约 1-2m
        # Z(T) = z0 + vz0*T - 0.5*g*T² = Z_end
        # 估计 Z_end
        T = t[-1]
        Z_end_estimate = 1.5  # 假设终点高度约1.5m
        # vz0 = (Z_end - z0 + 0.5*g*T²) / T
        vz0_estimate = (Z_end_estimate - z0 + 0.5 * G * T**2) / T
    
    # 使用优化方法微调 vz0
    def objective(params):
        vz0 = params[0]
        Z = estimate_height_projectile(t, z0, vz0, G)
        
        # 约束1：高度非负
        penalty = np.sum(np.maximum(-Z, 0) ** 2) * 1000
        
        # 约束2：终点高度合理（0 ~ 2.44m）
        Z_end = Z[-1]
        if Z_end < 0:
            penalty += (Z_end) ** 2 * 100
        elif Z_end > goal_height:
            penalty += (Z_end - goal_height) ** 2 * 10
        
        # 约束3：最大高度不超过合理范围（如 5m）
        Z_max = np.max(Z)
        if Z_max > 5:
            penalty += (Z_max - 5) ** 2 * 10
        
        # 约束4：抛物线形状应该与 v 坐标变化一致
        # v 最小处应该对应 Z 最大处
        Z_max_idx = np.argmax(Z)
        idx_diff = abs(Z_max_idx - v_min_idx)
        penalty += idx_diff ** 2 * 0.1
        
        return penalty
    
    result = minimize(objective, [vz0_estimate], method='Nelder-Mead')
    vz0_optimal = result.x[0]
    
    # 计算最终高度序列
    Z = estimate_height_projectile(t, z0, vz0_optimal, G)
    
    # 确保 Z >= 0
    Z = np.maximum(Z, 0)
    
    # 计算最大高度和达到时刻
    Z_max = np.max(Z)
    t_max = t[np.argmax(Z)]
    
    return {
        'z0': z0,
        'vz0': vz0_optimal,
        'Z': Z,
        'Z_max': Z_max,
        't_apex': t_max,
        'Z_end': Z[-1],
        'method': 'projectile_with_constraints'
    }


def compute_3d_velocity(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                        t: np.ndarray, smooth: bool = True) -> dict:
    """
    计算3D速度分量
    
    使用中心差分法计算速度，并对结果进行平滑处理
    
    Args:
        X, Y, Z: 位置序列
        t: 时间序列
        smooth: 是否平滑
    
    Returns:
        dict: 包含 vx, vy, vz, v_total 速度序列和初始速度
    """
    n = len(t)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vz = np.zeros(n)
    
    # 使用中心差分（对内部点）
    for i in range(1, n - 1):
        dt = t[i + 1] - t[i - 1]
        vx[i] = (X[i + 1] - X[i - 1]) / dt
        vy[i] = (Y[i + 1] - Y[i - 1]) / dt
        vz[i] = (Z[i + 1] - Z[i - 1]) / dt
    
    # 端点使用前向/后向差分
    dt_start = t[1] - t[0]
    vx[0] = (X[1] - X[0]) / dt_start
    vy[0] = (Y[1] - Y[0]) / dt_start
    vz[0] = (Z[1] - Z[0]) / dt_start
    
    dt_end = t[-1] - t[-2]
    vx[-1] = (X[-1] - X[-2]) / dt_end
    vy[-1] = (Y[-1] - Y[-2]) / dt_end
    vz[-1] = (Z[-1] - Z[-2]) / dt_end
    
    # 平滑处理
    if smooth and len(vx) >= 7:
        window = 7
        vx = savgol_filter(vx, window, 2)
        vy = savgol_filter(vy, window, 2)
        vz = savgol_filter(vz, window, 2)
    
    # 总速度
    v_total = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # 使用前5帧的平均速度作为初始速度估计（更稳健）
    n_avg = min(5, n)
    vx0_avg = np.mean(vx[:n_avg])
    vy0_avg = np.mean(vy[:n_avg])
    vz0_avg = np.mean(vz[:n_avg])
    v0_avg = np.sqrt(vx0_avg**2 + vy0_avg**2 + vz0_avg**2)
    
    return {
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'v_total': v_total,
        'v0': v0_avg,
        'vx0': vx0_avg,
        'vy0': vy0_avg,
        'vz0': vz0_avg
    }


def estimate_initial_angles(vx0: float, vy0: float, vz0: float) -> dict:
    """
    计算初始发射角度
    
    Args:
        vx0, vy0, vz0: 初始速度分量
    
    Returns:
        dict: 包含仰角和方位角
    """
    v_horizontal = np.sqrt(vx0**2 + vy0**2)
    v_total = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    
    # 仰角：与水平面的夹角
    elevation_angle = np.degrees(np.arctan2(vz0, v_horizontal))
    
    # 方位角：在水平面内与Y轴（指向球门）的夹角
    # 正值表示偏右，负值表示偏左
    azimuth_angle = np.degrees(np.arctan2(vx0, vy0))
    
    return {
        'elevation_angle_deg': elevation_angle,
        'azimuth_angle_deg': azimuth_angle,
        'v_horizontal': v_horizontal,
        'v_total': v_total
    }


def visualize_3d_trajectory(df: pd.DataFrame, output_path: Path, 
                            velocity_info: dict, angle_info: dict):
    """生成3D轨迹可视化"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(df['X'], df['Y'], df['Z'], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(df['X'].iloc[0], df['Y'].iloc[0], df['Z'].iloc[0], 
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(df['X'].iloc[-1], df['Y'].iloc[-1], df['Z'].iloc[-1], 
                c='red', s=100, marker='X', label='End')
    
    # 绘制球门
    goal_x = np.array([-3.66, 3.66, 3.66, -3.66, -3.66])
    goal_y = np.array([52.5, 52.5, 52.5, 52.5, 52.5])
    goal_z = np.array([0, 0, 2.44, 2.44, 0])
    ax1.plot(goal_x, goal_y, goal_z, 'k-', linewidth=2, label='Goal')
    
    ax1.set_xlabel('X (m) - Right positive')
    ax1.set_ylabel('Y (m) - Toward goal')
    ax1.set_zlabel('Z (m) - Up')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 2. XY平面俯视图
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df['X'], df['Y'], 'b-', linewidth=2)
    ax2.scatter(df['X'].iloc[0], df['Y'].iloc[0], c='green', s=100, marker='o', label='Start')
    ax2.scatter(df['X'].iloc[-1], df['Y'].iloc[-1], c='red', s=100, marker='X', label='End')
    
    # 绘制球门和禁区
    ax2.plot([-3.66, 3.66], [52.5, 52.5], 'k-', linewidth=3, label='Goal line')
    ax2.plot([-20.16, 20.16, 20.16, -20.16, -20.16], 
             [52.5, 52.5, 36, 36, 52.5], 'g--', linewidth=1, label='Penalty area')
    ax2.scatter([0], [41.5], c='orange', s=50, marker='o', label='Penalty spot')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY Plane)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. YZ平面侧视图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(df['Y'], df['Z'], 'b-', linewidth=2)
    ax3.scatter(df['Y'].iloc[0], df['Z'].iloc[0], c='green', s=100, marker='o', label='Start')
    ax3.scatter(df['Y'].iloc[-1], df['Z'].iloc[-1], c='red', s=100, marker='X', label='End')
    
    # 绘制球门
    ax3.plot([52.5, 52.5], [0, 2.44], 'k-', linewidth=3, label='Goal')
    ax3.axhline(y=0, color='brown', linestyle='-', linewidth=1, label='Ground')
    
    ax3.set_xlabel('Y (m) - Distance to goal')
    ax3.set_ylabel('Z (m) - Height')
    ax3.set_title('Side View (YZ Plane) - Trajectory Arc')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=-0.5)
    
    # 4. 速度变化图
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(df['t'], velocity_info['v_total'], 'b-', linewidth=2, label='Total velocity')
    ax4.plot(df['t'], np.sqrt(velocity_info['vx']**2 + velocity_info['vy']**2), 
             'g--', linewidth=1.5, label='Horizontal velocity')
    ax4.plot(df['t'], velocity_info['vz'], 'r:', linewidth=1.5, label='Vertical velocity')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加信息文本
    info_text = (
        f"Initial velocity: {velocity_info['v0']:.2f} m/s\n"
        f"Elevation angle: {angle_info['elevation_angle_deg']:.1f}°\n"
        f"Azimuth angle: {angle_info['azimuth_angle_deg']:.1f}°\n"
        f"Max height: {df['Z'].max():.2f} m\n"
        f"Flight time: {df['t'].iloc[-1]:.3f} s\n"
        f"Distance: {np.sqrt((df['X'].iloc[-1]-df['X'].iloc[0])**2 + (df['Y'].iloc[-1]-df['Y'].iloc[0])**2):.2f} m"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='3D轨迹重建')
    parser.add_argument('--trajectory', type=str, 
                        default='../../../output/trajectory/2d/trajectory_2d_clean.csv',
                        help='清洗后的2D轨迹文件')
    parser.add_argument('--homography', type=str,
                        default='../../../output/calibration/homography.json',
                        help='单应性矩阵文件')
    parser.add_argument('--keypoints', type=str,
                        default='../../../output/calibration/keypoints.json',
                        help='标定关键点文件')
    parser.add_argument('--output-ground', type=str,
                        default='../../../output/trajectory/3d/trajectory_ground.csv',
                        help='地面投影轨迹输出')
    parser.add_argument('--output-3d', type=str,
                        default='../../../output/trajectory/3d/trajectory_3d.csv',
                        help='3D轨迹输出')
    parser.add_argument('--vis-dir', type=str,
                        default='../../../src/img/vis/reconstruction',
                        help='可视化输出目录')
    args = parser.parse_args()
    
    # 解析路径
    script_dir = Path(__file__).parent
    trajectory_path = (script_dir / args.trajectory).resolve()
    homography_path = (script_dir / args.homography).resolve()
    keypoints_path = (script_dir / args.keypoints).resolve()
    output_ground_path = (script_dir / args.output_ground).resolve()
    output_3d_path = (script_dir / args.output_3d).resolve()
    vis_dir = (script_dir / args.vis_dir).resolve()
    
    print("=" * 60)
    print("3D轨迹重建模块")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    trajectory_df = pd.read_csv(trajectory_path)
    H = load_homography(homography_path)
    keypoints = load_keypoints(keypoints_path)
    print(f"  - 轨迹点数: {len(trajectory_df)}")
    print(f"  - 时间范围: {trajectory_df['t'].iloc[0]:.4f} ~ {trajectory_df['t'].iloc[-1]:.4f} s")
    print(f"  - 标定点数: {len(keypoints.get('keypoints', []))}")
    
    # 2. 地面投影（带透视修正）
    print("\n[2/5] 像素坐标 → 场地坐标（带透视修正）...")
    ground_df = project_trajectory_to_ground(trajectory_df, H, keypoints)
    print(f"  - X范围 (修正后): {ground_df['X'].min():.2f} ~ {ground_df['X'].max():.2f} m")
    print(f"  - Y范围: {ground_df['Y'].min():.2f} ~ {ground_df['Y'].max():.2f} m")
    
    # 保存地面投影结果
    output_ground_path.parent.mkdir(parents=True, exist_ok=True)
    ground_df[['frame', 't', 'X', 'Y', 'u', 'v']].to_csv(output_ground_path, index=False)
    print(f"  - 已保存: {output_ground_path}")
    
    # 3. 高度估计（抛物线模型）
    print("\n[3/5] 估计高度（抛物线物理模型）...")
    t = ground_df['t'].values
    pixel_v = ground_df['v'].values
    field_Y = ground_df['Y'].values
    
    height_result = fit_projectile_model(t, pixel_v, field_Y, H)
    Z = height_result['Z']
    
    print(f"  - 初始高度 z0: {height_result['z0']:.2f} m")
    print(f"  - 初始垂直速度 vz0: {height_result['vz0']:.2f} m/s")
    print(f"  - 最大高度: {height_result['Z_max']:.2f} m (t = {height_result['t_apex']:.3f} s)")
    print(f"  - 终点高度: {height_result['Z_end']:.2f} m")
    
    # 4. 计算速度和角度
    print("\n[4/5] 计算速度和发射角度...")
    X = ground_df['X'].values
    Y = ground_df['Y'].values
    
    velocity_info = compute_3d_velocity(X, Y, Z, t)
    angle_info = estimate_initial_angles(velocity_info['vx0'], 
                                          velocity_info['vy0'], 
                                          velocity_info['vz0'])
    
    print(f"  - 初始速度: {velocity_info['v0']:.2f} m/s")
    print(f"    - vx0: {velocity_info['vx0']:.2f} m/s")
    print(f"    - vy0: {velocity_info['vy0']:.2f} m/s")
    print(f"    - vz0: {velocity_info['vz0']:.2f} m/s")
    print(f"  - 仰角: {angle_info['elevation_angle_deg']:.1f}°")
    print(f"  - 方位角: {angle_info['azimuth_angle_deg']:.1f}° (正=偏右, 负=偏左)")
    
    # 5. 生成输出
    print("\n[5/5] 生成输出文件...")
    
    # 构建3D轨迹DataFrame
    trajectory_3d = pd.DataFrame({
        'frame': ground_df['frame'],
        't': t,
        'X': X,
        'Y': Y,
        'Z': Z,
        'u': ground_df['u'],
        'v': ground_df['v'],
        'vx': velocity_info['vx'],
        'vy': velocity_info['vy'],
        'vz': velocity_info['vz'],
        'v_total': velocity_info['v_total']
    })
    
    # 保存3D轨迹
    output_3d_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_3d.to_csv(output_3d_path, index=False)
    print(f"  - 3D轨迹已保存: {output_3d_path}")
    
    # 保存重建参数
    params = {
        'timestamp': datetime.now().isoformat(),
        'input_files': {
            'trajectory_2d': str(trajectory_path),
            'homography': str(homography_path)
        },
        'output_files': {
            'trajectory_ground': str(output_ground_path),
            'trajectory_3d': str(output_3d_path)
        },
        'projectile_model': {
            'z0': float(height_result['z0']),
            'vz0': float(height_result['vz0']),
            'g': G,
            'method': height_result['method']
        },
        'initial_velocity': {
            'v0': float(velocity_info['v0']),
            'vx0': float(velocity_info['vx0']),
            'vy0': float(velocity_info['vy0']),
            'vz0': float(velocity_info['vz0'])
        },
        'launch_angles': {
            'elevation_deg': float(angle_info['elevation_angle_deg']),
            'azimuth_deg': float(angle_info['azimuth_angle_deg'])
        },
        'trajectory_stats': {
            'num_points': len(trajectory_3d),
            'duration_s': float(t[-1] - t[0]),
            'max_height_m': float(np.max(Z)),
            'horizontal_distance_m': float(np.sqrt((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2)),
            'start_position': [float(X[0]), float(Y[0]), float(Z[0])],
            'end_position': [float(X[-1]), float(Y[-1]), float(Z[-1])]
        }
    }
    
    params_path = output_3d_path.parent / 'reconstruct_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  - 参数已保存: {params_path}")
    
    # 生成可视化
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualize_3d_trajectory(trajectory_3d, vis_dir / 'trajectory_3d.png',
                            velocity_info, angle_info)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("重建完成！摘要：")
    print("=" * 60)
    print(f"  轨迹点数: {len(trajectory_3d)}")
    print(f"  飞行时间: {t[-1] - t[0]:.3f} s")
    print(f"  水平距离: {np.sqrt((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2):.2f} m")
    print(f"  最大高度: {np.max(Z):.2f} m")
    print(f"  初始速度: {velocity_info['v0']:.2f} m/s ({velocity_info['v0']*3.6:.1f} km/h)")
    print(f"  发射角度: 仰角 {angle_info['elevation_angle_deg']:.1f}°, 方位角 {angle_info['azimuth_angle_deg']:.1f}°")
    print(f"  起点: ({X[0]:.2f}, {Y[0]:.2f}, {Z[0]:.2f})")
    print(f"  终点: ({X[-1]:.2f}, {Y[-1]:.2f}, {Z[-1]:.2f})")


if __name__ == '__main__':
    main()
