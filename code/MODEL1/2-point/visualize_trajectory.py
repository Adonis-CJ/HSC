#!/usr/bin/env python3
"""
足球轨迹可视化工具
根据手动标注数据生成轨迹分析图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 读取标注数据
    data_path = Path(__file__).parent / '../../../output/trajectory/2d/trajectory_2d_manual.csv'
    df = pd.read_csv(data_path.resolve())
    
    print(f'标注帧数: {len(df)}')
    print(f'时间范围: {df["t"].min():.3f} - {df["t"].max():.3f} 秒')
    print(f'u坐标范围: {df["u"].min():.0f} - {df["u"].max():.0f} px')
    print(f'v坐标范围: {df["v"].min():.0f} - {df["v"].max():.0f} px')
    
    # 创建输出目录
    vis_dir = Path(__file__).parent / '../../../src/img/vis/detection'
    vis_dir = vis_dir.resolve()
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 图1: 综合分析图 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 2D轨迹图 (像素坐标系)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['u'], df['v'], c=df['t'], cmap='viridis', s=60, 
                          edgecolors='black', linewidths=0.5)
    ax1.plot(df['u'], df['v'], 'b-', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('u (像素)', fontsize=12)
    ax1.set_ylabel('v (像素)', fontsize=12)
    ax1.set_title('足球2D轨迹 (像素坐标系)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(1080, 0)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('时间 (秒)')
    
    # 子图2: u-t 图 (水平位置随时间变化)
    ax2 = axes[0, 1]
    ax2.plot(df['t'], df['u'], 'bo-', markersize=5, linewidth=1.5)
    ax2.set_xlabel('时间 t (秒)', fontsize=12)
    ax2.set_ylabel('水平位置 u (像素)', fontsize=12)
    ax2.set_title('水平位置-时间曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: v-t 图 (垂直位置随时间变化)
    ax3 = axes[1, 0]
    ax3.plot(df['t'], df['v'], 'ro-', markersize=5, linewidth=1.5)
    ax3.set_xlabel('时间 t (秒)', fontsize=12)
    ax3.set_ylabel('垂直位置 v (像素)', fontsize=12)
    ax3.set_title('垂直位置-时间曲线', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 速度分析
    ax4 = axes[1, 1]
    dt = np.diff(df['t'].values)
    du = np.diff(df['u'].values)
    dv = np.diff(df['v'].values)
    # 避免除零
    dt[dt == 0] = 1/30
    speed = np.sqrt(du**2 + dv**2) / dt
    t_mid = (df['t'].values[:-1] + df['t'].values[1:]) / 2
    
    ax4.plot(t_mid, speed, 'go-', markersize=5, linewidth=1.5)
    ax4.axhline(np.mean(speed), color='r', linestyle='--', 
                label=f'平均速度: {np.mean(speed):.0f} px/s')
    ax4.set_xlabel('时间 t (秒)', fontsize=12)
    ax4.set_ylabel('速度 (像素/秒)', fontsize=12)
    ax4.set_title('足球速度变化', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'trajectory_analysis.png', dpi=150, bbox_inches='tight')
    print(f'\n轨迹分析图已保存: {vis_dir / "trajectory_analysis.png"}')
    
    # ===== 图2: 2D轨迹大图 =====
    fig2, ax = plt.subplots(figsize=(16, 9))
    scatter = ax.scatter(df['u'], df['v'], c=df['t'], cmap='plasma', s=100, 
                         edgecolors='white', linewidths=1)
    ax.plot(df['u'], df['v'], 'w-', alpha=0.6, linewidth=2)
    
    # 标记起点和终点
    ax.scatter(df['u'].iloc[0], df['v'].iloc[0], c='lime', s=200, marker='o', 
               edgecolors='black', linewidths=2, label='起点', zorder=10)
    ax.scatter(df['u'].iloc[-1], df['v'].iloc[-1], c='red', s=200, marker='s', 
               edgecolors='black', linewidths=2, label='终点', zorder=10)
    
    # 添加帧号标注(每5帧标注一次)
    for i in range(0, len(df), 5):
        ax.annotate(f'{df["frame"].iloc[i]}', 
                    (df['u'].iloc[i], df['v'].iloc[i] - 20), 
                    fontsize=9, ha='center', color='yellow')
    
    ax.set_xlabel('u (像素)', fontsize=14)
    ax.set_ylabel('v (像素)', fontsize=14)
    ax.set_title('足球飞行轨迹 (2D像素坐标)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_facecolor('#2E3440')
    ax.legend(loc='upper left', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('时间 (秒)', fontsize=12)
    
    plt.savefig(vis_dir / 'trajectory_2d.png', dpi=150, bbox_inches='tight', 
                facecolor='#2E3440')
    print(f'2D轨迹图已保存: {vis_dir / "trajectory_2d.png"}')
    
    # ===== 图3: 轨迹叠加在示例帧上 =====
    frames_dir = Path(__file__).parent / '../../../src/img/frames'
    frames_dir = frames_dir.resolve()
    frame_files = sorted(frames_dir.glob('*.jpg'))
    
    if frame_files:
        import cv2
        # 读取中间帧作为背景
        mid_frame = cv2.imread(str(frame_files[len(frame_files)//2]))
        mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
        
        fig3, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(mid_frame)
        
        # 绘制轨迹
        ax.plot(df['u'], df['v'], 'y-', linewidth=2, alpha=0.8)
        scatter = ax.scatter(df['u'], df['v'], c=df['t'], cmap='cool', s=80, 
                            edgecolors='white', linewidths=1)
        
        # 标记起点和终点
        ax.scatter(df['u'].iloc[0], df['v'].iloc[0], c='lime', s=150, marker='o', 
                   edgecolors='black', linewidths=2, label='起点', zorder=10)
        ax.scatter(df['u'].iloc[-1], df['v'].iloc[-1], c='red', s=150, marker='s', 
                   edgecolors='black', linewidths=2, label='终点', zorder=10)
        
        ax.set_title('足球轨迹叠加图', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=12)
        ax.axis('off')
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('时间 (秒)', fontsize=12)
        
        plt.savefig(vis_dir / 'trajectory_overlay.png', dpi=150, bbox_inches='tight')
        print(f'轨迹叠加图已保存: {vis_dir / "trajectory_overlay.png"}')
    
    print('\n===== 轨迹统计 =====')
    print(f'总帧数: {len(df)}')
    print(f'时间跨度: {df["t"].max() - df["t"].min():.3f} 秒')
    print(f'水平位移: {df["u"].iloc[-1] - df["u"].iloc[0]:.0f} px (向右为正)')
    print(f'垂直位移: {df["v"].iloc[-1] - df["v"].iloc[0]:.0f} px (向下为正)')
    print(f'平均速度: {np.mean(speed):.0f} px/s')
    print(f'最大速度: {np.max(speed):.0f} px/s')
    print(f'最小速度: {np.min(speed):.0f} px/s')
    
    plt.show()


if __name__ == '__main__':
    main()
