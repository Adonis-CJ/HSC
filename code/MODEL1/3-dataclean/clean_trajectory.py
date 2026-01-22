#!/usr/bin/env python3
"""
轨迹数据清洗模块
功能：
1. 异常点检测与剔除 (Z-Score / IQR / 速度阈值)
2. 缺失点插值补齐 (线性 / 三次 / 样条)
3. 轨迹平滑滤波 (Savitzky-Golay / 高斯 / Kalman)
"""

import numpy as np
import pandas as pd
import json
import argparse
import time
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, CubicSpline
import yaml


class TrajectoryCleanerError(Exception):
    """轨迹清洗异常"""
    pass


class OutlierDetector:
    """异常点检测器"""
    
    @staticmethod
    def zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Z-Score 异常点检测
        返回布尔数组，True 表示正常点
        """
        if len(data) < 3:
            return np.ones(len(data), dtype=bool)
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.ones(len(data), dtype=bool)
        z_scores = np.abs((data - mean) / std)
        return z_scores < threshold
    
    @staticmethod
    def iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """
        IQR 四分位距异常点检测
        """
        if len(data) < 4:
            return np.ones(len(data), dtype=bool)
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr_val = q3 - q1
        lower = q1 - multiplier * iqr_val
        upper = q3 + multiplier * iqr_val
        return (data >= lower) & (data <= upper)
    
    @staticmethod
    def velocity(u: np.ndarray, v: np.ndarray, t: np.ndarray, 
                 max_velocity: float = 3000.0) -> np.ndarray:
        """
        速度阈值异常点检测
        max_velocity: 最大允许速度 (像素/秒)
        """
        if len(u) < 2:
            return np.ones(len(u), dtype=bool)
        
        valid = np.ones(len(u), dtype=bool)
        
        for i in range(1, len(u)):
            dt = t[i] - t[i-1]
            if dt <= 0:
                continue
            du = u[i] - u[i-1]
            dv = v[i] - v[i-1]
            speed = np.sqrt(du**2 + dv**2) / dt
            if speed > max_velocity:
                valid[i] = False
        
        return valid


class Interpolator:
    """缺失点插值器"""
    
    @staticmethod
    def linear(t: np.ndarray, values: np.ndarray, 
               t_new: np.ndarray) -> np.ndarray:
        """线性插值"""
        f = interp1d(t, values, kind='linear', fill_value='extrapolate')
        return f(t_new)
    
    @staticmethod
    def cubic(t: np.ndarray, values: np.ndarray, 
              t_new: np.ndarray) -> np.ndarray:
        """三次插值"""
        if len(t) < 4:
            return Interpolator.linear(t, values, t_new)
        f = interp1d(t, values, kind='cubic', fill_value='extrapolate')
        return f(t_new)
    
    @staticmethod
    def spline(t: np.ndarray, values: np.ndarray, 
               t_new: np.ndarray) -> np.ndarray:
        """三次样条插值"""
        if len(t) < 3:
            return Interpolator.linear(t, values, t_new)
        cs = CubicSpline(t, values)
        return cs(t_new)


class Smoother:
    """轨迹平滑器"""
    
    @staticmethod
    def savgol(data: np.ndarray, window_size: int = 5, 
               poly_order: int = 2) -> np.ndarray:
        """Savitzky-Golay 滤波"""
        if len(data) < window_size:
            return data.copy()
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        # 确保多项式阶数小于窗口大小
        poly_order = min(poly_order, window_size - 1)
        return savgol_filter(data, window_size, poly_order)
    
    @staticmethod
    def gaussian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """高斯滤波"""
        return gaussian_filter1d(data, sigma)
    
    @staticmethod
    def kalman(data: np.ndarray, process_var: float = 1.0, 
               measure_var: float = 1.0) -> np.ndarray:
        """简单 1D Kalman 平滑"""
        n = len(data)
        result = np.zeros(n)
        
        # 初始化
        x_est = data[0]
        p_est = 1.0
        
        # 前向滤波
        estimates = []
        variances = []
        
        for i in range(n):
            # 预测
            x_pred = x_est
            p_pred = p_est + process_var
            
            # 更新
            k = p_pred / (p_pred + measure_var)
            x_est = x_pred + k * (data[i] - x_pred)
            p_est = (1 - k) * p_pred
            
            estimates.append(x_est)
            variances.append(p_est)
        
        # 后向平滑 (RTS 平滑器)
        result[-1] = estimates[-1]
        for i in range(n - 2, -1, -1):
            c = variances[i] / (variances[i] + process_var)
            result[i] = estimates[i] + c * (result[i + 1] - estimates[i])
        
        return result


class TrajectoryCleaner:
    """轨迹清洗主类"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 默认配置
        self.outlier_method = self.config.get('outlier', {}).get('method', 'zscore')
        self.outlier_threshold = self.config.get('outlier', {}).get('threshold', 3.0)
        
        self.interp_method = self.config.get('interpolation', {}).get('method', 'cubic')
        self.max_gap = self.config.get('interpolation', {}).get('max_gap', 5)
        
        self.smooth_method = self.config.get('smoothing', {}).get('method', 'savgol')
        self.window_size = self.config.get('smoothing', {}).get('window_size', 5)
        self.poly_order = self.config.get('smoothing', {}).get('poly_order', 2)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'valid_detections': 0,
            'outliers_removed': 0,
            'interpolated_frames': 0,
            'smoothing_method': self.smooth_method,
            'processing_time': 0
        }
    
    def detect_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """检测异常点，返回有效点掩码"""
        u = df['u'].values
        v = df['v'].values
        t = df['t'].values
        
        if self.outlier_method == 'zscore':
            # 对 u, v 分别检测
            valid_u = OutlierDetector.zscore(u, self.outlier_threshold)
            valid_v = OutlierDetector.zscore(v, self.outlier_threshold)
            valid = valid_u & valid_v
        elif self.outlier_method == 'iqr':
            valid_u = OutlierDetector.iqr(u, self.outlier_threshold)
            valid_v = OutlierDetector.iqr(v, self.outlier_threshold)
            valid = valid_u & valid_v
        elif self.outlier_method == 'velocity':
            valid = OutlierDetector.velocity(u, v, t, self.outlier_threshold * 1000)
        else:
            valid = np.ones(len(df), dtype=bool)
        
        return valid
    
    def interpolate_missing(self, df: pd.DataFrame, 
                            total_frames: int, fps: float) -> pd.DataFrame:
        """插值补齐缺失帧"""
        # 生成完整帧序列
        all_frames = np.arange(1, total_frames + 1)
        all_t = (all_frames - 1) / fps
        
        existing_frames = df['frame'].values
        existing_t = df['t'].values
        existing_u = df['u'].values
        existing_v = df['v'].values
        
        # 找出缺失帧
        missing_frames = np.setdiff1d(all_frames, existing_frames)
        
        # 检查缺失帧是否在有效范围内（不外推）
        min_frame = existing_frames.min()
        max_frame = existing_frames.max()
        missing_in_range = missing_frames[(missing_frames >= min_frame) & 
                                          (missing_frames <= max_frame)]
        
        if len(missing_in_range) == 0:
            return df
        
        # 插值
        if self.interp_method == 'linear':
            interp_func = Interpolator.linear
        elif self.interp_method == 'cubic':
            interp_func = Interpolator.cubic
        elif self.interp_method == 'spline':
            interp_func = Interpolator.spline
        else:
            interp_func = Interpolator.linear
        
        # 计算插值点
        missing_t = (missing_in_range - 1) / fps
        interp_u = interp_func(existing_t, existing_u, missing_t)
        interp_v = interp_func(existing_t, existing_v, missing_t)
        
        # 创建插值数据
        interp_data = pd.DataFrame({
            'frame': missing_in_range,
            't': missing_t,
            'u': interp_u,
            'v': interp_v,
            'score': 0.0,
            'is_interpolated': 1
        })
        
        # 标记原始数据
        df = df.copy()
        df['is_interpolated'] = 0
        
        # 合并并排序
        result = pd.concat([df, interp_data], ignore_index=True)
        result = result.sort_values('frame').reset_index(drop=True)
        
        self.stats['interpolated_frames'] = len(missing_in_range)
        
        return result
    
    def smooth_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """平滑轨迹"""
        df = df.copy()
        u = df['u'].values.astype(float)
        v = df['v'].values.astype(float)
        
        if self.smooth_method == 'savgol':
            u_smooth = Smoother.savgol(u, self.window_size, self.poly_order)
            v_smooth = Smoother.savgol(v, self.window_size, self.poly_order)
        elif self.smooth_method == 'gaussian':
            sigma = self.window_size / 4.0
            u_smooth = Smoother.gaussian(u, sigma)
            v_smooth = Smoother.gaussian(v, sigma)
        elif self.smooth_method == 'kalman':
            u_smooth = Smoother.kalman(u)
            v_smooth = Smoother.kalman(v)
        else:
            u_smooth = u
            v_smooth = v
        
        df['u'] = u_smooth
        df['v'] = v_smooth
        df['is_smoothed'] = 1
        
        return df
    
    def clean(self, df: pd.DataFrame, total_frames: int = None, 
              fps: float = 30.0) -> pd.DataFrame:
        """
        执行完整清洗流程
        """
        start_time = time.time()
        
        self.stats['total_frames'] = total_frames or len(df)
        self.stats['valid_detections'] = len(df)
        
        # 1. 异常点检测与剔除
        valid_mask = self.detect_outliers(df)
        outlier_count = (~valid_mask).sum()
        self.stats['outliers_removed'] = int(outlier_count)
        
        if outlier_count > 0:
            print(f"检测到 {outlier_count} 个异常点，已剔除")
            df = df[valid_mask].reset_index(drop=True)
        
        # 2. 缺失点插值
        if total_frames is not None:
            df = self.interpolate_missing(df, total_frames, fps)
            if self.stats['interpolated_frames'] > 0:
                print(f"插值补齐 {self.stats['interpolated_frames']} 个缺失帧")
        
        # 3. 轨迹平滑
        df = self.smooth_trajectory(df)
        print(f"轨迹平滑完成 (方法: {self.smooth_method})")
        
        self.stats['processing_time'] = round(time.time() - start_time, 3)
        
        return df
    
    def get_report(self) -> dict:
        """获取清洗报告"""
        return self.stats.copy()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('dataclean', {})


def main():
    parser = argparse.ArgumentParser(description='轨迹数据清洗')
    parser.add_argument('--input', type=str, 
                        default='../../../output/trajectory/2d/trajectory_2d_manual.csv',
                        help='输入轨迹文件')
    parser.add_argument('--output', type=str,
                        default='../../../output/trajectory/2d/trajectory_2d_clean.csv',
                        help='输出轨迹文件')
    parser.add_argument('--report', type=str,
                        default='../../../output/trajectory/2d/clean_report.json',
                        help='清洗报告文件')
    parser.add_argument('--config', type=str,
                        default='../../../config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--total-frames', type=int, default=36,
                        help='总帧数（用于插值）')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='视频帧率')
    
    args = parser.parse_args()
    
    # 转换路径
    script_dir = Path(__file__).parent
    input_path = (script_dir / args.input).resolve()
    output_path = (script_dir / args.output).resolve()
    report_path = (script_dir / args.report).resolve()
    config_path = (script_dir / args.config).resolve()
    
    print("=" * 50)
    print("轨迹数据清洗模块")
    print("=" * 50)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    
    # 加载配置
    config = {}
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"已加载配置: {config_path}")
    
    # 读取输入数据
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条记录")
    
    # 创建清洗器并执行
    cleaner = TrajectoryCleaner(config)
    df_clean = cleaner.clean(df, args.total_frames, args.fps)
    
    # 整理输出列
    output_cols = ['frame', 't', 'u', 'v', 'score', 'is_interpolated', 'is_smoothed']
    for col in output_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0
    
    df_clean = df_clean[output_cols]
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n清洗后数据: {len(df_clean)} 条记录")
    print(f"已保存: {output_path}")
    
    # 保存报告
    report = cleaner.get_report()
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_path}")
    
    # 打印统计
    print("\n===== 清洗统计 =====")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 可视化对比
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        df_orig = pd.read_csv(input_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 原始轨迹
        ax1 = axes[0]
        ax1.plot(df_orig['u'], df_orig['v'], 'b.-', markersize=8, alpha=0.7)
        ax1.scatter(df_orig['u'], df_orig['v'], c=df_orig['t'], cmap='viridis', s=50)
        ax1.set_xlabel('u (像素)')
        ax1.set_ylabel('v (像素)')
        ax1.set_title('原始轨迹', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 清洗后轨迹
        ax2 = axes[1]
        # 区分插值点和原始点
        orig_mask = df_clean['is_interpolated'] == 0
        interp_mask = df_clean['is_interpolated'] == 1
        
        ax2.plot(df_clean['u'], df_clean['v'], 'g-', linewidth=2, alpha=0.7)
        ax2.scatter(df_clean.loc[orig_mask, 'u'], df_clean.loc[orig_mask, 'v'], 
                   c='blue', s=50, label='原始点 (平滑后)', zorder=5)
        if interp_mask.any():
            ax2.scatter(df_clean.loc[interp_mask, 'u'], df_clean.loc[interp_mask, 'v'], 
                       c='red', s=50, marker='x', label='插值点', zorder=5)
        ax2.set_xlabel('u (像素)')
        ax2.set_ylabel('v (像素)')
        ax2.set_title('清洗后轨迹', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        vis_path = output_path.parent / 'clean_comparison.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存: {vis_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == '__main__':
    main()
