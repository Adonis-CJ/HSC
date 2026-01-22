"""
视频抽帧脚本
从视频中按指定帧率抽取帧图像，并生成帧元数据文件
"""

import cv2
import os
import csv
import argparse
from pathlib import Path
import yaml


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 默认配置路径
        script_dir = Path(__file__).parent
        config_path = script_dir.parent.parent.parent / "config" / "config.yaml"
    
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 30,
    frame_format: str = "%06d.jpg",
    metadata_path: str = None
) -> dict:
    """
    从视频中抽取帧
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出帧图像目录
        target_fps: 目标帧率（如果为0则使用原始帧率）
        frame_format: 帧文件命名格式
        metadata_path: 帧元数据CSV文件路径
    
    Returns:
        包含抽帧统计信息的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"视频信息:")
    print(f"  - 原始帧率: {original_fps:.2f} fps")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 分辨率: {width} x {height}")
    print(f"  - 时长: {duration:.2f} 秒")
    
    # 计算抽帧间隔
    if target_fps <= 0 or target_fps >= original_fps:
        # 使用原始帧率，抽取所有帧
        frame_interval = 1
        actual_fps = original_fps
    else:
        # 按目标帧率抽帧
        frame_interval = original_fps / target_fps
        actual_fps = target_fps
    
    print(f"\n目标帧率: {actual_fps:.2f} fps")
    print(f"抽帧间隔: 每 {frame_interval:.2f} 帧取1帧")
    
    # 准备元数据记录
    metadata = []
    
    # 抽帧
    frame_idx = 0
    saved_count = 0
    next_frame_to_save = 0
    
    print(f"\n开始抽帧...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 判断是否保存当前帧
        if frame_idx >= next_frame_to_save:
            saved_count += 1
            # 计算时间戳
            timestamp = frame_idx / original_fps
            
            # 生成文件名
            filename = frame_format % saved_count
            filepath = os.path.join(output_dir, filename)
            
            # 保存帧
            cv2.imwrite(filepath, frame)
            
            # 记录元数据
            metadata.append({
                'frame': saved_count,
                'original_frame': frame_idx,
                'timestamp': timestamp,
                'filepath': filepath
            })
            
            # 计算下一帧位置
            next_frame_to_save += frame_interval
            
            # 进度显示
            if saved_count % 50 == 0:
                progress = frame_idx / total_frames * 100
                print(f"  已处理: {saved_count} 帧 ({progress:.1f}%)")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n抽帧完成!")
    print(f"  - 总共保存: {saved_count} 帧")
    print(f"  - 输出目录: {output_dir}")
    
    # 保存元数据
    if metadata_path is None:
        metadata_path = os.path.join(
            os.path.dirname(output_dir), 
            "..", "..", "output", "trajectory", "2d", "frame_metadata.csv"
        )
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'original_frame', 'timestamp', 'filepath'])
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"  - 元数据文件: {metadata_path}")
    
    # 返回统计信息
    return {
        'video_path': video_path,
        'original_fps': original_fps,
        'target_fps': actual_fps,
        'total_original_frames': total_frames,
        'saved_frames': saved_count,
        'width': width,
        'height': height,
        'duration': duration,
        'output_dir': output_dir,
        'metadata_path': metadata_path
    }


def main():
    parser = argparse.ArgumentParser(description='视频抽帧工具')
    parser.add_argument('--input', '-i', type=str, 
                        default='src/video/Ball_video.mp4',
                        help='输入视频路径')
    parser.add_argument('--output', '-o', type=str,
                        default='src/img/frames',
                        help='输出帧图像目录')
    parser.add_argument('--fps', type=int, default=30,
                        help='目标帧率 (0表示使用原始帧率)')
    parser.add_argument('--format', type=str, default='%06d.jpg',
                        help='帧文件命名格式')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    video_config = config.get('video', {})
    
    # 使用命令行参数或配置文件中的值
    video_path = args.input or video_config.get('input_path', 'src/video/Ball_video.mp4')
    output_dir = args.output or video_config.get('output_frames_dir', 'src/img/frames')
    target_fps = args.fps if args.fps != 30 else video_config.get('fps', 30)
    frame_format = args.format or video_config.get('frame_format', '%06d.jpg')
    
    # 处理相对路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    if not os.path.isabs(video_path):
        video_path = project_root / video_path
    if not os.path.isabs(output_dir):
        output_dir = project_root / output_dir
    
    # 执行抽帧
    stats = extract_frames(
        video_path=str(video_path),
        output_dir=str(output_dir),
        target_fps=target_fps,
        frame_format=frame_format
    )
    
    print(f"\n统计信息: {stats}")


if __name__ == '__main__':
    main()
