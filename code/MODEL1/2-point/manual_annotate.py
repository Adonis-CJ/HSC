#!/usr/bin/env python3
"""
手动标注足球位置工具
使用方法：
1. 运行脚本后显示图像
2. 鼠标左键点击足球中心位置
3. 按 'n' 或 空格 进入下一帧
4. 按 'p' 返回上一帧
5. 按 'd' 删除当前帧标注（标记为无球）
6. 按 'z' 撤销上一次操作
7. 按 's' 保存并退出
8. 按 'q' 或 ESC 不保存退出
"""

import cv2
import os
import csv
import argparse
from pathlib import Path


class ManualAnnotator:
    def __init__(self, frames_dir, metadata_path, output_path):
        self.frames_dir = Path(frames_dir)
        self.metadata_path = Path(metadata_path)
        self.output_path = Path(output_path)
        
        # 加载帧列表
        self.frame_files = sorted([f for f in self.frames_dir.iterdir() 
                                   if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        self.num_frames = len(self.frame_files)
        
        if self.num_frames == 0:
            raise ValueError(f"在 {frames_dir} 中未找到图像文件")
        
        # 加载元数据（帧号到时间戳映射）
        self.metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.metadata[int(row['frame'])] = float(row['timestamp'])
        
        # 标注数据: {frame_idx: (u, v) or None}
        self.annotations = {}
        
        # 加载已有标注
        self._load_existing_annotations()
        
        # 当前帧索引
        self.current_idx = 0
        
        # 当前点击位置（临时）
        self.current_click = None
        
        # 操作历史（用于撤销）
        self.history = []
        
        # 窗口名称
        self.window_name = "Manual Ball Annotation - Click on ball center"
        
        # 缩放因子（如果图像太大）
        self.scale = 1.0
        
    def _load_existing_annotations(self):
        """加载已有的标注文件"""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame = int(row['frame'])
                    u = float(row['u'])
                    v = float(row['v'])
                    if u >= 0 and v >= 0:
                        self.annotations[frame] = (u, v)
                    else:
                        self.annotations[frame] = None
            print(f"已加载 {len(self.annotations)} 条已有标注")
    
    def _get_frame_number(self, idx):
        """从文件名提取帧号"""
        filename = self.frame_files[idx].stem
        try:
            return int(filename)
        except ValueError:
            return idx + 1
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 转换回原始坐标
            real_x = int(x / self.scale)
            real_y = int(y / self.scale)
            self.current_click = (real_x, real_y)
            
            # 保存历史
            frame_num = self._get_frame_number(self.current_idx)
            old_val = self.annotations.get(frame_num)
            self.history.append(('annotate', frame_num, old_val))
            
            # 更新标注
            self.annotations[frame_num] = (real_x, real_y)
            print(f"帧 {frame_num}: 标注位置 ({real_x}, {real_y})")
    
    def _draw_frame(self):
        """绘制当前帧及标注"""
        # 读取图像
        img = cv2.imread(str(self.frame_files[self.current_idx]))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        # 如果图像太大，缩放显示
        max_display = 1200
        if w > max_display:
            self.scale = max_display / w
            img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            self.scale = 1.0
        
        frame_num = self._get_frame_number(self.current_idx)
        
        # 绘制已有标注
        if frame_num in self.annotations:
            ann = self.annotations[frame_num]
            if ann is not None:
                u, v = ann
                cx, cy = int(u * self.scale), int(v * self.scale)
                # 绘制十字标记
                cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                # 绘制圆圈
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2)
                status = f"Annotated: ({int(u)}, {int(v)})"
            else:
                status = "Marked as NO BALL"
                cv2.putText(img, "NO BALL", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (0, 0, 255), 3)
        else:
            status = "Not annotated"
        
        # 绘制信息文字
        info_text = f"Frame {frame_num}/{self.num_frames} | {status}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 0), 2)
        
        # 绘制操作提示
        help_text = "[Click]=Mark | [N/Space]=Next | [P]=Prev | [D]=No ball | [Z]=Undo | [S]=Save | [Q]=Quit"
        cv2.putText(img, help_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
        
        # 显示进度条
        progress = int((self.current_idx + 1) / self.num_frames * (img.shape[1] - 20))
        cv2.rectangle(img, (10, img.shape[0] - 40), (10 + progress, img.shape[0] - 35), 
                     (0, 255, 0), -1)
        cv2.rectangle(img, (10, img.shape[0] - 40), (img.shape[1] - 10, img.shape[0] - 35), 
                     (100, 100, 100), 1)
        
        # 统计已标注数量
        annotated = sum(1 for v in self.annotations.values() if v is not None)
        no_ball = sum(1 for v in self.annotations.values() if v is None)
        stats = f"Annotated: {annotated} | No ball: {no_ball} | Remaining: {self.num_frames - len(self.annotations)}"
        cv2.putText(img, stats, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 200, 0), 1)
        
        return img
    
    def _save_annotations(self):
        """保存标注结果"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 't', 'u', 'v', 'score', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h', 'track_id', 'predicted'])
            
            for idx in range(self.num_frames):
                frame_num = self._get_frame_number(idx)
                t = self.metadata.get(frame_num, frame_num / 30.0)  # 默认30fps
                
                if frame_num in self.annotations:
                    ann = self.annotations[frame_num]
                    if ann is not None:
                        u, v = ann
                        # 手动标注置信度为1.0
                        writer.writerow([frame_num, f"{t:.3f}", f"{u:.1f}", f"{v:.1f}", 
                                       1.0, int(u)-5, int(v)-5, 10, 10, 0, 0])
                    # 标记为无球的帧不写入
        
        print(f"\n标注已保存到: {self.output_path}")
        print(f"共标注 {sum(1 for v in self.annotations.values() if v is not None)} 个有效点")
    
    def run(self):
        """运行标注工具"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "="*60)
        print("手动标注工具启动")
        print("="*60)
        print("操作说明:")
        print("  鼠标左键 - 点击足球中心位置")
        print("  N / 空格 - 下一帧")
        print("  P        - 上一帧")
        print("  D        - 标记当前帧无球")
        print("  Z        - 撤销上一次操作")
        print("  S        - 保存并退出")
        print("  Q / ESC  - 不保存退出")
        print("="*60 + "\n")
        
        while True:
            img = self._draw_frame()
            if img is None:
                print(f"无法读取帧 {self.current_idx}")
                self.current_idx = min(self.current_idx + 1, self.num_frames - 1)
                continue
            
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("退出，不保存")
                break
            
            elif key == ord('s'):  # Save
                self._save_annotations()
                break
            
            elif key == ord('n') or key == ord(' '):  # Next
                self.current_idx = min(self.current_idx + 1, self.num_frames - 1)
            
            elif key == ord('p'):  # Previous
                self.current_idx = max(self.current_idx - 1, 0)
            
            elif key == ord('d'):  # Delete / No ball
                frame_num = self._get_frame_number(self.current_idx)
                old_val = self.annotations.get(frame_num)
                self.history.append(('delete', frame_num, old_val))
                self.annotations[frame_num] = None
                print(f"帧 {frame_num}: 标记为无球")
            
            elif key == ord('z'):  # Undo
                if self.history:
                    action, frame_num, old_val = self.history.pop()
                    if old_val is None:
                        self.annotations.pop(frame_num, None)
                    else:
                        self.annotations[frame_num] = old_val
                    print(f"撤销: 帧 {frame_num} 恢复为 {old_val}")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='手动标注足球位置')
    parser.add_argument('--frames', type=str, 
                       default='../../../src/img/frames',
                       help='帧图像目录')
    parser.add_argument('--metadata', type=str,
                       default='../../../output/trajectory/2d/frame_metadata.csv',
                       help='帧元数据文件')
    parser.add_argument('--output', type=str,
                       default='../../../output/trajectory/2d/trajectory_2d_manual.csv',
                       help='输出标注文件')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).parent
    frames_dir = (script_dir / args.frames).resolve()
    metadata_path = (script_dir / args.metadata).resolve()
    output_path = (script_dir / args.output).resolve()
    
    print(f"帧目录: {frames_dir}")
    print(f"输出文件: {output_path}")
    
    annotator = ManualAnnotator(frames_dir, metadata_path, output_path)
    annotator.run()


if __name__ == '__main__':
    main()
