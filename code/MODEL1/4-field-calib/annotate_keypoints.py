#!/usr/bin/env python3
"""
场地标定模块 - 手动标注关键点
用于建立像素坐标到场地物理坐标的映射
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime


class KeypointAnnotator:
    """场地关键点标注工具"""
    
    # 预定义的场地关键点及其物理坐标 (x, y, z) 单位：米
    # 坐标系：原点在场地中心，X轴平行于球门线向右为正，Y轴指向球门为正，Z轴竖直向上
    FIELD_POINTS = {
        'goal_left_bottom': (-3.66, 52.5, 0),      # 球门左底部点（X负方向）
        'goal_right_bottom': (3.66, 52.5, 0),      # 球门右底部点（X正方向）
        'goal_left_top': (-3.66, 52.5, 2.44),      # 球门左顶部点
        'goal_right_top': (3.66, 52.5, 2.44),      # 球门右顶部点
        'field_corner_left': (-34, 52.5, 0),       # 场地左顶点（X负方向）
        'penalty_spot': (0, 41.5, 0),              # 罚球点（X=0）
        'penalty_area_left': (-20.16, 36, 0),      # 大禁区左下点（X负方向）
        'penalty_area_right': (20.16, 36, 0),      # 大禁区右下点（X正方向）
    }
    
    def __init__(self, image_path, output_path):
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        
        # 读取图像
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
        
        self.image_size = (self.image.shape[1], self.image.shape[0])
        
        # 标注数据
        self.annotations = []  # [(name, pixel_uv, field_xyz), ...]
        self.current_point_idx = 0
        self.point_names = list(self.FIELD_POINTS.keys())
        
        # 显示状态
        self.scale = 1.0
        self.window_name = "Field Calibration - Click on keypoints"
        
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point_idx >= len(self.point_names):
                print("所有点已标注完成！按 S 保存")
                return
            
            # 转换回原始坐标
            real_x = int(x / self.scale)
            real_y = int(y / self.scale)
            
            name = self.point_names[self.current_point_idx]
            field_coord = self.FIELD_POINTS[name]
            
            self.annotations.append({
                'name': name,
                'pixel': (real_x, real_y),
                'field': field_coord
            })
            
            print(f"标注 {name}: 像素({real_x}, {real_y}) -> 场地{field_coord}")
            self.current_point_idx += 1
    
    def _draw_frame(self):
        """绘制当前标注状态"""
        img = self.image.copy()
        h, w = img.shape[:2]
        
        # 缩放显示
        max_display = 1400
        if w > max_display:
            self.scale = max_display / w
            img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            self.scale = 1.0
        
        # 绘制已标注的点
        for ann in self.annotations:
            px, py = ann['pixel']
            px_scaled = int(px * self.scale)
            py_scaled = int(py * self.scale)
            
            # 绘制点
            cv2.circle(img, (px_scaled, py_scaled), 8, (0, 255, 0), -1)
            cv2.circle(img, (px_scaled, py_scaled), 10, (255, 255, 255), 2)
            
            # 绘制标签
            label = f"{ann['name']}"
            cv2.putText(img, label, (px_scaled + 15, py_scaled - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示当前需要标注的点
        if self.current_point_idx < len(self.point_names):
            current_name = self.point_names[self.current_point_idx]
            current_coord = self.FIELD_POINTS[current_name]
            info = f"Please click: {current_name} -> {current_coord}"
            cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 255), 2)
        else:
            cv2.putText(img, "All points annotated! Press S to save", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示进度
        progress = f"Progress: {len(self.annotations)}/{len(self.point_names)}"
        cv2.putText(img, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        # 操作提示
        help_text = "[Click]=Mark | [Z]=Undo | [S]=Save | [Q]=Quit"
        cv2.putText(img, help_text, (10, img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img
    
    def save(self):
        """保存标注结果"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'image_path': str(self.image_path),
            'image_size': list(self.image_size),
            'coordinate_system': {
                'origin': 'field_center',
                'x_axis': 'parallel to goal line, right positive',
                'y_axis': 'perpendicular to goal line, toward goal positive',
                'z_axis': 'vertical up',
                'unit': 'meters'
            },
            'keypoints': [
                {
                    'name': ann['name'],
                    'pixel': list(ann['pixel']),
                    'field': list(ann['field'])
                }
                for ann in self.annotations
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n标注已保存: {self.output_path}")
        print(f"共标注 {len(self.annotations)} 个点")
    
    def run(self):
        """运行标注工具"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("场地标定 - 关键点标注工具")
        print("=" * 60)
        print("请按顺序点击以下场地特征点：")
        for i, (name, coord) in enumerate(self.FIELD_POINTS.items(), 1):
            print(f"  {i}. {name}: {coord}")
        print("\n操作说明:")
        print("  鼠标左键 - 标注当前点")
        print("  Z        - 撤销上一个点")
        print("  S        - 保存并退出")
        print("  Q / ESC  - 不保存退出")
        print("=" * 60 + "\n")
        
        while True:
            img = self._draw_frame()
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q') or key == 27:
                print("退出，不保存")
                break
            
            elif key == ord('s'):
                if len(self.annotations) >= 4:
                    self.save()
                else:
                    print("至少需要标注 4 个点才能保存！")
                    continue
                break
            
            elif key == ord('z'):
                if self.annotations:
                    removed = self.annotations.pop()
                    self.current_point_idx -= 1
                    print(f"撤销: {removed['name']}")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='场地关键点标注')
    parser.add_argument('--image', type=str, 
                        default='../../../src/img/frames/000001.jpg',
                        help='参考帧图像路径')
    parser.add_argument('--output', type=str,
                        default='../../../output/calibration/keypoints.json',
                        help='输出标注文件')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    image_path = (script_dir / args.image).resolve()
    output_path = (script_dir / args.output).resolve()
    
    print(f"图像: {image_path}")
    print(f"输出: {output_path}")
    
    annotator = KeypointAnnotator(image_path, output_path)
    annotator.run()


if __name__ == '__main__':
    main()
