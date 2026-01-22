#!/usr/bin/env python3
"""快速测试X坐标映射"""

import json
import numpy as np
import pandas as pd

# 加载标定点
with open('output/calibration/keypoints.json') as f:
    kp_data = json.load(f)

# 提取球门标定点
goal_left = None
goal_right = None
for kp in kp_data['keypoints']:
    if kp['name'] == 'goal_left_bottom':
        goal_left = {'u': kp['pixel'][0], 'X': kp['field'][0]}
    elif kp['name'] == 'goal_right_bottom':
        goal_right = {'u': kp['pixel'][0], 'X': kp['field'][0]}

print("=" * 50)
print("球门标定点:")
print(f"  左柱: u={goal_left['u']}, X={goal_left['X']}m")
print(f"  右柱: u={goal_right['u']}, X={goal_right['X']}m")

# 计算映射
dX_du = (goal_right['X'] - goal_left['X']) / (goal_right['u'] - goal_left['u'])
u_center = (goal_left['u'] + goal_right['u']) / 2
X_center = (goal_left['X'] + goal_right['X']) / 2

print(f"\n线性映射: X = {X_center:.2f} + {dX_du:.5f} * (u - {u_center:.0f})")

# 加载轨迹
df = pd.read_csv('output/trajectory/2d/trajectory_2d_clean.csv')
u_start = df['u'].iloc[0]
u_end = df['u'].iloc[-1]

X_start = X_center + dX_du * (u_start - u_center)
X_end = X_center + dX_du * (u_end - u_center)

print("\n" + "=" * 50)
print("球轨迹映射:")
print(f"  起点: u={u_start:.0f} -> X={X_start:.2f}m")
print(f"  终点: u={u_end:.0f} -> X={X_end:.2f}m")
print(f"\n方向: 从 X={X_start:.1f}m {'(左侧)' if X_start < 0 else '(右侧)'}")
print(f"     飞向 X={X_end:.1f}m {'(左侧)' if X_end < 0 else '(右侧)'}")

# 相对于球门
print(f"\n相对于球门(宽度±3.66m):")
if abs(X_end) <= 3.66:
    print(f"  ✓ 终点在球门内! X={X_end:.2f}m")
elif X_end > 3.66:
    print(f"  终点在球门右侧外 {X_end - 3.66:.2f}m")
else:
    print(f"  终点在球门左侧外 {abs(X_end) - 3.66:.2f}m")
