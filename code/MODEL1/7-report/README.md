# 7-report: 可视化与报告导出模块

## 功能说明
该模块负责生成论文所需的可视化图表和结果汇总报告。

## 输入
- 各阶段输出数据（轨迹、拟合参数、速度估计等）
- 帧图像

## 输出
- 可视化图片：`src/img/vis/`
- 汇总报告：`output/report/`
- LaTeX 插图素材

## 可视化内容

### 1. 轨迹可视化
- 2D 像素轨迹叠加在视频帧上
- 3D 轨迹空间图（含场地平面）
- 地面投影轨迹（俯视图）

### 2. 拟合可视化
- 原始点 + 拟合曲线
- 残差分布图
- 置信区间

### 3. 速度可视化
- 速度随时间变化曲线
- 速度矢量图
- 初速度分解示意图

### 4. 场地标定可视化
- 标定点与重投影
- 场地线检测结果
- 坐标系示意图

## 输出文件结构

```
src/img/vis/
├── detection/           # 检测结果可视化
│   └── frame_xxx_det.png
├── calibration/         # 标定可视化
│   └── field_lines.png
├── reconstruction/      # 重建可视化
│   ├── trajectory_2d.png
│   ├── trajectory_3d.png
│   └── ground_projection.png
├── fitting/             # 拟合可视化
│   ├── curve_fit.png
│   └── residuals.png
└── velocity/            # 速度可视化
    ├── velocity_profile.png
    └── v0_decomposition.png

output/report/
├── summary.json         # 汇总数据
├── summary.md           # 可读报告
└── latex_figures/       # LaTeX 用图片副本
```

## 使用方法

### 生成所有可视化
```python
python generate_visualizations.py --config config/config.yaml
```

### 生成汇总报告
```python
python generate_report.py --output output/report/
```

## 图片规格建议

| 用途 | 分辨率 | 格式 | DPI |
|------|--------|------|-----|
| 论文插图 | 宽度 ≥ 1200px | PNG/PDF | 300 |
| 演示 | 1920×1080 | PNG | 150 |
| 缩略图 | 400×300 | JPG | 72 |

## LaTeX 插入示例

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{img/vis/fitting/curve_fit.png}
    \caption{轨迹拟合结果}
    \label{fig:curve_fit}
\end{figure}
```

## 配置参数
参见 `config/config.yaml` 中的 `visualization` 部分。
