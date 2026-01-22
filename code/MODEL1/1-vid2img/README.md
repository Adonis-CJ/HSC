# 1-vid2img: 视频抽帧模块

## 功能说明
该模块负责从原始视频中抽取帧序列，并建立帧序号与时间戳的对应关系。

## 输入
- 原始视频文件：`src/video/Ball_video.mp4`

## 输出
- 帧图像序列：`src/img/frames/000001.jpg`, `000002.jpg`, ...
- 帧元数据文件：`output/trajectory/2d/frame_metadata.csv`

## 输出格式

### frame_metadata.csv
```csv
frame,timestamp,filepath
1,0.0000,src/img/frames/000001.jpg
2,0.0333,src/img/frames/000002.jpg
...
```

## 使用方法

### 快速抽帧（使用 ffmpeg）
```bash
ffmpeg -i src/video/Ball_video.mp4 -vf fps=30 src/img/frames/%06d.jpg
```

### Python 脚本
```python
python extract_frames.py --input src/video/Ball_video.mp4 --output src/img/frames/ --fps 30
```

## 配置参数
参见 `config/config.yaml` 中的 `video` 部分。

## 注意事项
1. 保持帧率一致性，建议统一使用 30fps
2. 帧序号从 1 开始，使用 6 位数字命名
3. 时间戳计算公式：`t = (frame - 1) / fps`
