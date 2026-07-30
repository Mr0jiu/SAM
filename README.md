# SAM 自动标注

基于 [Segment Anything](https://github.com/facebookresearch/segment-anything) 的半自动标注工具。读取已有的 COCO bbox 标注，用 SAM 生成像素级分割掩膜，并导出为标准 COCO 格式。

## 工作流程

```
COCO bbox 标注  →  SAM (bbox prompt)  →  掩膜后处理  →  COCO segmentation 导出
```

## 目录结构

```
SAM/
├── pipeline.py                 # 主流程：bbox → SAM → COCO segmentation
├── visualize_coco_masks.py     # 可视化工具：渲染 COCO 掩膜到图片上
├── segment-anything-main/      # SAM 官方库（本地安装）
├── yolo to SAM111.coco/        # 输入数据（图片 + COCO bbox 标注）
│   └── train/
├── reslut.coco/                # 输出结果（COCO segmentation）
│   └── result.json
└── sam_vit_b_01ec64.pth        # SAM 权重文件（需自行下载）
```

## 环境准备

```bash
# 进入 SAM 官方库目录安装
cd segment-anything-main
pip install -e .

# 安装其他依赖
pip install numpy pillow opencv-python
```

SAM 权重下载地址：https://github.com/facebookresearch/segment-anything#model-checkpoints

## 使用方法

### 1. 运行标注流程

```bash
python pipeline.py \
  --image-dir "yolo to SAM111.coco/train" \
  --coco-json "yolo to SAM111.coco/train/_annotations.coco.json" \
  --output-json "reslut.coco/result.json" \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --device cuda
```

**常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image-dir` | 图像目录 | `yolo to SAM111.coco\test` |
| `--coco-json` | 输入 COCO 标注 | `...\_annotations.coco.json` |
| `--output-json` | 输出路径 | `reslut.coco\result.json` |
| `--sam-checkpoint` | SAM 权重 | `sam_vit_b_01ec64.pth` |
| `--device` | `cuda` 或 `cpu` | `cuda` |
| `--min-mask-area` | 最小掩膜面积 | `64` |
| `--min-score` | 最小 IoU 分数 | `0.5` |
| `--polygon` | 导出多边形（默认 RLE） | 关闭 |

### 2. 可视化检查结果

```bash
python visualize_coco_masks.py \
  --image-dir "yolo to SAM111.coco/train" \
  --coco-json "reslut.coco/result.json" \
  --out-dir test \
  --alpha 0.45 \
  --max-images 50
```

渲染后的图片会输出到 `--out-dir`，文件名带 `_vis` 后缀。

## 输出格式

输出文件为标准 COCO 格式，`annotations` 中的 `segmentation` 字段支持：

- **RLE**（默认）：`{"size": [h, w], "counts": [...]}`
- **Polygon**（加 `--polygon`）：`[[x1, y1, x2, y2, ...], ...]`

每条标注额外包含 SAM 的预测分数 `score` 字段。
