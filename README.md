# SAM 自动标注

基于 [Segment Anything](https://github.com/facebookresearch/segment-anything) 的半自动标注工具。读取已有的 COCO bbox 标注，用 SAM 生成像素级分割掩膜，并导出为标准 COCO 格式。

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

## 工作流程

```
COCO bbox 标注  →  SAM (bbox prompt)  →  掩膜后处理  →  COCO segmentation 导出
```

`pipeline.py` 由 4 个核心模块组成，按以下顺序依次执行：

| 阶段 | 模块 | 具体工作 |
|------|------|----------|
| **1. 数据加载** | `DataManager` | 读取输入的 COCO JSON，解析 `images`（文件名、宽高、路径）、`categories`（类别表）、`annotations`（仅提取 `bbox` 字段，忽略原始 `segmentation`）。将每个 bbox 从 `xywh` 转换为 `xyxy` 格式，按 `image_id` 分组为 SAM 的提示输入。 |
| **2. SAM 分割** | `SamSegmentor` | 逐张图片调用 `SamPredictor`：先 `set_image` 编码图像特征，再以每个 bbox 作为 box prompt 调用 `predict(box=..., multimask_output=False)`，得到二值掩膜和 IoU 分数。掩膜编码支持 RLE（Fortran 序遍历）和 Polygon（`cv2.findContours` 提取轮廓）两种模式。 |
| **3. 后处理过滤** | `MaskPostProcessor` | 对 SAM 输出的掩膜按面积阈值（`--min-mask-area`，默认 64 像素）和置信度阈值（`--min-score`，默认 0.5）过滤，丢弃过小的碎片和低质量预测。 |
| **4. COCO 导出** | `CocoExporter` | 将过滤后的掩膜组装为标准 COCO 格式 JSON：`images` 保留原始元数据，`annotations` 写入 `segmentation`/`area`/`bbox`/`score` 字段，`categories` 沿用输入的类别表（`category_id` 不变）。输出到 `--output-json` 指定路径。 |

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
