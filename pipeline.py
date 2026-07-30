from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class CategoryRecord:
    # 保留原 COCO 类别表，导出时继续使用原始 category_id。
    id: int
    name: str
    supercategory: str = "default"


@dataclass
class PipelineConfig:
    # 当前脚本固定走 bbox + SAM 流程，输入来自 COCO 检测标注。
    image_dir: Path
    coco_json: Path
    output_json: Path
    sam_checkpoint: Path
    sam_model_type: str = "vit_b"
    device: str = "cuda"
    min_mask_area: int = 64
    min_score: float = 0.5
    use_rle: bool = True


@dataclass
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int
    path: Path
    license_id: Optional[int] = None
    date_captured: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class PromptRecord:
    # 每条 prompt 只保留 bbox 和类别，不再读取输入 annotation 的 segmentation。
    image_id: int
    category_id: int
    category_name: str
    box_xyxy: List[float]
    source_annotation_id: Optional[int] = None
    iscrowd: int = 0


@dataclass
class MaskRecord:
    # 这是 SAM 推理后的中间结果，包含导出所需的掩膜、框和类别信息。
    image_id: int
    category_id: int
    category_name: str
    segmentation: Dict[str, Any] | List[List[float]]
    area: float
    bbox_xywh: List[float]
    score: float
    iscrowd: int = 0
    prompt: Optional[PromptRecord] = None


class ProgressBar:
    # 用标准输出画一个轻量进度条，避免额外依赖 tqdm。
    def __init__(self, total: int, width: int = 30) -> None:
        self.total = max(total, 1)
        self.width = width
        self._last_length = 0

    def update(self, current: int, extra_text: str = "") -> None:
        ratio = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        remaining = max(self.total - current, 0)
        message = (
            f"\r标注进度 [{bar}] {current}/{self.total} "
            f"({ratio * 100:6.2f}%) 剩余 {remaining}"
        )
        if extra_text:
            message += f" | {extra_text}"

        terminal_width = shutil.get_terminal_size((120, 20)).columns
        if len(message) > terminal_width - 1:
            message = message[: max(terminal_width - 4, 1)] + "..."

        padding = " " * max(self._last_length - len(message), 0)
        sys.stdout.write("\r" + message + padding)
        sys.stdout.flush()
        self._last_length = len(message)

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class DataManager:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._coco_cache: Optional[Dict[str, Any]] = None

    def load_coco(self) -> Dict[str, Any]:
        if self._coco_cache is None:
            with self.config.coco_json.open("r", encoding="utf-8") as fp:
                self._coco_cache = json.load(fp)
        return self._coco_cache

    def load_categories(self) -> List[CategoryRecord]:
        coco = self.load_coco()
        return [
            CategoryRecord(
                id=int(item["id"]),
                name=str(item["name"]),
                supercategory=str(item.get("supercategory", "default")),
            )
            for item in coco.get("categories", [])
        ]

    def load_images(self) -> List[ImageRecord]:
        coco = self.load_coco()
        images: List[ImageRecord] = []
        for image_info in coco.get("images", []):
            image_path = self._resolve_image_path(image_info)
            images.append(
                ImageRecord(
                    id=int(image_info["id"]),
                    file_name=str(image_info["file_name"]),
                    width=int(image_info["width"]),
                    height=int(image_info["height"]),
                    path=image_path,
                    license_id=image_info.get("license"),
                    date_captured=image_info.get("date_captured"),
                    extra=image_info.get("extra"),
                )
            )
        return images

    def load_bbox_prompts(
        self,
        images: Sequence[ImageRecord],
        categories: Sequence[CategoryRecord],
    ) -> Dict[int, List[PromptRecord]]:
        coco = self.load_coco()
        image_ids = {image.id for image in images}
        category_name_map = {category.id: category.name for category in categories}
        prompts_by_image_id: Dict[int, List[PromptRecord]] = {image.id: [] for image in images}

        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in image_ids:
                continue

            bbox = self._bbox_xywh_to_xyxy(annotation.get("bbox"))
            if bbox is None:
                continue

            category_id = int(annotation["category_id"])
            prompts_by_image_id[image_id].append(
                PromptRecord(
                    image_id=image_id,
                    category_id=category_id,
                    category_name=category_name_map.get(category_id, str(category_id)),
                    box_xyxy=bbox,
                    source_annotation_id=int(annotation["id"]),
                    iscrowd=int(annotation.get("iscrowd", 0)),
                )
            )
        return prompts_by_image_id

    def _resolve_image_path(self, image_info: Dict[str, Any]) -> Path:
        candidates = [image_info.get("file_name")]
        extra = image_info.get("extra") or {}
        if isinstance(extra, dict):
            candidates.append(extra.get("name"))

        for candidate in candidates:
            if not candidate:
                continue
            path = self.config.image_dir / str(candidate)
            if path.exists():
                return path

        raise FileNotFoundError(
            f"在 {self.config.image_dir} 中找不到图像: {image_info.get('file_name')}"
        )

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox: Optional[Sequence[float]]) -> Optional[List[float]]:
        if not bbox or len(bbox) != 4:
            return None
        x, y, w, h = [float(v) for v in bbox]
        return [x, y, x + w, y + h]


class SamSegmentor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._predictor = None

    def predict(self, image: ImageRecord, prompts: Sequence[PromptRecord]) -> List[MaskRecord]:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("请先安装 numpy，用于组织 SAM 的 bbox 输入。") from exc

        if not prompts:
            return []

        predictor = self._get_predictor()
        predictor.set_image(self._load_image_array(image.path))

        mask_records: List[MaskRecord] = []
        for prompt in prompts:
            # SAM 的 predictor.predict 要求 box 是 numpy array，而不是 Python list。
            box = np.asarray(prompt.box_xyxy, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            if len(masks) == 0:
                continue

            mask = masks[0]
            mask_records.append(
                MaskRecord(
                    image_id=image.id,
                    category_id=prompt.category_id,
                    category_name=prompt.category_name,
                    segmentation=self._encode_binary_mask(mask),
                    area=float(mask.sum()),
                    bbox_xywh=self._mask_to_bbox(mask),
                    score=float(scores[0]),
                    iscrowd=prompt.iscrowd,
                    prompt=prompt,
                )
            )
        return mask_records

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError("请先安装 segment-anything，并确保依赖可用。") from exc

        sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(self.config.sam_checkpoint))
        sam.to(self.config.device)
        self._predictor = SamPredictor(sam)
        return self._predictor

    @staticmethod
    def _load_image_array(image_path: Path) -> Any:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise ImportError("请先安装 pillow 和 numpy，用于读取图像。") from exc

        with Image.open(image_path).convert("RGB") as image:
            return np.asarray(image)

    def _encode_binary_mask(self, mask: Any) -> Dict[str, Any] | List[List[float]]:
        if self.config.use_rle:
            return self._encode_rle(mask)
        return self._encode_polygon(mask)

    @staticmethod
    def _mask_to_bbox(mask: Any) -> List[float]:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("请先安装 numpy，用于计算 mask bbox。") from exc

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        return [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0]

    @staticmethod
    def _encode_rle(mask: Any) -> Dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("请先安装 numpy，用于 RLE 编码。") from exc

        flat = np.asarray(mask, dtype="uint8").ravel(order="F")
        counts: List[int] = []
        last_value = 0
        run_length = 0

        for value in flat:
            value = int(value)
            if value == last_value:
                run_length += 1
            else:
                counts.append(run_length)
                run_length = 1
                last_value = value
        counts.append(run_length)

        return {
            "size": list(mask.shape),
            "counts": counts,
        }

    @staticmethod
    def _encode_polygon(mask: Any) -> List[List[float]]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError("polygon 导出需要安装 opencv-python 和 numpy。") from exc

        contours, _ = cv2.findContours(
            np.asarray(mask).astype("uint8"),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons: List[List[float]] = []
        for contour in contours:
            if len(contour) < 3:
                continue
            flattened = contour.reshape(-1, 2).astype(float).flatten().tolist()
            if len(flattened) >= 6:
                polygons.append(flattened)
        return polygons


class MaskPostProcessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(self, masks: Sequence[MaskRecord]) -> List[MaskRecord]:
        filtered: List[MaskRecord] = []
        for mask in masks:
            if mask.area < self.config.min_mask_area:
                continue
            if mask.score < self.config.min_score:
                continue
            filtered.append(mask)
        return filtered


class CocoExporter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def export(
        self,
        images: Sequence[ImageRecord],
        masks: Sequence[MaskRecord],
        categories: Sequence[CategoryRecord],
    ) -> Dict[str, Any]:
        # 保持标准 COCO 结构：annotation 使用 category_id，名称仍放在 categories 表里。
        output = {
            "images": [self._build_image_item(image) for image in images],
            "annotations": [],
            "categories": [self._build_category_item(category) for category in categories],
        }

        for annotation_id, mask in enumerate(masks, start=1):
            output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": mask.image_id,
                    "category_id": mask.category_id,
                    "segmentation": mask.segmentation,
                    "area": mask.area,
                    "bbox": mask.bbox_xywh,
                    "iscrowd": mask.iscrowd,
                    "score": mask.score,
                }
            )
        return output

    def save(self, output: Dict[str, Any]) -> None:
        self.config.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_json.open("w", encoding="utf-8") as fp:
            json.dump(output, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_image_item(image: ImageRecord) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": image.id,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
        }
        if image.license_id is not None:
            item["license"] = image.license_id
        if image.date_captured is not None:
            item["date_captured"] = image.date_captured
        if image.extra is not None:
            item["extra"] = image.extra
        return item

    @staticmethod
    def _build_category_item(category: CategoryRecord) -> Dict[str, Any]:
        return {
            "id": category.id,
            "name": category.name,
            "supercategory": category.supercategory,
        }


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.data_manager = DataManager(config)
        self.segmentor = SamSegmentor(config)
        self.post_processor = MaskPostProcessor(config)
        self.exporter = CocoExporter(config)

    def run(self) -> Dict[str, Any]:
        # 固定流程：读 COCO bbox 和类别 -> SAM -> 过滤 -> 导出。
        categories = self.data_manager.load_categories()
        images = self.data_manager.load_images()
        prompts_by_image_id = self.data_manager.load_bbox_prompts(images, categories)

        all_masks: List[MaskRecord] = []
        progress = ProgressBar(total=len(images))
        for index, image in enumerate(images, start=1):
            prompts = prompts_by_image_id.get(image.id, [])
            masks = self.segmentor.predict(image, prompts)
            filtered_masks = self.post_processor.process(masks)
            all_masks.extend(filtered_masks)
            progress.update(
                current=index,
                extra_text=(
                    f"当前图片 {image.file_name} | "
                    f"框 {len(prompts)} 个 | "
                    f"新增标注 {len(filtered_masks)} 个 | "
                    f"累计标注 {len(all_masks)} 个"
                ),
            )
        progress.finish()

        output = self.exporter.export(images, all_masks, categories)
        self.exporter.save(output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 COCO bbox 作为提示，调用 SAM 生成掩膜。")
    parser.add_argument("--image-dir", type=Path, default='yolo to SAM111.coco\\test', help="图像目录。")
    parser.add_argument("--coco-json", type=Path, default='yolo to SAM111.coco\\test\_annotations.coco.json', help="输入 COCO 标注文件。")
    parser.add_argument("--output-json", type=Path, default='reslut.coco\\result.json', help="输出结果路径。")
    parser.add_argument("--sam-checkpoint", type=Path, default='sam_vit_b_01ec64.pth', help="SAM 权重文件路径。")
    parser.add_argument("--sam-model-type", type=str, default="vit_b", help="SAM 模型类型。")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备，如 cuda 或 cpu。")
    parser.add_argument("--min-mask-area", type=int, default=64, help="最小 mask 面积阈值。")
    parser.add_argument("--min-score", type=float, default=0.5, help="最小保留分数阈值。")
    parser.add_argument(
        "--polygon",
        action="store_true",
        help="使用 polygon 导出，默认导出 RLE。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        image_dir=args.image_dir,
        coco_json=args.coco_json,
        output_json=args.output_json,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        min_mask_area=args.min_mask_area,
        min_score=args.min_score,
        use_rle=not args.polygon,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = PipelineRunner(config)
    output = runner.run()
    print(
        f"标注完成: images={len(output['images'])}, "
        f"annotations={len(output['annotations'])}, "
        f"output={config.output_json}"
    )


if __name__ == "__main__":
    main()
