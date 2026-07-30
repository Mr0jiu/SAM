from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MASK_COLOR = (60, 120, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 COCO segmentation 渲染为掩膜叠加图，便于检查标注质量。")
    parser.add_argument("--image-dir", type=Path, default=Path(r"yolo to SAM111.coco\test"), help="图像目录。")
    parser.add_argument("--coco-json", type=Path, default=Path(r"reslut.coco\result.json"), help="COCO 标注 JSON（包含 segmentation）。")
    parser.add_argument("--out-dir", type=Path, default=Path(r"test"), help="可视化输出目录。")
    parser.add_argument("--alpha", type=float, default=0.45, help="掩膜叠加透明度。")
    parser.add_argument("--max-images", type=int, default=50, help="最多渲染多少张图片。")
    parser.add_argument("--start", type=int, default=0, help="从 images 列表的第几个开始渲染。")
    parser.add_argument("--open", action="store_true", help="每张图渲染后用系统默认图片查看器打开。")
    return parser.parse_args()


def load_coco(coco_json: Path) -> Dict[str, Any]:
    with coco_json.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_image_path(image_dir: Path, image_info: Dict[str, Any]) -> Path:
    candidates = [image_info.get("file_name")]
    extra = image_info.get("extra") or {}
    if isinstance(extra, dict):
        candidates.append(extra.get("name"))

    for candidate in candidates:
        if not candidate:
            continue
        path = image_dir / str(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"在 {image_dir} 中找不到图像: {image_info.get('file_name')}")


def decode_rle_counts_list(seg: Dict[str, Any]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("请先安装 numpy。") from exc

    h, w = [int(v) for v in seg["size"]]
    counts = seg["counts"]
    total = h * w
    flat: List[int] = []
    val = 0
    for run_len in counts:
        run_len = int(run_len)
        if run_len <= 0:
            continue
        flat.extend([val] * run_len)
        val = 1 - val

    if len(flat) < total:
        flat.extend([0] * (total - len(flat)))

    array = np.asarray(flat[:total], dtype="uint8")
    return array.reshape((h, w), order="F")


def decode_segmentation(segmentation: Any, height: int, width: int) -> Optional[Any]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("请先安装 pillow 和 numpy。") from exc

    if segmentation is None:
        return None

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            return decode_rle_counts_list(segmentation)
        if isinstance(counts, str):
            try:
                from pycocotools import mask as mask_utils
            except ImportError:
                return None
            size = segmentation.get("size", [height, width])
            decoded = mask_utils.decode({"size": size, "counts": counts.encode("utf-8")})
            return np.asarray(decoded, dtype="uint8")
        return None

    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline=1, fill=1)
        return np.asarray(canvas, dtype="uint8")

    return None


def overlay_masks(
    image_rgb: Any,
    masks: Sequence[Any],
    colors: Sequence[Tuple[int, int, int]],
    alpha: float,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("请先安装 numpy。") from exc

    overlay = image_rgb.copy()
    alpha = float(max(0.0, min(alpha, 1.0)))
    for mask, color in zip(masks, colors):
        if mask is None:
            continue
        m = (mask > 0).astype("uint8")
        if m.ndim != 2:
            continue
        overlay[m > 0] = (overlay[m > 0] * (1.0 - alpha) + np.asarray(color) * alpha).astype("uint8")
    return overlay


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.coco_json)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    category_name_map = {int(c["id"]): str(c["name"]) for c in categories if "id" in c and "name" in c}
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("请先安装 pillow 和 numpy。") from exc

    selected_images = images[args.start : args.start + max(args.max_images, 0)]
    for img_info in selected_images:
        image_id = int(img_info["id"])
        img_path = resolve_image_path(args.image_dir, img_info)
        image = Image.open(img_path).convert("RGB")
        image_np = np.asarray(image).copy()

        anns = annotations_by_image_id.get(image_id, [])
        masks: List[Any] = []
        colors: List[Tuple[int, int, int]] = []

        for ann in anns:
            seg = ann.get("segmentation")
            mask = decode_segmentation(seg, int(img_info["height"]), int(img_info["width"]))
            if mask is None:
                continue
            masks.append(mask)
            colors.append(MASK_COLOR)

        overlay_np = overlay_masks(image_np, masks, colors, alpha=args.alpha)
        overlay_img = Image.fromarray(overlay_np)

        draw = ImageDraw.Draw(overlay_img)
        for ann in anns:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            category_id = int(ann.get("category_id", 0))
            name = category_name_map.get(category_id, str(category_id))
            color = MASK_COLOR
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x + 2, y + 2), name, fill=color)

        out_path = args.out_dir / f"{Path(img_info['file_name']).stem}_vis.jpg"
        overlay_img.save(out_path, quality=95)

        if args.open:
            os.startfile(str(out_path))


if __name__ == "__main__":
    main()
