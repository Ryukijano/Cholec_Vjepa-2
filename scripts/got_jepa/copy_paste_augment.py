#!/usr/bin/env python3
"""
Copy-paste augmentation for rare surgical tool classes.

Addresses the severe class imbalance in CholecTrack20:
  bipolar: 70.2% | irrigator: 9.8% | grasper: 6.5% | scissors: 4.9% | clipper: 4.4% | hook: 4.2%

Strategy: Cut tool instances from images where they appear and paste them onto
random training images. This creates new training samples for rare classes
without duplicating entire images.

Usage:
    python scripts/got_jepa/copy_paste_augment.py \
        --source_dir /scratch/kcwp264/data/surgi_world_track/cholec20_coco/train \
        --output_dir /scratch/kcwp264/data/surgi_world_track/cholec20_coco_train_augmented \
        --target_per_class 3000
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

RARE_CLASSES = ["clipper", "hook", "scissors", "grasper"]
COMMON_CLASSES = ["bipolar", "irrigator"]


def load_coco(annotation_path: str) -> dict:
    with open(annotation_path) as f:
        return json.load(f)


def save_coco(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def extract_instances_by_class(coco_data: dict, image_dir: str) -> dict[str, list[dict]]:
    """Group annotations by class, attaching image info and crop data."""
    images = {img["id"]: img for img in coco_data["images"]}
    cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}

    by_class = defaultdict(list)
    for ann in coco_data["annotations"]:
        cat_name = cat_map[ann["category_id"]]
        img_info = images[ann["image_id"]]
        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        by_class[cat_name].append({
            "annotation": ann,
            "image_info": img_info,
            "image_path": img_path,
        })
    return by_class


def cut_and_paste(
    source_instance: dict,
    target_image: Image.Image,
    target_annotations: list,
    target_img_info: dict,
    category_id: int,
) -> tuple[Image.Image, dict | None]:
    """Cut a tool instance from source and paste onto target image."""
    ann = source_instance["annotation"]
    src_img = Image.open(source_instance["image_path"]).convert("RGB")

    x, y, w, h = ann["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Add small padding for context
    pad = max(5, int(min(w, h) * 0.1))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(src_img.width, x + w + pad)
    y2 = min(src_img.height, y + h + pad)

    crop = src_img.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size

    # Find a random spot on target image that fits the crop
    if crop_w >= target_image.width or crop_h >= target_image.height:
        return target_image, None

    max_x = target_image.width - crop_w
    max_y = target_image.height - crop_h
    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)

    # Simple paste (no blending for surgical tools — they have distinct shapes)
    # Could add Poisson blending here for more realism
    target_image.paste(crop, (paste_x, paste_y))

    new_bbox = [paste_x, paste_y, crop_w, crop_h]
    new_ann = {
        "id": -1,  # Will be assigned later
        "image_id": target_img_info["id"],
        "category_id": category_id,
        "bbox": new_bbox,
        "area": float(crop_w * crop_h),
        "iscrowd": 0,
        "segmentation": [],
    }

    return target_image, new_ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source COCO dataset directory (with _annotations.coco.json and images/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for augmented dataset")
    parser.add_argument("--target_per_class", type=int, default=3000,
                        help="Target number of instances per rare class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    source_anno = os.path.join(args.source_dir, "_annotations.coco.json")
    # COCO json file_name includes 'images/' prefix (e.g. 'images/VID02_006701.png')
    # so source_images should be the source_dir itself, not source_dir/images
    source_images = args.source_dir
    coco = load_coco(source_anno)

    # Count current instances
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
    counts = Counter()
    for ann in coco["annotations"]:
        counts[cat_map[ann["category_id"]]] += 1

    print("Original class distribution:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print()

    # Group instances by class
    by_class = extract_instances_by_class(coco, source_images)
    for name, instances in by_class.items():
        print(f"  {name}: {len(instances)} source instances available")

    # Calculate how many copies we need per rare class
    augmentation_plan = {}
    for cls in RARE_CLASSES:
        current = counts.get(cls, 0)
        needed = max(0, args.target_per_class - current)
        if needed > 0 and cls in by_class:
            augmentation_plan[cls] = needed
            print(f"\n  Plan: {cls} needs {needed} more instances (have {current}, target {args.target_per_class})")

    if not augmentation_plan:
        print("No augmentation needed — all rare classes meet target.")
        return

    # Prepare output directory — file_name in COCO json includes 'images/' prefix
    # so output_dir itself is the base, not output_dir/images
    out_images = args.output_dir
    os.makedirs(os.path.join(out_images, "images"), exist_ok=True)

    # Copy original images and annotations
    print("\nCopying original images...")
    original_images = {img["id"]: img for img in coco["images"]}
    next_img_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = max(ann["id"] for ann in coco["annotations"]) + 1

    # Copy all original images
    for img_info in tqdm(coco["images"], desc="Copying originals"):
        src = os.path.join(source_images, img_info["file_name"])
        dst = os.path.join(out_images, img_info["file_name"])  # file_name already has 'images/' prefix
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Generate augmented images
    all_images = list(coco["images"])
    all_annotations = list(coco["annotations"])

    print("\nGenerating copy-paste augmented images...")
    total_created = 0

    for cls_name, num_needed in augmentation_plan.items():
        source_instances = by_class[cls_name]
        cat_id = cat_ids[cls_name]
        created = 0

        pbar = tqdm(total=num_needed, desc=f"Augmenting {cls_name}")
        while created < num_needed:
            # Pick a random target image
            target_img_info = random.choice(all_images)
            target_path = os.path.join(out_images, target_img_info["file_name"])
            if not os.path.exists(target_path):
                continue

            target_img = Image.open(target_path).convert("RGB")

            # Pick 1-3 source instances of the rare class to paste
            num_paste = random.randint(1, min(3, len(source_instances)))
            pasted_anns = []
            for _ in range(num_paste):
                src_inst = random.choice(source_instances)
                target_img, new_ann = cut_and_paste(
                    src_inst, target_img, all_annotations, target_img_info, cat_id
                )
                if new_ann is not None:
                    new_ann["id"] = next_ann_id
                    next_ann_id += 1
                    pasted_anns.append(new_ann)

            if pasted_anns:
                # Save augmented image with new filename
                # Strip 'images/' prefix from original file_name for the unique part
                orig_basename = os.path.basename(target_img_info["file_name"])
                aug_filename = f"images/aug_{cls_name}_{created}_{orig_basename}"
                aug_path = os.path.join(out_images, aug_filename)
                target_img.save(aug_path)

                # Add new image entry
                new_img = {
                    "id": next_img_id,
                    "file_name": aug_filename,
                    "width": target_img.width,
                    "height": target_img.height,
                }
                next_img_id += 1
                all_images.append(new_img)

                # Update image_id in annotations
                for ann in pasted_anns:
                    ann["image_id"] = new_img["id"]
                    all_annotations.append(ann)

                created += len(pasted_anns)
                total_created += len(pasted_anns)
                pbar.update(len(pasted_anns))

        pbar.close()

    # Save augmented annotations
    coco["images"] = all_images
    coco["annotations"] = all_annotations
    save_coco(coco, os.path.join(args.output_dir, "_annotations.coco.json"))

    # Final distribution
    final_counts = Counter()
    for ann in all_annotations:
        final_counts[cat_map[ann["category_id"]]] += 1
    print(f"\nFinal class distribution ({len(all_images)} images, {len(all_annotations)} annotations):")
    for name, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} ({100*count/sum(final_counts.values()):.1f}%)")
    print(f"\nTotal new instances created: {total_created}")


if __name__ == "__main__":
    main()
