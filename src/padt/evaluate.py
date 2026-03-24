
import torch
from transformers import AutoProcessor
from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm
import json

# ----------------------
# Config / Paths
# ----------------------
MODEL_PATH = "/home/jovyan/models/padt_7b_rec"
DATA_ROOT_PATH = Path("~/datasets/synthetic_data/data/real_data_v2/custom_subset").expanduser()
ANNOTATIONS_PATH = DATA_ROOT_PATH / "annotations.json"
IMAGES_PATH = DATA_ROOT_PATH / "images"
REFEXPS_PATH = DATA_ROOT_PATH / "refexps.json"

SAVE_DEBUG_IMAGES = True

DEBUG_SAVE_DIR = Path("./debug_images")
if SAVE_DEBUG_IMAGES:
    DEBUG_SAVE_DIR.mkdir(exist_ok=True)

# ----------------------
# Helpers
# ----------------------
def resize_image(image: Image.Image, max_pixels: int = 768 * 28 * 28) -> Image.Image:
    orig_w, orig_h = image.size
    total_pixels = orig_w * orig_h

    if total_pixels <= max_pixels:
        return image

    scale = (max_pixels / total_pixels) ** 0.5
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)

    return image.resize((new_w, new_h), Image.LANCZOS)


def sanitize_box(box, img_w, img_h):
    x0 = max(0, min(img_w, box[0] * img_w))
    y0 = max(0, min(img_h, box[1] * img_h))
    x1 = max(0, min(img_w, box[2] * img_w))
    y1 = max(0, min(img_h, box[3] * img_h))
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    print(f"[DEBUG] Box to be drawn: ({x0}, {y0}), {x1}, {y1}")
    return [x0, y0, x1, y1]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW, interH = max(0.0, xB - xA), max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


def disable_flash_attn(module):
    for name, child in module.named_children():
        if hasattr(child, "use_flash_attn"):
            print(f"[DEBUG] Disabling flash_attn in {name}")
            child.use_flash_attn = False
        disable_flash_attn(child)


# ----------------------
# Load annotations & refexps
# ----------------------
with open(ANNOTATIONS_PATH, "r") as f:
    ann_json = json.load(f)

with open(REFEXPS_PATH, "r") as f:
    refexps = json.load(f)

image_id_to_file = {img["id"]: img["file_name"] for img in ann_json.get("images", [])}
ann_id_to_ann    = {ann["id"]: ann for ann in ann_json.get("annotations", [])}

# ----------------------
# Load model
# ----------------------
print("[DEBUG] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("[DEBUG] Loading model...")
model = PaDTForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = VisonTextProcessingClass(
    processor,
    model.config.vision_config.spatial_merge_size
)
processor.prepare(model.model.embed_tokens.weight.shape[0])

model.eval()
print("[DEBUG] Model and processor loaded to GPU")

# ----------------------
# Evaluation
# ----------------------
total_iou = 0.0
count = 0
correct_50 = 0
correct_75 = 0

for ref in tqdm(refexps):
    ann_id   = ref["ann_id"]
    image_id = ref["image_id"]
    sent     = ref["sentences"][0]["sent"]

    # Validate all lookups before doing any work
    if ann_id not in ann_id_to_ann:
        print(f"[WARNING] ann_id {ann_id} not found in annotations, skipping")
        continue
    if image_id not in image_id_to_file:
        print(f"[WARNING] image_id {image_id} not found in images, skipping")
        continue

    image_path = IMAGES_PATH / image_id_to_file[image_id]
    if not image_path.exists():
        print(f"[WARNING] Image file not found: {image_path}, skipping")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"[WARNING] Could not read image: {image_path}, skipping")
        continue

    orig_w, orig_h = image.size
    image = resize_image(image, max_pixels=768 * 28 * 28)

    ann = ann_id_to_ann[ann_id]
    bbox = ann.get("bbox", [0, 0, 1, 1])
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    gt_box = [x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h]

    print(f"[DEBUG] ref_id={ref['ref_id']} sent='{sent}'")

    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": sent}
        ]
    }]

    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(message)

    prompt_inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False
    ).to("cuda")

    with torch.inference_mode():
        gen_ret = model.generate(
            **prompt_inputs,
            use_cache=True,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            repetition_penalty=1.05,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        prompt_len = prompt_inputs["input_ids"].size(1)
        completion_ids = gen_ret['sequences'][:, prompt_len:]

        completions, feats, _, _, _ = parseVRTintoCompletion(
            processor,
            completion_ids,
            gen_ret['hidden_states'],
            torch.Tensor([False])
        )

        print("[DEBUG] completion:", completions[0])

        low_res_image_embeds  = gen_ret.past_image_embeds
        high_res_image_embeds = gen_ret.past_high_res_image_embeds
        visual_pe             = gen_ret.past_visual_pe

        try:
            decoded = model.vl_decode(
                feats,
                low_res_image_embeds,
                high_res_image_embeds,
                prompt_inputs['image_grid_thw'],
                visual_pe
            )
        except Exception as e:
            print(f"[WARNING] vl_decode failed: {e}")
            continue

        pred_boxes = decoded.get('pred_boxes', None)
        if pred_boxes is None or pred_boxes.shape[0] == 0:
            print(f"[WARNING] No predicted boxes for ref_id={ref['ref_id']}")
            continue

        # Convert [cx, cy, w, h] -> [x1, y1, x2, y2] (all normalized)
        cx, cy, pw, ph = pred_boxes[0].tolist()
        pred_box = [cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2]

        print(f"[DEBUG] pred_box (xyxy): {pred_box}")
        print(f"[DEBUG] gt_box   (xyxy): {gt_box}")
        print(f"[DEBUG] image size:      {image.size}")

        iou = compute_iou(pred_box, gt_box)
        total_iou += iou
        count += 1
        if iou >= 0.5:
            correct_50 += 1
        if iou >= 0.75:
            correct_75 += 1

        print(f"[DEBUG] IoU: {iou:.4f}")

        if SAVE_DEBUG_IMAGES:
            draw = ImageDraw.Draw(image)
            w, h = image.size

            gt_px   = sanitize_box(gt_box,   w, h)
            pred_px = sanitize_box(pred_box, w, h)

            draw.rectangle(gt_px,   outline="green", width=3)
            draw.rectangle(pred_px, outline="red",   width=3)

            save_path = DEBUG_SAVE_DIR / f"ref{ref['ref_id']}_iou{iou:.2f}.png"
            image.save(save_path)

# ----------------------
# Results
# ----------------------
results = {
    "mean_iou": total_iou/count if count else 0,
    "acc@0.5":  correct_50/count if count else 0,
    "acc@0.75": correct_75/count if count else 0,
    "total":    count
}

print("\nEvaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
