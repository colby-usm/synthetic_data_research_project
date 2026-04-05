import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import numpy as np
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ------------------------
# Load config
# ------------------------
with open('train_cfg.json', 'r', encoding='utf-8') as file:
    train_cfg = json.load(file)

# ------------------------
# Paths
# ------------------------
MODEL_PATH       = Path(train_cfg.get("padt_7b_rec"))  # directory to check / download
DATA_ROOT_PATH   = Path(train_cfg.get("directories").get("synthetic_data", "synthetic_data"))
ANNOTATIONS_PATH = DATA_ROOT_PATH / "annotations.json"
IMAGES_PATH      = DATA_ROOT_PATH / "images"
REFEXPS_PATH     = DATA_ROOT_PATH / "refexps.json"

ZOO_ROOT = Path(train_cfg.get("training_sessions", "training_sessions"))
ZOO_ROOT.mkdir(parents=True, exist_ok=True)

RUN_ID   = max([0] + [int(p.name.split("_")[-1]) for p in ZOO_ROOT.glob("run_*") if p.name.startswith("run_")]) + 1
RUN_PATH = ZOO_ROOT / f"run_{RUN_ID}"
MODEL_SAVE_PATH = RUN_PATH / "models"
RUN_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.mkdir(exist_ok=True)

LOG_PATH  = MODEL_SAVE_PATH / "train_log.json"
PLOT_PATH = MODEL_SAVE_PATH / "training_curves.png"

train_log = []

print(f"[INFO] Saving checkpoints to: {MODEL_SAVE_PATH}")

# ------------------------
# Train config
# ------------------------
DEVICE         = train_cfg.get("train", {}).get("device","cpu")
EPOCHS         = train_cfg.get("train", {}).get("epochs", 100)
LR             = train_cfg.get("train", {}).get("lr", 1e-5)
PLOT_EVERY     = train_cfg.get("train", {}).get("plot_every", 1)
BATCH_SIZE     = train_cfg.get("train", {}).get("batch_size", 4)
MAX_IMAGE_SIZE = train_cfg.get("train", {}).get("max_image_size", 448*448)

# ------------------------
# LoRA config
# ------------------------
LORA_R         = train_cfg.get("lora", {}).get("r_value", 8)
LORA_ALPHA     = train_cfg.get("lora", {}).get("alpha_value", 16)
TARGET_MODULES = train_cfg.get("lora", {}).get("finetuning_modules", ["q_proj","k_proj","v_proj","o_proj"])
LORA_DROPOUT   = train_cfg.get("lora", {}).get("dropout", 0.05)
LORA_TASK      = train_cfg.get("lora", {}).get("task", "CASUAL_LM")
LORA_BIAS      = train_cfg.get("lora", {}).get("bias", None)

# ------------------------
# Load / download model
# ------------------------
if MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists() and (MODEL_PATH / "pytorch_model.bin").exists():
    print(f"[INFO] Loading model from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
else:
    print("[INFO] Downloading model from Hugging Face...")
    processor = AutoProcessor.from_pretrained("PaDT-MLLM/PaDT_REC_7B")
    model = AutoModelForSeq2SeqLM.from_pretrained("PaDT-MLLM/PaDT_REC_7B")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

print("[INFO] Model and processor ready.")


# Helpers
def resize_image(image: Image.Image, max_pixels: int = 448*448) -> Image.Image:
    orig_w, orig_h = image.size
    scale = min(1.0, (max_pixels / (orig_w * orig_h)) ** 0.5)
    patch_merge = 14 * 2
    new_w = max(patch_merge, (int(orig_w * scale) // patch_merge) * patch_merge)
    new_h = max(patch_merge, (int(orig_h * scale) // patch_merge) * patch_merge)
    return image.resize((new_w, new_h), Image.LANCZOS)


def bbox_to_patches(bbox, img_w, img_h):
    """COCO [x, y, w, h] → array of patch indices covering the bbox."""
    x, y, w, h = bbox
    patch_w = round(img_w / 28)

    px0 = int(x / 28)
    py0 = int(y / 28)
    px1 = min(patch_w - 1, int((x + w) / 28))
    py1 = min(round(img_h / 28) - 1, int((y + h) / 28))

    patches = []
    for py in range(py0, py1 + 1):
        for px in range(px0, px1 + 1):
            patches.append(py * patch_w + px)

    return np.array(patches), patch_w


def select_patches(patches, patch_w):
    """Select boundary + center patches, matching PaDT's training strategy."""
    if len(patches) == 0:
        return patches

    patches_x = patches % patch_w
    patches_y = patches // patch_w

    left_m   = patches_x == patches_x.min()
    right_m  = patches_x == patches_x.max()
    top_m    = patches_y == patches_y.min()
    bottom_m = patches_y == patches_y.max()
    centre_m = (left_m + right_m + top_m + bottom_m) == 0

    centre_patches = patches[centre_m] if centre_m.sum() > 0 else patches

    return np.array([
        np.random.choice(centre_patches),
        np.random.choice(patches[left_m]),
        np.random.choice(patches[top_m]),
        np.random.choice(patches[right_m]),
        np.random.choice(patches[bottom_m]),
    ])


def compute_accuracy(logits, labels, vocab_size):
    """Token-level accuracy on non-masked (target) tokens only."""
    shift_logits = logits[:, :-1, :vocab_size]
    shift_labels = labels[:, 1:]
    # Only evaluate on non-VRT, non-masked tokens
    mask = (shift_labels != -100) & (shift_labels < vocab_size)
    if mask.sum() == 0:
        return 0.0
    preds   = shift_logits.argmax(dim=-1)
    correct = (preds[mask] == shift_labels[mask]).float().sum()
    return (correct / mask.sum()).item()


def giou_loss(pred_xyxy, gt_xyxy):
    """GIoU loss between predicted and GT boxes, both in [x1,y1,x2,y2] normalized."""
    lt    = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
    rb    = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    wh    = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area2 = (gt_xyxy[:, 2]   - gt_xyxy[:, 0])   * (gt_xyxy[:, 3]   - gt_xyxy[:, 1])
    union = (area1 + area2 - inter).clamp(min=1e-6)
    iou   = inter / union

    enclosing_lt   = torch.min(pred_xyxy[:, :2], gt_xyxy[:, :2])
    enclosing_rb   = torch.max(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    enclosing_wh   = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=1e-6)

    giou = iou - (enclosing_area - union) / enclosing_area
    return (1 - giou).sum(), iou.detach()


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)


def rolling_mean(vals, w):
    out = []
    for i in range(len(vals)):
        start = max(0, i - w + 1)
        out.append(sum(vals[start:i+1]) / (i - start + 1))
    return out


# Plot
def save_plot(state, path: Path, step_window: int = 50):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training curves", fontsize=13, fontweight="bold")

    steps  = list(range(1, len(state["batch_losses"]) + 1))
    window = min(step_window, max(1, len(state["batch_losses"])))

    def epoch_x(epoch_vals):
        n     = len(epoch_vals)
        total = len(steps)
        return [round((i + 1) * total / n) for i in range(n)]

    ax = axes[0]
    ax.plot(steps, state["batch_losses"], alpha=0.25, color="steelblue", linewidth=0.8, label="batch loss")
    ax.plot(steps, rolling_mean(state["batch_losses"], window), color="steelblue", linewidth=1.8, label=f"rolling mean ({window})")
    if state["epoch_losses"]:
        ax.plot(epoch_x(state["epoch_losses"]), state["epoch_losses"], "o--", color="navy", markersize=5, linewidth=1.2, label="epoch avg")
    ax.set_title("Loss"); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, state["batch_accs"], alpha=0.25, color="darkorange", linewidth=0.8, label="batch acc")
    ax.plot(steps, rolling_mean(state["batch_accs"], window), color="darkorange", linewidth=1.8, label=f"rolling mean ({window})")
    if state["epoch_accs"]:
        ax.plot(epoch_x(state["epoch_accs"]), state["epoch_accs"], "o--", color="saddlebrown", markersize=5, linewidth=1.2, label="epoch avg")
    ax.set_title("Token Accuracy (target tokens)"); ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scale_bbox(bbox, orig_w, orig_h, new_w, new_h):
    x, y, w, h = bbox
    return [x * new_w / orig_w, y * new_h / orig_h,
            w * new_w / orig_w, h * new_h / orig_h]


# Dataset
class RefExpBBoxDataset(Dataset):
    def __init__(self, annotations_path, refexps_path, images_path):
        with open(annotations_path) as f:
            ann_json = json.load(f)
        with open(refexps_path) as f:
            refexps_json = json.load(f)

        self.images      = {img["id"]: img["file_name"] for img in ann_json.get("images", [])}
        self.annotations = {ann["id"]: ann for ann in ann_json.get("annotations", [])}
        self.images_path = images_path
        self.data        = []

        for ref in refexps_json:
            if ref.get("ref_id") is None:
                continue
            ann_id   = ref.get("ann_id")
            image_id = ref.get("image_id")
            if ann_id not in self.annotations or image_id not in self.images:
                continue
            sentences = [s["sent"] for s in ref.get("sentences", []) if s.get("sent")]
            if not sentences:
                continue
            ann  = self.annotations[ann_id]
            bbox = ann.get("bbox", [0, 0, 1, 1])
            self.data.append({
                "ref_id":   ref["ref_id"],
                "ann_id":   ann_id,
                "image_id": image_id,
                "bbox":     bbox,
                "sentences": sentences,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item      = self.data[idx]
        image_path = self.images_path / self.images[item["image_id"]]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            return None

        orig_w, orig_h = image.size
        image          = resize_image(image)
        new_w, new_h   = image.size
        bbox           = scale_bbox(item["bbox"], orig_w, orig_h, new_w, new_h)

        return {"image": image, "bbox": bbox, "sentences": item["sentences"]}


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return {
        "image":     [item["image"]     for item in batch],
        "bbox":      [item["bbox"]      for item in batch],
        "sentences": [item["sentences"] for item in batch],
    }


def main():
    print("[INFO] Loading model...")
    model = PaDTForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)


    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias=LORA_BIAS, task_type=LORA_TASK,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor = VisonTextProcessingClass(processor, model.config.vision_config.spatial_merge_size)
    processor.prepare(model.config.vocab_size)

    vocab_size = model.config.vocab_size
    loss_fct   = torch.nn.CrossEntropyLoss(ignore_index=-100)

    dataset    = RefExpBBoxDataset(ANNOTATIONS_PATH, REFEXPS_PATH, IMAGES_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    print(f"[INFO] Dataset size: {len(dataset)} bboxes")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR,
    )

    plot_state = {"epoch_losses": [], "epoch_accs": [], "batch_losses": [], "batch_accs": [], "step": 0}
    best_acc   = 0.0

    print(f"\n{'Epoch':>6} {'Step':>7} {'Loss':>10} {'Acc':>10}")
    print("-" * 38)

    for epoch in range(EPOCHS):
        epoch_losses, epoch_accs = [], []

        # Track gradient accumulation
        accum_count    = 0
        accum_loss_sum = 0.0
        accum_acc_sum  = 0.0

        optimizer.zero_grad()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True):
            image     = batch["image"][0]
            bbox      = batch["bbox"][0]
            sentences = batch["sentences"][0]

            img_w, img_h = image.size
            patches, patch_w = bbox_to_patches(bbox, img_w, img_h)

            if len(patches) == 0:
                continue

            n_sentences   = len(sentences)
            sample_loss   = 0.0
            sample_acc    = 0.0

            for sent in sentences:
                # Re-select patches per sentence for augmentation
                pick_patch = select_patches(patches, patch_w)
                vrt_str    = processor.pid2vrt(pick_patch)
                target_str = f'The "{sent}" refers to {vrt_str} in this image.{processor.tokenizer.eos_token}'

                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  sent},
                    ],
                }]

                prompt_text = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                full_text = prompt_text + target_str

                image_inputs, _ = process_vision_info(message)

                inputs = processor(
                    text=[full_text],
                    images=image_inputs,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                image_grid_thw = inputs.get("image_grid_thw")
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(DEVICE)

                with torch.no_grad():
                    prompt_only = processor(
                        text=[prompt_text],
                        images=None,
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                prompt_len = prompt_only["input_ids"].shape[1]

                input_ids      = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                pixel_values   = inputs["pixel_values"].to(DEVICE, dtype=torch.bfloat16)

                # Labels: mask prompt, keep full completion including VRT tokens
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                )

                logits = outputs.logits  # [B, L, vocab + n_patches]

                # SFT loss over vrt vocab
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()

                labels_vrt_only = shift_labels.clone()
                labels_vrt_only[(shift_labels < vocab_size) & (shift_labels != -100)] = -100


                sft_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    labels_vrt_only.view(-1).to(shift_logits.device),
                )

                # Bbox loss via vl_decoder
                bbox_loss = torch.tensor(0.0, device=DEVICE)
                completion_ids = input_ids[:, prompt_len:]
                hidden_states  = (
                    torch.stack(outputs.hidden_states, dim=1)   # [B, n_layers, L, D]
                    [:, -1:, prompt_len-1:-1]                   # last layer, completion positions
                    .permute(2, 1, 0, 3)                        # [L, 1, B, D]
                    .unsqueeze(-2)
                    .contiguous()
                )

                _, feats, _, _, _ = parseVRTintoCompletion(
                    processor,
                    completion_ids,
                    hidden_states,
                    torch.Tensor([False]),
                )

                if len(feats[0]) > 0:
                    decoded = model.vl_decode(
                        feats,
                        outputs.past_image_embeds,
                        outputs.past_high_res_image_embeds,
                        image_grid_thw,
                        outputs.past_visual_pe,
                    )
                    pred_boxes = decoded["pred_boxes"].float()  # [N, 4] cxcywh

                    x, y, bw, bh = bbox
                    gt_xyxy = torch.tensor(
                        [[x/img_w, y/img_h, (x+bw)/img_w, (y+bh)/img_h]],
                        device=DEVICE, dtype=torch.float32,
                    )
                    gt_cxcywh = xyxy_to_cxcywh(gt_xyxy)

                    pred_xyxy = cxcywh_to_xyxy(pred_boxes)
                    giou, _   = giou_loss(pred_xyxy, gt_xyxy)
                    l1        = F.l1_loss(pred_boxes, gt_cxcywh, reduction='sum')
                    bbox_loss = giou + l1

                # Scale by 1/BATCH_SIZE for gradient accumulation
                loss = (sft_loss + bbox_loss) / BATCH_SIZE
                loss.backward()
                torch.cuda.empty_cache()

                with torch.no_grad():
                    acc = compute_accuracy(logits, labels, vocab_size)

                sample_loss += (sft_loss.item() + bbox_loss.item())
                sample_acc  += acc / n_sentences

            accum_loss_sum += sample_loss
            accum_acc_sum  += sample_acc
            accum_count    += 1

            # Step when we've accumulated BATCH_SIZE samples
            if accum_count == BATCH_SIZE:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accum_loss_sum / BATCH_SIZE
                avg_acc  = accum_acc_sum  / BATCH_SIZE

                plot_state["step"] += 1
                plot_state["batch_losses"].append(avg_loss)
                plot_state["batch_accs"].append(avg_acc)
                epoch_losses.append(avg_loss)
                epoch_accs.append(avg_acc)

                print(
                    f"\r{epoch+1:>6} {plot_state['step']:>7} "
                    f"{avg_loss:>10.4f} {avg_acc:>9.3f}",
                    end="", flush=True,
                )

                if plot_state["step"] % PLOT_EVERY == 0:
                    save_plot(plot_state, PLOT_PATH)

                accum_count    = 0
                accum_loss_sum = 0.0
                accum_acc_sum  = 0.0

        # Flush any remaining accumulated gradients at end of epoch
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = accum_loss_sum / accum_count
            avg_acc  = accum_acc_sum  / accum_count
            plot_state["step"] += 1
            plot_state["batch_losses"].append(avg_loss)
            plot_state["batch_accs"].append(avg_acc)
            epoch_losses.append(avg_loss)
            epoch_accs.append(avg_acc)

            accum_count    = 0
            accum_loss_sum = 0.0
            accum_acc_sum  = 0.0

        e_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        e_acc  = sum(epoch_accs)   / max(len(epoch_accs),   1)
        plot_state["epoch_losses"].append(e_loss)
        plot_state["epoch_accs"].append(e_acc)
        save_plot(plot_state, PLOT_PATH)

        print(f"\n[Epoch {epoch+1}] avg loss: {e_loss:.4f}  avg acc: {e_acc:.3f}  → plot saved to {PLOT_PATH}")

        train_log.append({"epoch": epoch + 1, "avg_loss": e_loss, "avg_acc": e_acc})
        with open(LOG_PATH, "w") as f:
            json.dump(train_log, f, indent=2)

        # Save every epoch
        epoch_path = MODEL_SAVE_PATH / f"epoch_{epoch+1}"
        epoch_path.mkdir(exist_ok=True)
        model.save_pretrained(epoch_path)
        processor.save_pretrained(epoch_path)
        print(f"[INFO] Saved checkpoint at: {epoch_path}")

        # Save best
        if e_acc > best_acc:
            best_acc = e_acc
            best_path = MODEL_SAVE_PATH / "best"
            best_path.mkdir(exist_ok=True)
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"[INFO] New best model saved (acc={best_acc:.3f}) at: {best_path}")

    print("\n[INFO] Training finished")
    save_plot(plot_state, PLOT_PATH)


if __name__ == "__main__":
    main()

