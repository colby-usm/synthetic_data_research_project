"""
COCO Annotation Editor
======================
Verify and edit bounding box annotations in a COCO-style annotations.json file.

Usage:
    python coco_annotation_editor.py [OPTIONS]

Options:
    --start-image-id INT       Image ID to start editing from (default: first image)
    --resume-annotations PATH  Path to a previously saved output annotations file to resume from
    --output PATH              Where to write the output file (default: output_annotations.json)

Controls (in the editor window):
    SPACE   Keep the current annotation as-is, move to the next
    Y       Enter draw mode — click and drag to draw a new bbox
    C       Confirm the newly drawn bbox (only valid after drawing in Y or Z mode)
    R       Discard the newly drawn bbox and try again / cancel add
    D       Delete the current annotation entirely
    Z       Add a brand-new annotation — drag bbox then press class number
    Q       Quit the editor and save progress
"""

import json
import os
import copy
import argparse
import cv2
import numpy as np

# ──────────────────────────────────────────────
# Configuration — fill these in before running
# ──────────────────────────────────────────────
ANNOTATIONS_IN = "./remapped_bulk_output_annotations.json"
IMAGE_DIR      = "./new_sim_data/"
IOU_THRESHOLD  = 0.85


# ──────────────────────────────────────────────
# Colours (BGR for OpenCV)
# ──────────────────────────────────────────────
COLOR_ACTIVE     = (0, 255, 0)
COLOR_OTHER      = (180, 180, 180)
COLOR_DRAW       = (0, 165, 255)
COLOR_HUD_MODE   = (255, 255, 0)
COLOR_HUD_CTRL   = (200, 200, 200)
COLOR_HUD_STATUS = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path, msg=""):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    suffix = f"  ({msg})" if msg else ""
    print(f"[INFO] Saved → {path}{suffix}")


def print_summary(coco):
    images      = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories  = coco.get("categories", [])
    all_image_ids      = {img["id"] for img in images}
    image_ids_with_ann = {a["image_id"] for a in annotations}
    ann_ids       = [a["id"] for a in annotations]
    n_dup_ann_ids = len(ann_ids) - len(set(ann_ids))
    print("\n" + "=" * 55)
    print("  COCO Dataset Summary")
    print("=" * 55)
    print(f"  Total images              : {len(images)}")
    print(f"  Total annotations         : {len(annotations)}")
    print(f"  Total categories          : {len(categories)}")
    print(f"  Annotations w/o image     : {sum(1 for a in annotations if a['image_id'] not in all_image_ids)}")
    print(f"  Images w/o annotation     : {sum(1 for img in images if img['id'] not in image_ids_with_ann)}")
    if n_dup_ann_ids:
        print(f"  !! Duplicate ann IDs      : {n_dup_ann_ids}")
    print("=" * 55 + "\n")


def clamp_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return [x, y, w, h]


def build_updated_annotation(source_ann, new_bbox, new_id, img_w, img_h):
    clamped = clamp_bbox(new_bbox, img_w, img_h)
    x, y, w, h = clamped
    new_ann = copy.deepcopy(source_ann)
    new_ann["id"]           = new_id
    new_ann["bbox"]         = [float(v) for v in clamped]
    new_ann["area"]         = float(w * h)
    new_ann["segmentation"] = [[
        float(x),     float(y),
        float(x + w), float(y),
        float(x + w), float(y + h),
        float(x),     float(y + h),
    ]]
    new_ann.setdefault("iscrowd", 0)
    return new_ann


def compute_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def bboxes_match(b1, b2):
    return compute_iou(b1, b2) >= IOU_THRESHOLD


def find_similar_annotations(reference_bbox, reference_ann_id, reference_category_id,
                              out_annotations, already_reviewed):
    matches = []
    for ann in out_annotations:
        if ann["id"] == reference_ann_id:
            continue
        if ann["image_id"] in already_reviewed:
            continue
        if ann["category_id"] != reference_category_id:
            continue
        if bboxes_match(ann["bbox"], reference_bbox):
            matches.append(ann)
    return matches


def render_bulk_confirm_frame(win_w, win_h, reference_bbox, n_matches, action_label, status_text=""):
    frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    cx = win_w // 2
    lines = [
        "Similar annotations found!",
        "",
        f"Your action  :  {action_label}",
        f"Bbox         :  {[round(v, 1) for v in reference_bbox]}",
        f"Matches      :  {n_matches} annotation(s) in unreviewed images",
        f"IoU threshold:  {IOU_THRESHOLD}",
        "",
        "A  —  Apply this action to ALL matching annotations",
        "S  —  Skip  (leave matching annotations unchanged)",
        "Q  —  Quit and save",
    ]
    y = win_h // 4
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (200, 200, 200)
        scale = 0.75 if i == 0 else 0.6
        thick = 2 if i == 0 else 1
        (tw, _), _ = cv2.getTextSize(line, FONT, scale, thick)
        cv2.putText(frame, line, (cx - tw // 2, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += 38
    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, FONT, 0.6, 2)
        cv2.putText(frame, status_text, (cx - tw // 2, win_h - 30),
                    FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# Drawing state
# ──────────────────────────────────────────────

class DrawState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing = False
        self.start   = None
        self.end     = None

    def to_bbox(self):
        if self.start is None or self.end is None:
            return None
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        return [x1, y1, x2 - x1, y2 - y1]

    def is_valid(self, min_px=3):
        b = self.to_bbox()
        return b is not None and b[2] >= min_px and b[3] >= min_px


def make_mouse_callback(ds, mode_ref):
    def callback(event, x, y, flags, _):
        if not mode_ref["draw_mode"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            ds.drawing = True
            ds.start = (x, y)
            ds.end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ds.drawing:
            ds.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            ds.drawing = False
            ds.end = (x, y)
    return callback


# ──────────────────────────────────────────────
# Frame rendering
# ──────────────────────────────────────────────

def _draw_labelled_rect(frame, bbox, color, thickness, label):
    bx, by, bw, bh = (int(v) for v in bbox)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    ty = max(by - 5, th + 4)
    cv2.rectangle(frame, (bx, ty - th - 3), (bx + tw + 4, ty + 2), (30, 30, 30), -1)
    cv2.putText(frame, label, (bx + 2, ty), FONT, 0.5, color, 1, cv2.LINE_AA)


def render_frame(image, active_ann, other_anns, draw_mode, ds, label_map,
                 status_text="", progress_text="",
                 add_mode=False, class_select=False,
                 categories=None, pending_bbox=None):
    frame = image.copy()
    h, w  = frame.shape[:2]

    for ann in other_anns:
        cat = label_map.get(ann["category_id"], f"cat:{ann['category_id']}")
        _draw_labelled_rect(frame, ann["bbox"], COLOR_OTHER, 1, cat)

    if active_ann:
        cat = label_map.get(active_ann["category_id"], f"cat:{active_ann['category_id']}")
        _draw_labelled_rect(frame, active_ann["bbox"], COLOR_ACTIVE, 2, f"► {cat}")

    # Live bbox preview — both Y (redraw existing) and Z (add new)
    if (draw_mode or add_mode) and ds.start and ds.end:
        bbox = ds.to_bbox()
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_DRAW, 2)
            cv2.putText(frame, f"{bw}×{bh}", (x, max(y - 5, 10)),
                        FONT, 0.45, COLOR_DRAW, 1, cv2.LINE_AA)

    # Class-select overlay: shown after C confirms bbox in Z mode
    if class_select and pending_bbox is not None and categories:
        px, py, pw, ph = (int(v) for v in pending_bbox)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), COLOR_DRAW, 2)
        n_cats    = len(categories)
        panel_h   = 36 + (n_cats + 2) * 26
        overlay   = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (400, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.putText(frame, "Select class — press number key:",
                    (10, h - panel_h + 24), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        for i, cat in enumerate(categories, start=1):
            cv2.putText(frame, f"  {i}.  {cat['name']}",
                        (10, h - panel_h + 24 + i * 26),
                        FONT, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "  0 / R — cancel",
                    (10, h - panel_h + 24 + (n_cats + 1) * 26),
                    FONT, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

    # Bottom HUD
    if add_mode:
        mode_str = "ADD MODE — click & drag new bbox"
    elif draw_mode:
        mode_str = "DRAW MODE — click & drag"
    else:
        mode_str = "REVIEW MODE"
    cv2.putText(frame, mode_str, (10, h - 50), FONT, 0.55, COLOR_HUD_MODE, 2, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=keep  Y=redraw  Z=add new  C=confirm  R=retry  D=delete  Q=quit",
                (10, h - 22), FONT, 0.45, COLOR_HUD_CTRL, 1, cv2.LINE_AA)

    if progress_text:
        (pw2, _), _ = cv2.getTextSize(progress_text, FONT, 0.5, 1)
        cv2.putText(frame, progress_text, (w - pw2 - 10, 25),
                    FONT, 0.5, COLOR_HUD_CTRL, 1, cv2.LINE_AA)

    if status_text:
        cv2.putText(frame, status_text, (10, 30), FONT, 0.65, COLOR_HUD_STATUS, 2, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Main editor
# ──────────────────────────────────────────────

def run_editor(coco_in, start_image_id, resume_path, output_path):

    images      = coco_in.get("images", [])
    annotations = coco_in.get("annotations", [])
    categories  = coco_in.get("categories", [])
    label_map   = {cat["id"]: cat["name"] for cat in categories}
    image_meta  = {img["id"]: img for img in images}
    images_sorted = sorted(images, key=lambda img: img["id"])
    total_images  = len(images_sorted)

    if resume_path and os.path.isfile(resume_path):
        print(f"[INFO] Resuming from {resume_path}")
        out_coco = load_json(resume_path)
        out_coco["images"]     = copy.deepcopy(images)
        out_coco["categories"] = copy.deepcopy(categories)
    else:
        out_coco = {
            "info":        coco_in.get("info", {}),
            "licenses":    coco_in.get("licenses", []),
            "images":      copy.deepcopy(images),
            "annotations": copy.deepcopy(annotations),
            "categories":  copy.deepcopy(categories),
        }
        print(f"[INFO] Copying {len(annotations)} annotations to output file.")
        save_json(out_coco, output_path, f"initial copy — {len(annotations)} annotations")

    out_annotations = out_coco["annotations"]

    def build_id_index():
        return {a["id"]: i for i, a in enumerate(out_annotations)}

    reviewed_key     = "_reviewed_image_ids"
    already_reviewed = set(out_coco.get(reviewed_key, []))

    def mark_reviewed(img_id):
        already_reviewed.add(img_id)
        out_coco[reviewed_key] = sorted(already_reviewed)

    all_ids  = [a["id"] for a in out_annotations]
    _next_id = [max(all_ids, default=0) + 1]

    def consume_id():
        nid = _next_id[0]
        _next_id[0] += 1
        return nid

    start_idx = 0
    if start_image_id is not None:
        for i, img in enumerate(images_sorted):
            if img["id"] == start_image_id:
                start_idx = i
                break
        else:
            print(f"[WARN] image_id {start_image_id} not found — starting from the beginning.")

    WIN = "COCO Annotation Editor"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 900)
    ds       = DrawState()
    mode_ref = {"draw_mode": False}
    cv2.setMouseCallback(WIN, make_mouse_callback(ds, mode_ref))

    # ── Bulk-propagation helper ───────────────────────────────────────────────
    def _handle_bulk(action, reference_ann, original_bbox, new_bbox,
                     out_annotations, already_reviewed, image_meta,
                     consume_id, build_id_index, mark_reviewed,
                     out_coco, output_path, WIN):
        ref_cat_id = reference_ann["category_id"]
        ref_ann_id = reference_ann["id"]
        matches = find_similar_annotations(
            original_bbox, ref_ann_id, ref_cat_id, out_annotations, already_reviewed)
        if not matches:
            return False
        n_matches    = len(matches)
        action_label = ("KEEP  (original bbox kept)" if action == "KEEP"
                        else f"UPDATE bbox → {[round(v, 1) for v in new_bbox]}")
        print(f"  [BULK] Found {n_matches} similar annotation(s) — showing prompt.")
        _, _, win_w, win_h = cv2.getWindowImageRect(WIN)
        win_w = max(win_w, 640)
        win_h = max(win_h, 400)
        status = ""
        while True:
            frame = render_bulk_confirm_frame(win_w, win_h, original_bbox, n_matches, action_label, status)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                return True
            elif key == ord('s'):
                print(f"  [BULK] Skipped.")
                return False
            elif key == ord('a'):
                applied = 0
                affected = set()
                if action == "KEEP":
                    for m in matches:
                        affected.add(m["image_id"])
                        applied += 1
                    print(f"  [BULK] KEEP applied to {applied} annotation(s).")
                else:
                    id_index = build_id_index()
                    for m in matches:
                        iid  = m["image_id"]
                        meta = image_meta.get(iid, {})
                        miw  = meta.get("width", 0) or 9999
                        mih  = meta.get("height", 0) or 9999
                        upd  = build_updated_annotation(m, new_bbox, consume_id(), miw, mih)
                        idx  = id_index.get(m["id"])
                        if idx is not None:
                            out_annotations[idx] = upd
                            id_index = build_id_index()
                        affected.add(iid)
                        applied += 1
                    print(f"  [BULK] UPDATE applied to {applied} annotation(s).")
                save_json(out_coco, output_path, f"{applied} annotation(s) bulk-updated")
                return False
            else:
                status = "Press  A = apply all   S = skip   Q = quit"

    # ── Per-image loop ────────────────────────────────────────────────────────
    quit_requested = False

    for img_idx, img_info in enumerate(images_sorted[start_idx:], start=start_idx):
        if quit_requested:
            break

        img_id    = img_info["id"]
        file_name = img_info["file_name"]
        print(f"\n[IMAGE {img_idx + 1}/{total_images}] id={img_id}  file={file_name}")

        if img_id in already_reviewed:
            print(f"  [SKIP] Already reviewed.")
            continue

        img_path = os.path.join(IMAGE_DIR, file_name)
        if not os.path.isfile(img_path):
            print(f"  [WARN] File not found — skipping: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Could not read image — skipping: {img_path}")
            continue

        actual_h, actual_w = image.shape[:2]
        meta = image_meta.get(img_id, {})
        iw = meta.get("width") or actual_w
        ih = meta.get("height") or actual_h

        anns_for_image = [a for a in out_annotations if a["image_id"] == img_id]
        total_anns     = len(anns_for_image)

        if total_anns == 0:
            print(f"  [INFO] No annotations on this image — marking reviewed.")
            mark_reviewed(img_id)
            save_json(out_coco, output_path, "no annotations on image")
            continue

        ann_idx = 0

        while ann_idx < len(anns_for_image):
            if quit_requested:
                break

            ann        = anns_for_image[ann_idx]
            other_anns = [a for a in out_annotations
                          if a["image_id"] == img_id and a["id"] != ann["id"]]

            draw_mode        = False
            add_mode         = False
            class_select     = False
            pending_add_bbox = None
            mode_ref["draw_mode"] = False
            ds.reset()
            status_text  = ""
            progress_txt = f"ann {ann_idx + 1}/{total_anns}  img {img_idx + 1}/{total_images}"

            while True:
                frame = render_frame(
                    image, ann, other_anns,
                    draw_mode, ds, label_map,
                    status_text, progress_txt,
                    add_mode=add_mode,
                    class_select=class_select,
                    categories=categories,
                    pending_bbox=pending_add_bbox,
                )
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(30) & 0xFF

                # ── Q — quit ─────────────────────────────────────────────────
                if key == ord('q'):
                    quit_requested = True
                    break

                # ── SPACE — keep ──────────────────────────────────────────────
                elif key == ord(' ') and not add_mode and not class_select:
                    print(f"  [KEEP]   ann id={ann['id']}")
                    quit_requested = _handle_bulk(
                        action="KEEP", reference_ann=ann, original_bbox=ann["bbox"],
                        new_bbox=None, out_annotations=out_annotations,
                        already_reviewed=already_reviewed, image_meta=image_meta,
                        consume_id=consume_id, build_id_index=build_id_index,
                        mark_reviewed=mark_reviewed, out_coco=out_coco,
                        output_path=output_path, WIN=WIN)
                    ann_idx += 1
                    break

                # ── D — delete ────────────────────────────────────────────────
                elif key == ord('d') and not add_mode and not class_select:
                    id_index = build_id_index()
                    if ann["id"] in id_index:
                        out_annotations.pop(id_index[ann["id"]])
                    anns_for_image.pop(ann_idx)
                    print(f"  [DELETE] ann id={ann['id']}")
                    break

                # ── Y — enter redraw mode (edit existing annotation bbox) ─────
                elif key == ord('y') and not add_mode and not class_select:
                    draw_mode = True
                    mode_ref["draw_mode"] = True
                    ds.reset()
                    status_text = "Click & drag to draw new bbox, then C to confirm"

                # ── C — confirm redraw (Y mode) ───────────────────────────────
                elif key == ord('c') and draw_mode and not add_mode:
                    if ds.is_valid():
                        new_ann = build_updated_annotation(ann, ds.to_bbox(), consume_id(), iw, ih)
                        id_index = build_id_index()
                        if ann["id"] in id_index:
                            out_annotations[id_index[ann["id"]]] = new_ann
                        else:
                            out_annotations.append(new_ann)
                        anns_for_image[ann_idx] = new_ann
                        print(f"  [UPDATE] old id={ann['id']} → new id={new_ann['id']}  bbox={new_ann['bbox']}")
                        draw_mode = False
                        mode_ref["draw_mode"] = False
                        quit_requested = _handle_bulk(
                            action="UPDATE", reference_ann=new_ann, original_bbox=ann["bbox"],
                            new_bbox=new_ann["bbox"], out_annotations=out_annotations,
                            already_reviewed=already_reviewed, image_meta=image_meta,
                            consume_id=consume_id, build_id_index=build_id_index,
                            mark_reviewed=mark_reviewed, out_coco=out_coco,
                            output_path=output_path, WIN=WIN)
                        ann_idx += 1
                        break
                    else:
                        status_text = "Bbox too small — draw again (R to clear)"

                # ── Z — enter add mode (draw a brand-new annotation) ──────────
                elif key == ord('z') and not draw_mode and not class_select:
                    add_mode         = True
                    pending_add_bbox = None
                    mode_ref["draw_mode"] = True
                    ds.reset()
                    status_text = "ADD: drag bbox, then C to confirm"

                # ── C — confirm bbox in add mode, switch to class-select ──────
                elif key == ord('c') and add_mode and not class_select:
                    if ds.is_valid():
                        bx, by, bw, bh = ds.to_bbox()
                        bx = max(0, min(int(bx), iw - 1))
                        by = max(0, min(int(by), ih - 1))
                        bw = max(1, min(int(bw), iw - bx))
                        bh = max(1, min(int(bh), ih - by))
                        pending_add_bbox      = [bx, by, bw, bh]
                        add_mode              = False
                        mode_ref["draw_mode"] = False
                        ds.reset()
                        class_select = True
                        status_text  = "Press class number (0 / R to cancel)"
                    else:
                        status_text = "Bbox too small — draw again (R to cancel)"

                # ── Number keys — assign class after Z draw ───────────────────
                elif class_select and pending_add_bbox is not None:
                    num = None
                    if ord('1') <= key <= ord('9'):
                        num = key - ord('0')
                    elif key == ord('0') or key == ord('r'):
                        class_select     = False
                        pending_add_bbox = None
                        status_text      = "Add cancelled — Z to try again"
                    if num is not None:
                        if 1 <= num <= len(categories):
                            chosen_cat = categories[num - 1]
                            ax, ay, aw, ah = pending_add_bbox
                            new_ann = {
                                "id":           consume_id(),
                                "image_id":     img_id,
                                "category_id":  chosen_cat["id"],
                                "bbox":         [float(ax), float(ay), float(aw), float(ah)],
                                "area":         float(aw * ah),
                                "segmentation": [[
                                    float(ax),      float(ay),
                                    float(ax + aw), float(ay),
                                    float(ax + aw), float(ay + ah),
                                    float(ax),      float(ay + ah),
                                ]],
                                "iscrowd": 0,
                            }
                            out_annotations.append(new_ann)
                            anns_for_image.append(new_ann)
                            total_anns = len(anns_for_image)
                            other_anns = [a for a in out_annotations
                                          if a["image_id"] == img_id and a["id"] != ann["id"]]
                            print(f"  [ADD]    id={new_ann['id']}  "
                                  f"class='{chosen_cat['name']}'  bbox={new_ann['bbox']}")
                            class_select     = False
                            pending_add_bbox = None
                            status_text      = f"Added '{chosen_cat['name']}' — Z to add another"
                        else:
                            status_text = f"Invalid — press 1–{len(categories)}, or 0/R to cancel"

                # ── R — retry / cancel ────────────────────────────────────────
                elif key == ord('r'):
                    if class_select:
                        class_select     = False
                        pending_add_bbox = None
                        status_text      = "Add cancelled — Z to try again"
                    elif add_mode:
                        ds.reset()
                        status_text = "Cleared — draw again"
                    elif draw_mode:
                        ds.reset()
                        status_text = "Cleared — draw again"

            # end inner while

        # end while ann_idx

        if not quit_requested:
            mark_reviewed(img_id)
            n_kept = len([a for a in out_annotations if a["image_id"] == img_id])
            save_json(out_coco, output_path, f"image id={img_id} done — {n_kept} annotation(s) kept")

    # end for img_info

    save_json(out_coco, output_path, f"session ended — {len(out_annotations)} annotations total")
    cv2.destroyAllWindows()
    print("\n" + "=" * 55)
    print("  Editing complete.")
    print(f"  Output : {output_path}")
    print(f"  Kept   : {len(out_annotations)} annotations")
    print("=" * 55)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COCO annotation verification and bbox editor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start-image-id", type=int, default=None, metavar="INT")
    parser.add_argument("--resume-annotations", type=str, default=None, metavar="PATH")
    parser.add_argument("--output", type=str, default=None, metavar="PATH")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    elif args.resume_annotations:
        output_path = args.resume_annotations
    else:
        output_path = "NEWESTremapped_bulk_output_annotations.json"

    if not ANNOTATIONS_IN:
        parser.error("Please set ANNOTATIONS_IN at the top of the script.")
    if not IMAGE_DIR:
        parser.error("Please set IMAGE_DIR at the top of the script.")
    if not os.path.isfile(ANNOTATIONS_IN):
        parser.error(f"Annotations file not found: {ANNOTATIONS_IN}")
    if not os.path.isdir(IMAGE_DIR):
        parser.error(f"Image directory not found: {IMAGE_DIR}")

    coco = load_json(ANNOTATIONS_IN)
    print_summary(coco)
    run_editor(coco, args.start_image_id, args.resume_annotations, output_path)


if __name__ == "__main__":
    main()
