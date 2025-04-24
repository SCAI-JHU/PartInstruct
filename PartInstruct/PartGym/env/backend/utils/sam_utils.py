import sys
import os
sam_path = '/home/jimmyhan/Desktop/lgplm/PartInstruct/baselines/third_party/segment-anything-2-real-time'
sys.path.append(sam_path)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import display
import numpy as np
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2_camera_predictor

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT_IMAGE = "/home/jimmyhan/Desktop/lgplm/PartInstruct/baselines/third_party/sam2_hiera_large.pt"
SAM2_CONFIG_IMAGE = "sam2_hiera_l.yaml"

OUTPUT_DIR = '/home/jimmyhan/Desktop/lgplm/PartInstruct/baselines/sam_test'

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

sam2_model = build_sam2(SAM2_CONFIG_IMAGE, SAM2_CHECKPOINT_IMAGE, device=device)
sam2_image_predictor = SAM2ImagePredictor(sam2_model)

print("HERE ####")
def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"
    device = model.device
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

def phrase_grounding_and_segmentation(
    image,
    florence2_model=florence2_model,
    florence2_processor=florence2_processor,
    sam2_image_predictor=sam2_image_predictor,
    task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    scene_text_input=None,
    part_text_input=None,
    output_dir=OUTPUT_DIR
):
    print("scene_text_input", scene_text_input)
    results_scene = run_florence2(task_prompt, scene_text_input, florence2_model, florence2_processor, image)
    assert scene_text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results_scene = results_scene[task_prompt]
    bbox = np.array(results_scene["bboxes"])
    class_names = results_scene["labels"]
    print("class_names scene", class_names)
    image_np = np.array(image)
    image_np_test = image_np
    x_min, y_min, x_max, y_max = bbox[0].astype(int)

    # Enlarge the crop region for better part-level grounding
    x_min -= 10
    y_min -= 10
    x_max += 10
    y_max += 10
    original_height, original_width = image_np.shape[:2]
    if x_min < 0 or y_min < 0 or x_max >= original_width or y_max >= original_height:
        print("Invalid crop: exceeds image boundaries")
        return False

    cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
    cropped_image_np = np.array(cropped_image)
    print("cropped_image", cropped_image_np.shape)
    # plt.imshow(cropped_image_np)
    # plt.show()
    # plt.savefig(f"{output_dir}/cropped_image.png")
    # plt.close()

    cropped_image = Image.fromarray(cropped_image)
    print("part_text_input", part_text_input)
    results_part = run_florence2(task_prompt, part_text_input, florence2_model, florence2_processor, cropped_image)
    assert part_text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results_part = results_part[task_prompt]
    input_boxes = np.array(results_part["bboxes"])
    class_names = results_part["labels"]
    class_ids = np.array(list(range(len(class_names))))
    print("class_names part", class_names)

    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # predict mask with SAM 2
    sam2_image_predictor.set_image(cropped_image)
    masks, scores, logits = sam2_image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=cropped_image_np.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imwrite(os.path.join(output_dir, "image_croppped_mask.jpg"), annotated_frame)
    
    sampled_points = []
    original_sampled_points = []
    for i in range(masks.shape[0]):
        indices = np.argwhere(masks[i] == 1)  # Get indices of points with value 1
        if indices.size > 0:  
            sampled_point = indices[np.random.choice(indices.shape[0])]
            sampled_points.append((i, sampled_point))  # Store (mask_index, (y, x))
            original_point = (sampled_point[0] + y_min, sampled_point[1] + x_min) 
            print("original_point", original_point)
            original_sampled_points.append((i, original_point))

    # for mask_index, point in original_sampled_points:
    #     print(f"Sampled point for mask {mask_index}: (y={point[0]}, x={point[1]})")

    # for mask_index, point in sampled_points:
    #     print(f"Sampled !! point for mask {mask_index}: (y={point[0]}, x={point[1]})")

    # for mask_index, point in original_sampled_points:
    #     y, x = point
    #     cv2.circle(image_np, (x, y), 1, (0, 255, 0), -1)  # Draw a filled circle
    # cv2.imwrite(os.path.join(output_dir, "image_full_point.jpg"), image_np)

    for mask_index, point in sampled_points:
        y, x = point
        cv2.circle(cropped_image_np, (x, y), 1, (0, 255, 0), -1)  # Draw a filled circle
    cv2.imwrite(os.path.join(output_dir, "image_cropped_point.jpg"), cropped_image_np)

    # padded_mask = np.zeros((original_height, original_width), dtype=bool)
    # mask_height, mask_width = masks.shape[1], masks.shape[2]
    # y_start = y_min
    # y_end = y_start + mask_height
    # x_start = x_min
    # x_end = x_start + mask_width

    # y_end = min(y_end, original_height)
    # x_end = min(x_end, original_width)

    # padded_mask[y_start:y_end, x_start:x_end] = masks[0]  # masks[0] to get the 2D mask
    # padded_mask = np.expand_dims(padded_mask, axis=0)
    # # import ipdb; ipdb.set_trace()
    # print("Padded mask shape:", padded_mask.shape)

    # print("input_boxes", input_boxes.shape)
    # print("masks", masks.shape)
    # print("class_ids", class_ids.shape)
    # detections = sv.Detections(
    #     xyxy=input_boxes[0],
    #     mask=padded_mask.astype(bool),
    #     class_id=class_ids[0]
    # )

    # # Proceed with annotation as before
    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=image_np.copy(), detections=detections)

    # label_annotator = sv.LabelAnnotator()
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # mask_annotator = sv.MaskAnnotator()
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    # cv2.imwrite(os.path.join(output_dir, "image_two_step_mask.jpg"), annotated_frame)
    
    return original_sampled_points


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_np = mask.reshape(h, w, 1)
    mask_np_result = np.where(mask_np[:, :, 0] > 0, 1, 0) 
    mask_image =  mask_np * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_np_result


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))

def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union

    total_pixels = mask1.size
    true_positive = np.logical_and(mask1, mask2).sum()
    true_negative = np.logical_and(np.logical_not(mask1), np.logical_not(mask2)).sum()
    accuracy = (true_positive + true_negative) / total_pixels
    
    return iou, accuracy