import sys
import os
sam_path = '/scratch/tshu2/yyin34/segment-anything-2-real-time'
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
SAM2_CHECKPOINT_IMAGE = "/scratch/tshu2/yyin34/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG_IMAGE = "sam2_hiera_l.yaml"

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
    text_input=None,
    output_dir='./output'
):
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<CAPTION_TO_PHRASE_GROUNDING>': 
        {
            'bboxes': 
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594], 
                    [1.5999999046325684, 4.079999923706055, 639.0399780273438, 305.03997802734375]
                ], 
            'labels': ['A green car', 'a yellow building']
        }
    }
    """
    assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    print("class_names", class_names)
    class_ids = np.array(list(range(len(class_names))))
    import matplotlib.pyplot as plt

    # predict mask with SAM 2
    sam2_image_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    print("masks", masks.shape)
    np.save("masks.npy", masks)

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    sampled_points = []
    for i in range(masks.shape[0]):
        # Find the indices of the pixels where the mask has a value of 1
        indices = np.argwhere(masks[i] == 1)  # Get indices of points with value 1
        
        if indices.size > 0:  # Check if there are any points with value 1
            # Sample one point randomly from the indices
            sampled_point = indices[np.random.choice(indices.shape[0])]
            sampled_points.append((i, sampled_point))  # Store (mask_index, (y, x))

    # Print the sampled points
    for mask_index, point in sampled_points:
        print(f"Sampled point for mask {mask_index}: (y={point[0]}, x={point[1]})")

    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    image_np = np.array(image)
    # visualization results
    for mask_index, point in sampled_points:
        y, x = point
        cv2.circle(image_np, (x, y), 1, (0, 255, 0), -1)  # Draw a filled circle

    # Save the annotated image_np
    cv2.imwrite('image_TEST.png', image_np)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "TEST.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "TEST_with_mask.jpg"), annotated_frame)

    return sampled_points


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


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