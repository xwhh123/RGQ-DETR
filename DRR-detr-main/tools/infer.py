import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
# EPDDD 17类植物病害名称
COCO_CLASSES = [
    'Alstonia scholaris_Rust disease',
    'Apple_Alternaria blotch',
    'Apple_Alternaria Leaf Spot',
    'Apple_Black Rot',
    'Arjun tree_Rust Disease',
    'Cashew_Leaf Miner',
    'Cassava_Bacterial Blight',
    'Cassava_Brown Spot',
    'Chinar_Leaf Spot',
    'Citrus_Black Spot',
    'Coffee_Leaf Miner',
    'Coffee_Leaf Rust',
    'Coffee_Sooty Blotch',
    'Grape_Black Rot',
    'Guava_Leaf blight',
    'Jamun_Leaf Spot',
    'Lemon_Canker',
]

def get_font(base_size=20):
    """尝试获取可缩放字体，失败时返回默认字体"""
    import platform
    system = platform.system()
    font_paths = []
    if system == 'Windows':
        font_paths = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/simhei.ttf',
                      'C:/Windows/Fonts/msyh.ttc']
    elif system == 'Darwin':
        font_paths = ['/System/Library/Fonts/Helvetica.ttc', '/Library/Fonts/Arial.ttf']
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, base_size)
    return ImageFont.load_default()

def draw(images, labels, boxes, scores, thrh=0.6, path="", class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES
    for i, im in enumerate(images):
        w, h = im.size
        # 基于短边计算缩放因子，基准尺寸640
        scale = min(w, h) / 640.0
        line_width = max(1, int(2 * scale))
        font_size = max(10, int(20 * scale))
        font = get_font(font_size)

        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red', width=line_width)
            cls_id = int(lab[j].item())
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            text = f"{cls_name} {round(scrs[j].item(), 2)}"
            draw.text((b[0], max(0, b[1] - font_size - 4)), text=text, font=font, fill='yellow')
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            # 兼容两种格式: {'module': ...} 或直接的 state_dict
            ema_state = checkpoint['ema']
            state = ema_state['module'] if 'module' in ema_state else ema_state
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    print(f"模型加载完成, 设备: {args.device}")

    im_pil = Image.open(args.im_file).convert('RGB')
    print(f"图片加载完成: {args.im_file}, 尺寸: {im_pil.size}")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    print("开始推理...")
    if args.sliced:
        num_boxes = args.numberofboxes

        aspect_ratio = w / h
        num_cols = int(np.sqrt(num_boxes * aspect_ratio))
        num_rows = int(num_boxes / num_cols)
        slice_height = h // num_rows
        slice_width = w // num_cols
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with autocast():  # Use AMP for each slice
                output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
            torch.cuda.empty_cache()
            labels, boxes, scores = output

            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predictions.append((labels, boxes, scores))

        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        output = model(im_data, orig_size)
        labels, boxes, scores = output

    # 根据输入文件名生成输出路径
    im_basename = os.path.basename(args.im_file)
    im_name = os.path.splitext(im_basename)[0]
    output_path = f"results_{im_name}.jpg"

    print(f"推理完成, 检测到 {len(labels[0]) if len(labels) > 0 else 0} 个目标")
    draw([im_pil], labels, boxes, scores, 0.6, path=output_path)
    print(f"结果已保存到 {output_path}")
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="configs/rtdetr/rtdetr_r101vd_6x_coco.yml")
    parser.add_argument('-r', '--resume', type=str, required=True, help='path to a trained checkpoint')
    parser.add_argument('-f', '--im-file', type=str, required=True, help='path to an input image')
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)










