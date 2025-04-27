import os
import io
import datetime
import gradio as gr
import easyocr
# fix fugashi for manga_ocr: https://github.com/pypa/pip/issues/10605 (mecab dll)
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from scipy.spatial import KDTree

# works with input gr.Image (one at a time)
# doesn't work with file/files (multiple)

folder_name = "output_images"
os.makedirs(folder_name, exist_ok=True)
reader = easyocr.Reader(['ja'], gpu=False)

def new_filename():
    utc_now = datetime.datetime.utcnow()
    return f'image_{utc_now.strftime("%Y%m%d_%H%M%S")}.png'

def translate_pages(image_list):
    output = []
    total = len(image_list)
    count = 0
    for image in image_list:
        bytesIO = io.BytesIO(image)
        img = Image.open(bytesIO)
        img = img.convert("RGB")
        yield img, output
        result_img = translate_manga(img)
        result_img.save(f'{folder_name}/{new_filename()}')
        output.append(result_img)
        yield result_img, output
        count += 1
        print(f'----- {count} out of {total} -----')

def translate_manga(image):
    img = image
    # img = Image.fromarray(image).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    draw = ImageDraw.Draw(img)

    # process image and use it to pull text
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV,  # Invert for black text on white
        blockSize=21, 
        C=7
    )

    img_processed = Image.fromarray(img_thresh)

    results = reader.readtext(np.array(img_processed))

    for bbox, _, _ in results:
        # Get bounding box corners
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        min_x, min_y = int(min(xs)), int(min(ys))
        max_x, max_y = int(max(xs)), int(max(ys))

        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)

    bboxes = [result[0] for result in results]

    new_bboxes = merge_bounding_boxes(bboxes, draw)

    return img

def merge_bounding_boxes(bboxes, draw = None):
    # use KDtree to store center points
    bboxes = [convert_to_minmax_box(bbox) for bbox in bboxes]

    centers = [(xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2) for (xmin, ymin, xmax, ymax) in bboxes]
    # diagonals = [np.linalg.norm(np.array([xmax, ymax]) - np.array([xmin,ymin])) for (xmin, ymin, xmax, ymax) in bboxes]
    shortest_side = [np.min([xmax - xmin,ymax - ymin]) for (xmin, ymin, xmax, ymax) in bboxes] 

    bbox_tree = KDTree(centers)
    # we will iterate through each bounding box and group them together
    # we will store the group numbers of each bounding box index first
    # then form the new bounding boxes from those groups later
    bbox_groups = list(range(len(bboxes))) # the value will be the group, index will be identifying bbox

    for i in range(len(bboxes)):
        center = centers[i]
        # radius = diagonals[i]
        radius = 2*shortest_side[i]
        bbox_near_indices = bbox_tree.query_ball_point(center, radius)
        for b in bbox_near_indices:
            if b == i:
                continue
            bbox_groups[b] = bbox_groups[i]
        (xmin, ymin, xmax, ymax) = bboxes[i]
    
    if draw != None:
        font = ImageFont.truetype("arial.ttf", 20)
        for i in range(len(bbox_groups)):
            print(f'Bbox {i} in group {bbox_groups[i]}')
            draw.text(np.array(centers[i]) + np.array((2,2)), f'{bbox_groups[i]}', fill="white", font=font)
            draw.text(centers[i], f'{bbox_groups[i]}', fill="blue", font=font)
    
    return bboxes

def convert_to_minmax_box(bbox):
    # bbox is expected to be in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox = np.array(bbox)
    
    # Calculate the xmin, ymin, xmax, ymax
    xmin = np.min(bbox[:, 0])
    ymin = np.min(bbox[:, 1])
    xmax = np.max(bbox[:, 0])
    ymax = np.max(bbox[:, 1])
    
    return (xmin, ymin, xmax, ymax)

with gr.Blocks() as demo:
    with gr.Row():  # Create a row for horizontal layout
        input_files = gr.File(label="Upload Images", file_count="multiple", type="binary")
        intermediate_output = gr.Image(label="Current Image")
        output_gallery = gr.Gallery(label="Processed Images")

    # Connect the function to the inputs and outputs
    input_files.change(translate_pages, inputs=input_files, outputs=[intermediate_output, output_gallery])

demo.launch(inbrowser=True)