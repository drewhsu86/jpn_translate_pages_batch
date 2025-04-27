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
mocr = MangaOcr()

long_cluster_length = False
def change_cluster_length(is_checked):
    long_cluster_length = is_checked

def new_filename():
    utc_now = datetime.datetime.utcnow()
    return f'image_{utc_now.strftime("%Y%m%d_%H%M%S")}.png'

def calculate_progress(count, total):
    return f'{int(100*count/total)}%, {count} out of {total}'

def translate_pages(image_list):
    intermediate = []
    output = []
    total = len(image_list)
    count = 0    
    for image in image_list:
        bytesIO = io.BytesIO(image)
        img = Image.open(bytesIO)
        img = img.convert("RGB")
        yield intermediate, output, calculate_progress(count, total)
        result_img, result_intermediate = translate_manga(img)
        result_img.save(f'{folder_name}/{new_filename()}')
        output.append(result_img)
        intermediate.append(result_intermediate)
        count += 1
        yield intermediate, output, calculate_progress(count, total)
        print(f'----- {calculate_progress(count, total)} -----')

def translate_manga(image):
    img = image
    # img = Image.fromarray(image).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img_copy = img.copy()
    draw = ImageDraw.Draw(img)
    draw_intermediate = ImageDraw.Draw(img_copy)

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

        draw_intermediate.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)

    bboxes = [result[0] for result in results]

    new_bboxes = merge_bounding_boxes(bboxes, draw_intermediate)

    # in the new bounding boxes, take a screenshot, use ocr and replace the text 
    translations = []
    for bbox in new_bboxes:
        cropped = img_processed.crop(bbox)
        text = mocr(cropped)
        print(text)
        translated = GoogleTranslator(source='ja', target='en').translate(text)
        print(translated)
        translations.append(translated)

    b = 2 # buffer
    for (x1, y1, x2, y2) in bboxes:
        draw.rectangle([x1-b, y1-b, x2+b, y2+b], fill="white")

    for i in range(len(new_bboxes)):
        bbox = new_bboxes[i]
        translated = translations[i]
        draw_multiline_inside_box(draw, bbox, translated)

    return img, img_copy

def merge_bounding_boxes(bboxes, draw = None):
    # use KDtree to store center points
    bboxes = [convert_to_minmax_box(bbox) for bbox in bboxes]

    centers = [(xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2) for (xmin, ymin, xmax, ymax) in bboxes]
    
    if long_cluster_length:
        diagonals = [np.linalg.norm(np.array([xmax, ymax]) - np.array([xmin,ymin])) for (xmin, ymin, xmax, ymax) in bboxes]
    else:
        shortest_side = [np.min([xmax - xmin,ymax - ymin]) for (xmin, ymin, xmax, ymax) in bboxes] 

    bbox_tree = KDTree(centers)
    # we will iterate through each bounding box and group them together
    # we will store the group numbers of each bounding box index first
    # then form the new bounding boxes from those groups later
    bbox_groups = list(range(len(bboxes))) # the value will be the group, index will be identifying bbox

    for i in range(len(bboxes)):
        center = centers[i]      
        
        if long_cluster_length:
            radius = diagonals[i]
        else:
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

    def merge_two_bbox(bbox1, bbox2):
        (xmin1, ymin1, xmax1, ymax1) = bbox1
        (xmin2, ymin2, xmax2, ymax2) = bbox2
        return (np.min([xmin1,xmin2]), np.min([ymin1,ymin2]), np.max([xmax1,xmax2]), np.max([ymax1,ymax2]))

    merged_bboxes_dict = {}

    for i in range(len(bbox_groups)):
        group = bbox_groups[i]
        if group in merged_bboxes_dict:
            merged_bboxes_dict[group].append(i)
        else:
            merged_bboxes_dict[group] = [i]

    new_bboxes = []
    for grouped_inds in merged_bboxes_dict.values():
        new_bbox = bboxes[grouped_inds[0]]
        for i in range(1, len(grouped_inds)):
            new_bbox = merge_two_bbox(new_bbox, bboxes[grouped_inds[i]])
        new_bboxes.append(new_bbox)

    if draw != None:
        for (x1,y1,x2,y2) in new_bboxes:
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        
    return new_bboxes

def convert_to_minmax_box(bbox):
    # bbox is expected to be in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox = np.array(bbox)
    
    # Calculate the xmin, ymin, xmax, ymax
    xmin = np.min(bbox[:, 0])
    ymin = np.min(bbox[:, 1])
    xmax = np.max(bbox[:, 0])
    ymax = np.max(bbox[:, 1])
    
    return (xmin, ymin, xmax, ymax)

def draw_multiline_inside_box(draw, bbox, text):
    if not draw or not bbox or not text:
        return
    shadow_offset = 2
    font_size = 20
    font = ImageFont.truetype("arial.ttf", font_size)
    # we use capital A as a reference for character size
    # we want to preserve the width and let the height run
    # if the estimated height is too large we'll scale the font down
    char_width = draw.textlength('A', font=font)
    approx_text_volume = char_width * font_size* len(text)
    (x1, y1, x2, y2) = bbox
    approx_box_volume = (x2-x1) * (y2-y1)
    if approx_text_volume > approx_box_volume:
        ratio = approx_box_volume / approx_text_volume 
        font_size = int(ratio * 20)
        font = ImageFont.truetype("arial.ttf", font_size)

    # check how many characters should go on one line
    lined_text = ''
    words = text.split(' ')
    # to split the text, we add up the words until they break the width limit
    # then we add a new line \n after the next word and space
    current_string = ''
    for word in words:
        current_width = draw.textlength(current_string, font=font)
        proposed_width = draw.textlength(current_string + f'{word} ', font=font)
        if proposed_width >= x2-x1:
            lined_text += current_string + '\n'
            current_string = ''
        current_string += f'{word} '
    lined_text += current_string
            
    draw.multiline_text((x1 + shadow_offset, y1 + shadow_offset), lined_text, font=font, fill="white")
    draw.multiline_text((x1,y1), lined_text, font=font, fill="black")

with gr.Blocks() as demo:
    with gr.Row():
        submit_button = gr.Button("Submit")
        progress_count = gr.Textbox('0%', label="Progress")
        checkbox = gr.Checkbox(label="Use longer clustering length", value=long_cluster_length)
    with gr.Row():  # Create a row for horizontal layout
        input_files = gr.File(label="Upload Images", file_count="multiple", type="binary")
        intermediate_output = gr.Gallery(label="Processing Images")
        output_gallery = gr.Gallery(label="Processed Images")

    # Connect the function to the inputs and outputs
    checkbox.change(change_cluster_length, inputs=checkbox, outputs=[])
    submit_button.click(translate_pages, inputs=input_files, outputs=[intermediate_output, output_gallery, progress_count])
demo.launch(inbrowser=True)