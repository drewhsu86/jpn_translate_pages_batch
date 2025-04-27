import os
import io
import datetime
import gradio as gr
import easyocr
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
import pytesseract

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
        blockSize=15, 
        C=5
    )

    img_processed = Image.fromarray(img_thresh)

    results = reader.readtext(np.array(img_processed))

    for bbox, _, _ in results:
        # Get bounding box corners
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        min_x, min_y = int(min(xs)), int(min(ys))
        max_x, max_y = int(max(xs)), int(max(ys))

        # Crop from original image
        cropped = img_processed.crop((min_x, min_y, max_x, max_y))

        # Resize (upscale) the cropped region
        upscale_factor = 10
        cropped = cropped.resize(
            (cropped.width * upscale_factor, cropped.height * upscale_factor),
            resample=Image.LANCZOS
        )

        # Convert to grayscale and binarize (thresholding)
        gray = cropped.convert("L")
        bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
        bw_gray = bw.convert("L")  # Convert to grayscale (L mode)

        # Run OCR on the preprocessed crop
        # new_text = reader.readtext(np.array(bw_gray), detail=0, paragraph=True, decoder='greedy')
        custom_config = r'--oem 3 --psm 6'
        new_text = pytesseract.image_to_string(bw_gray, lang="jpn-vert", config=custom_config)
        # manga_ocr = MangaOcr()
        # new_text = manga_ocr(bw_gray)

        # Draw the upright bounding box on the original image
        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)

        # Label the box with recognized text
        if new_text:
            print(f'New Text: {new_text}')
            # phrase = ''.join(new_text) # easyocr returns list
            phrase = new_text
            translated = GoogleTranslator(source='ja', target='en').translate(phrase)
            if not translated:
                continue
            word_list = translated.split(' ')
            increment = 24
            shadow_incr = 2
            font = ImageFont.truetype("arial.ttf", increment)
            start = 0
            for word in word_list:
                draw.text((min_x - shadow_incr, min_y - increment + start - shadow_incr), word, fill="white", font=font)
                draw.text((min_x + shadow_incr, min_y - increment + start + shadow_incr), word, fill="white", font=font)
                draw.text((min_x, min_y - increment + start), word, fill="blue", font=font)
                start += increment

    return img

with gr.Blocks() as demo:
    with gr.Row():  # Create a row for horizontal layout
        input_files = gr.File(label="Upload Images", file_count="multiple", type="binary")
        intermediate_output = gr.Image(label="Current Image")
        output_gallery = gr.Gallery(label="Processed Images")

    # Connect the function to the inputs and outputs
    input_files.change(translate_pages, inputs=input_files, outputs=[intermediate_output, output_gallery])

demo.launch(inbrowser=True)