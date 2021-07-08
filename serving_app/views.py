import base64
import json
import os
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageColor, ImageDraw, ImageFont
from django.shortcuts import render
from django.http import HttpResponseRedirect

from . import models
from .forms import ImageUploadForm


# Create your views here.
def index(request):
    upload_form = ImageUploadForm()
    content = {'upload_form': upload_form}
    return render(request, 'serving_app/index.html', content)


def upload(request):
    # 상용 서비스라면 별도 DB 서버로 넘겨야겠다.
    file = request.FILES['file']
    image_model = models.Image(file=file)
    image_model.save()
    
    # 이미지 추론 by `comm_method`
    comm_method = request.POST.get('comm-method')
    if comm_method == 'serving_api':
        print('*** Serving api')
        pred, bboxed_image = send_api(file)
    elif comm_method == 'internal_tf':
        print('*** Internal tf')
        pred, bboxed_image = predict(file)
    
    if not os.path.exists('bboxed.jpg'):
        print('image not found.')
    
    # 추론 결과를 이미지에 그려서 프론트로 보내기.
    base64str = convert_image_to_base64(bboxed_image)
    content = {'bboxed_image': base64str, 'comm_method': comm_method}
    return render(request, 'serving_app/predict.html', content)


def convert_image_to_base64(image):
    img_in_memory = BytesIO()
    tf.keras.preprocessing.image.array_to_img(image).save(img_in_memory, "JPEG")
    img_in_memory.seek(0)  # 쓰기하느라 이동한 파일 포인터 첫 위치로 돌리기.
    base64str = base64.b64encode(img_in_memory.getvalue())  # base64로 바이트 로딩.
    base64str = base64str.decode('utf8')
    return base64str


def predict(image_file):
    model_path = 'detectors/efficientdet_d1'
    if not os.path.exists(model_path):
        print(f'model not found.{os.getcwd()}')
    
    # dtype=tf.uint8, shape=[1, height, width, 3], range=[0, 255].
    detector = tf.saved_model.load(model_path)
    
    # Load image.
    image_array = convert_image_to_array(image_file)
    
    # Predict on the image array(+ draw result on image)
    pred = detector.signatures['serving_default'](image_array)
    essential_pred = {
        'clsses': pred['detection_classes'].numpy(),
        'boxes': pred['detection_boxes'].numpy(),
        'scores': pred['detection_scores'].numpy(),
    }
    bboxed_image = draw_boxes(
        image_array[0].numpy(), pred['detection_boxes'][0].numpy(),
        pred['detection_classes'][0].numpy(), pred['detection_scores'][0].numpy())
    return essential_pred, bboxed_image


def send_api(image_file):
    """request by API of TFServing
    
    추론 모델이 무엇인가에 따라 요청/응답에 딸려오는 데이터 형식이 달라질 수 있음에 주의.
    :param image_file: 백엔드 상 이미지 파일의 경로.
    :return: 추론 결과(labels, scores, bboxes)와, 추론 결과를 그린 이미지.
    """
    image_array = convert_image_to_array(image_file)
    
    request_url = 'http://49.50.165.247:8501/v1/models/detector:predict'
    # `image_tensor` is original type.
    data = json.dumps({"signature_name": "serving_default", "instances": image_array.numpy().tolist()})
    headers = {"content-type": "application/json"}
    
    session = requests.Session()
    req = requests.Request('POST', request_url, data=data, headers=headers)
    prepped = req.prepare()
    
    json_response = session.send(prepped)
    print(f"The response created at {json_response.headers['Date']}")
    print(f'time elapsed per response: {json_response.elapsed.total_seconds()}')
    
    # Draw result on image.
    pred = json.loads(json_response.text)['predictions'][0]
    bboxed_image = draw_boxes(
        image_array[0].numpy(), np.array(pred['detection_boxes']),
        np.array(pred['detection_classes']), np.array(pred['detection_scores']))
    return pred, bboxed_image


def convert_image_to_array(image_file):
    image = Image.open(image_file)
    image_array = np.array(image)
    # image_array = tf.image.convert_image_dtype(image_array, dtype=tf.uint8)  # <- 이 줄에서 변환 후 문제 생김.
    # image_array = Image.fromarray(np.uint8(image_array)).convert("RGB")
    image_array = tf.expand_dims(image_array, axis=0)
    return image_array


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i],
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image
