import http
import os
import traceback
from io import BytesIO

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np

import json
import base64

from abeja.datasets import Client as DatasetsClient
import train
import parameters


def bitget(number, pos):
    return (number >> pos) & 1

def create_colormap(max_num):
    colormap = {}
    for i in range(0,max_num):
        id = i
        r=0
        g=0
        b=0
        for j in range(0,8):
            r = r | (bitget(id,0) << (7-j))
            g = g | (bitget(id,1) << (7-j))
            b = b | (bitget(id,2) << (7-j))
            id = id >> 3
        colormap[i] = [r,g,b]
    return colormap

def load_model(training_dir, device):
    seg_model = ''
    num_classes = 0
    with open(os.path.join(training_dir,'parameters.json'), 'r') as f:
        jf = json.load(f)
        seg_model = jf['SEG_MODEL']
        num_classes = int(jf['NUM_CLASSES'])
    model = train.create_model(num_classes=num_classes, model_name=seg_model)
    model.load_state_dict(torch.load(os.path.join(training_dir,'model.pth')))
    model.to(device)
    return model.eval(), num_classes

def get_dataset_properties(dataset_ids):
    datasets_client = DatasetsClient()
    props = {}
    for dataset_id in dataset_ids:
        dataset = datasets_client.get_dataset(dataset_id)
        props = dataset.props
        break
    return props

# Define the helper function
def decode_segmap(out, label_colors):
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    r = np.zeros_like(om).astype(np.uint8)
    g = np.zeros_like(om).astype(np.uint8)
    b = np.zeros_like(om).astype(np.uint8)
  
    for l,c in label_colors.items():
        idx = om == l
        r[idx] = c[0]
        g[idx] = c[1]
        b[idx] = c[2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb
    
def segmentation(model, img, device):
    trf = T.Compose([T.Resize(520), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(device)
    output = model(img)
    segmap = decode_segmap(output, color_map)
    return Image.fromarray(segmap)


training_dir = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
dataset_ids = os.environ.get('TRAINING_JOB_DATASET_IDS', '').split(',')
device = torch.device(parameters.DEVICE)

model, num_classes = load_model(training_dir, device)
color_map = create_colormap(num_classes)
props = get_dataset_properties(dataset_ids)


def handler(request, context):
    print('Start predict handler.')
    if 'http_method' not in request:
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        return {
            'status_code': http.HTTPStatus.BAD_REQUEST,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': message}
        }

    try:
        data = request.read()
        img = BytesIO(data)
        rgbimg = Image.open(img).convert('RGB')

        segmap = segmentation(rgbimg)
        segmap.save(os.path.join(training_dir, 'tmp.png')))
        b64 = base64.encodestring(open('tmp.png', 'rb').read())

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'properties': props, 'result': b64}
        }

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return None


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Abeja Segmentation Template for Prediction')

    parser.add_argument('--src', default='', help='source image file path')
    parser.add_argument('--dst', default='', help='target image file path to save')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img = Image.open(args.src)
    img.save(args.dst)
    segmentation(img)
