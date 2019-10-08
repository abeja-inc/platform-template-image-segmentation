import http
import os
import traceback
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

import json
import base64
import tempfile

from abeja.datasets import Client as DatasetsClient
import train
import parameters
from utils import create_colormap


def bitget(number, pos):
    return (number >> pos) & 1


def load_model(training_dir, device):
    with open(os.path.join(training_dir,'parameters.json'), 'r') as f:
        jf = json.load(f)
        seg_model = jf['SEGMENTATION_MODEL']
        num_classes = int(jf['NUM_CLASSES'])
    model = train.create_model(num_classes=num_classes, model_name=seg_model)
    model.load_state_dict(torch.load(os.path.join(training_dir,'model.pth')))
    model.to(device)
    return model.eval(), num_classes


def get_dataset_labels(dataset_ids):
    datasets_client = DatasetsClient()
    labels = []
    for dataset_id in dataset_ids:
        dataset = datasets_client.get_dataset(dataset_id)
        labels = dataset.props['categories'][0]['labels']
        break
    return labels


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


training_dir = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
dataset_ids = os.environ.get('TRAINING_JOB_DATASET_IDS', '').split(',')
device_name = parameters.DEVICE if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

model, num_classes = load_model(training_dir, device)
dataset_labels = get_dataset_labels(dataset_ids)
color_map = create_colormap(dataset_labels)


def segmentation(img):
    trf = T.Compose([T.Resize(520), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(device)
    output = model(inp)['out']
    print(output)
    segmap = decode_segmap(output, color_map)
    return Image.fromarray(segmap)


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
        with tempfile.NamedTemporaryFile(suffix=".png") as tf:
            save_file = tf.name
            segmap.save(save_file)
            b64 = base64.b64encode(open(save_file, 'rb').read()).decode('ascii')

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'labels': dataset_labels, 'result': b64}
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
    img = Image.open(args.src).convert('RGB')
    seg_img = segmentation(img)
    seg_img.save(args.dst)
