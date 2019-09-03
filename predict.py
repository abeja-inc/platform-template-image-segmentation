import http
import os
import traceback
from io import BytesIO

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np

from abeja.datasets import Client as DatasetsClient
from abeja.datalake import Client as DatalakeClient
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

datasets_client = DatasetsClient()
datalake_client = DatalakeClient()
dataset = datasets_client.get_dataset(os.environ.get('DATASET_ID', ''))
num_classes = len(dataset.props['categories'][0]['labels']) + 1
color_map = create_colormap(num_classes)

model = train.create_model(num_classes=num_classes, model_name=parameters.SEG_MODEL, finetuning=parameters.FINE_TUNING)
model.load_state_dict(torch.load('model.pth'))
device = torch.device(parameters.DEVICE)
model.to(device)
model = model.eval()

trf = T.Compose([T.Resize(520), 
                #T.CenterCrop(224), 
                T.ToTensor(), 
                T.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229, 0.224, 0.225])])

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
        img = Image.open(img).convert('RGB')
        inp = trf(img).unsqueeze(0).to(device)
        output = model(img)
        
        segmap = decode_segmap(output, color_map)
        return segmap

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return None

