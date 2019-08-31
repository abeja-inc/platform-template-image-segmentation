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


datasets_client = DatasetsClient()
datalake_client = DatalakeClient()
dataset = datasets_client.get_dataset(os.environ.get('DATASET_ID', ''))
num_classes = len(dataset.props['categories'][0]['labels']) + 1

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
def decode_segmap(out, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    r = np.zeros_like(om).astype(np.uint8)
    g = np.zeros_like(om).astype(np.uint8)
    b = np.zeros_like(om).astype(np.uint8)
  
    for l in range(0, nc):
        idx = om == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
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
        
        segmap = decode_segmap(output, num_classes)
        return segmap

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return None

