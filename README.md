# [Template] Image Segmentation
This is the template of image-segmentation task for ABEJA Platform.

This template uses transfer-learning from Fully Convolutional Network or DeepLab version3 with MS COCO. Making it easy to try building ML model, this template uses many hard-coding parameters. You can change the parameters by setting environmental variables or editing code directly.



## Requirements
- Python 3.6.x
- [For local] Install [abeja-sdk](https://developers.abeja.io/sdk/)


## Docker
- abeja/all-cpu:19.04
- abeja/all-gpu:19.04


## Conditions
- Fully Convolutional Network or DeepLab v3 (both ResNet101 backbone)
- Allow only 1 training dataset and 1 validation dataset


## Parameters
| env | type | description |
| --- | --- | --- |
| BATCH_SIZE | int | Batch size. Default `32`. |
| EPOCHS | int | Epoch number. This template applies "Early stopping". Default `50`. |
| LEARNING_RATE | float | Learning rate. Need to be from `0.0` to `1.0`. Default `0.01`. |
| MOMENTUM | float | Weight of the previous update. Need to be from `0.0`. Default `0.9`. |
| WEIGHT_DECAY | float | SGD parameter "decay". Need to be from `0.0`. Default `1e-4`. |
| EARLY_STOPPING_TEST_SIZE | float | Test data size for "Early stopping". Need to be from `0.0` to `1.0`. Default `0.2`. |
| USE_ON_MEMORY | bool | Load data on memory. If you use a big dataset, set it to `false`. Default `true` |
| USE_CACHE | bool | Image cache. If you use a big dataset, set it to `false`. If `USE_ON_MEMORY=true`, then `USE_CACHE=true` automatically. Default `true` |
| NUM_DATA_LOAD_THREAD | int | Number of thread image loads. MUST NOT over `BATCH_SIZE`. Default `1` |
| SEGMENTATION_MODEL | string | Segmentation Model "fcn_resnet101" or "deeplabv3_resnet101". Default `deeplabv3_resnet101` |
| DEVICE | string | Device name to use: "cuda" or "cpu". Default `cuda`. |
| FINE_TUNING | bool | If "False", only the last layer is trained. Default `False` |
| PRETRAINED | bool | If "True", training starts from pretrained model by MS COCO. Default `True` |


## Run on local
Set environment variables.

| env | type | description |
| --- | --- | --- |
| ABEJA_ORGANIZATION_ID | str | Your organization ID. |
| ABEJA_PLATFORM_USER_ID | str | Your user ID. |
| ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN | str | Your Access Token. |
| DATASET_ID | str | Dataset ID. |

```
$ DATASET_ID='xxx' ABEJA_ORGANIZATION_ID='xxx' ABEJA_PLATFORM_USER_ID='user-xxx' ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN='xxx' python train.py
```
