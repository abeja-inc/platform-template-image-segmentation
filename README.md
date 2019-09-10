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
| env | type | required | default | description |
| --- | --- | --- | --- | --- |
| BATCH_SIZE | int | true | 32 | Batch size. |
| EPOCHS | int | true | 50 | Epoch number. This template applies "Early stopping". |
| LEARNING_RATE | float | true | 0.01 | Learning rate. Need to be from `0.0` to `1.0`. |
| MOMENTUM | float | true | 0.9 | Weight of the previous update. Need to be from `0.0`. |
| WEIGHT_DECAY | float | true | 1e-4 | SGD parameter "decay". Need to be from `0.0`. |
| USE_ON_MEMORY | bool | true | true | Load data on memory. If you use a big dataset, set it to `false`. |
| USE_CACHE | bool | true | true | Image cache. If you use a big dataset, set it to `false`. If `USE_ON_MEMORY=true`, then `USE_CACHE=true` automatically. |
| NUM_DATA_LOAD_THREAD | int | true | 1 | Number of thread image loads. MUST NOT over `BATCH_SIZE`. |
| SEGMENTATION_MODEL | string | true | deeplabv3_resnet101 | Segmentation Model "fcn_resnet101" or "deeplabv3_resnet101". |
| DEVICE | string | true | cuda | Device name to use: "cuda" or "cpu". |
| FINE_TUNING | bool | true | false | If "False", only the last layer is trained. |
| PRETRAINED | bool | true | true | If "True", training starts from pretrained model by MS COCO. |
| PRINT_FREQ | int | true | 10 | Log frequency (epoch). |
| RANDOM_SEED | int | false | 42 | Random seed. |
| EARLY_STOPPING_TEST_SIZE | float | false | 0.2 | Test data size for "Early stopping". Need to be from `0.0` to `1.0`. |
| RESUME | str | false | None | Filepath. Set if you want to use pretrained your model. |
| AUX_LOSS | bool | false | false | Set if you want to use aux loss. |

### TBD
Distributed mode is being developed.

| env | type | required | default | description |
| --- | --- | --- | --- | --- |
| LOCAL_RANK | int | false | 0 | Name of the GPU to use |
| SLURM_PROCID | int | false | None | SLURM PROCID. |
| RANK | int | false | 1 | Rank of the current process. |
| DIST_URL | string | false | env:// | URL specifying how to initialize the process group. |
| WORLD_SIZE | int | false | 1 | Number of processes participating in the job. |


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
