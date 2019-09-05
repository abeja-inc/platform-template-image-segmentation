import datetime
import os
import time
import traceback

import torch
import torch.utils.data
from torch import nn
import torchvision

import json
import transforms as T
import utils

from abeja_dataset import AbejaDataset, getDatasetSize
import parameters

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')

"""
def get_dataset(name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": ('/root/Datasets/VOC2012/', torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": ('/datasets01/SBDD/072318/', sbd, 21),
        "coco": ('/datasets01/COCO/022719/', get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes
"""


def create_model(num_classes, model_name, pretrained = True, finetuning = False):
    """
    Create ML Network.
    :param num_classes: Number of classes.
    :param model_name: deeplabv3_resnet101 or fcn_resnet101.
    :param finetuning: if true, all paramters are trainable
    :return model: ML Network.
    """
    if((model_name!='deeplabv3_resnet101') and (model_name!='fcn_resnet101')):
        raise ValueError(model_name + " is not supported")
    
    model = torchvision.models.segmentation.__dict__[model_name](pretrained = pretrained)
    if(model_name=='deeplabv3_resnet101'):
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if(not finetuning):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.aux_classifier.parameters():
            param.requires_grad = True

    return model



def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def handler(context):
    print('Start train handler.')
    if not isinstance(context, dict):
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        raise Exception(message)

    try:
        args = parse_args()
        args.batch_size = parameters.BATCH_SIZE
        args.epochs = parameters.EPOCHS
        args.workers = parameters.NUM_DATA_LOAD_THREAD
        args.output_dir = ABEJA_TRAINING_RESULT_DIR
        args.model = parameters.SEG_MODEL

        if ABEJA_TRAINING_RESULT_DIR:
            utils.mkdir(ABEJA_TRAINING_RESULT_DIR)

        utils.init_distributed_mode(args)

        device = torch.device(parameters.DEVICE)

        DATASET_ID = ""
        dataset_ids = context['datasets'].values()
        for idx in dataset_ids:
            DATASET_ID = idx
            break
        
        dataset_size = getDatasetSize(DATASET_ID)
        test_size = int(dataset_size * parameters.EARLY_STOPPING_TEST_SIZE)
        train_list = range(test_size,dataset_size)
        test_list = range(0,test_size)
        
        dataset =  AbejaDataset(root = None,
                            dataset_id = DATASET_ID,
                            transforms=get_transform(train=True),
                            indices = train_list)
        dataset_test =  AbejaDataset(root = None,
                            dataset_id = DATASET_ID,
                            transforms=get_transform(train=False),
                            indices = test_list)
        num_classes = dataset.num_class()
 
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters.BATCH_SIZE,
            sampler=train_sampler, num_workers=parameters.NUM_DATA_LOAD_THREAD,
            collate_fn=utils.collate_fn, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=parameters.NUM_DATA_LOAD_THREAD,
            collate_fn=utils.collate_fn)

        model = create_model(num_classes=num_classes, 
                        model_name=parameters.SEG_MODEL, 
                        pretrained=parameters.PRETRAINED, 
                        finetuning=parameters.FINE_TUNING)
        model.to(device)

        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if parameters.RESUME:
            checkpoint = torch.load(parameters.RESUME, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        if parameters.TEST_ONLY:
            confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
            print(confmat)
            return

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        if parameters.AUX_LOSS:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": parameters.LEARNING_RATE * 10})
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=parameters.LEARNING_RATE, momentum=parameters.MOMENTUM, weight_decay=parameters.WEIGHT_DECAY)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(data_loader) * parameters.EPOCHS)) ** 0.9)

        print('num classes:', num_classes)
        print(len(dataset), 'train samples')
        print(len(dataset_test), 'test samples')
        print(args)
    
        start_time = time.time()
        for epoch in range(parameters.EPOCHS):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, parameters.PRINT_FREQ)
            confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
            print(confmat)
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                },
                os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model_{}.pth'.format(epoch)))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        #save final model
        torch.save(model.state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
        save_param = {'SEG_MODEL': parameters.SEG_MODEL,'NUM_CLASSES': num_classes}
        f = open(os.path.join(ABEJA_TRAINING_RESULT_DIR,'parameters.json'), 'w')
        json.dump(save_param,f)
        f.close()

                   
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        raise e


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--dataset', default='voc', help='dataset')
    parser.add_argument('--model', default='fcn_resnet101', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    #parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_false",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    context = dict()
    context['datasets'] = {"data": os.environ.get('DATASET_ID')}
    handler(context)



