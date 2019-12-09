import datetime
import json
import os, glob
import random
import time
import traceback

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

import transforms as T
import parameters
import utils
from abeja_dataset import AbejaDataset, get_dataset_size
from callbacks import Statistics


if parameters.RANDOM_SEED is not None:
    torch.manual_seed(parameters.RANDOM_SEED)
    np.random.seed(parameters.RANDOM_SEED)
    random.seed(parameters.RANDOM_SEED)

statistics = Statistics(parameters.EPOCHS)

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)


def create_model(num_classes, model_name, pretrained=True, finetuning=False):
    """
    Create ML Network.
    :param num_classes: Number of classes.
    :param model_name: deeplabv3_resnet101 or fcn_resnet101.
    :param pretrained: if true, use pretrained model
    :param finetuning: if true, all paramters are trainable
    :return model: ML Network.
    """
    if model_name != 'deeplabv3_resnet101' and model_name != 'fcn_resnet101':
        raise ValueError(model_name + " is not supported")
    
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained)
    if model_name == 'deeplabv3_resnet101':
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if not finetuning:
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
    transforms = list()
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def get_trainval_dataset_index(dataset_list, early_stopping_test_size):
    train_dataset_id = None
    val_dataset_id = None
    train_list = None
    val_list = None
    for name, idx in dataset_list.items():
        if name == "val":
            val_dataset_id = idx
        elif train_dataset_id is None:
            train_dataset_id = idx
    
    if val_dataset_id is None:
        val_dataset_id = train_dataset_id
        dataset_size = get_dataset_size(train_dataset_id)
        test_size = int(dataset_size * early_stopping_test_size)
        train_list = range(test_size, dataset_size)
        val_list = range(0, test_size)
     
    return {
        'train_dataset_id': train_dataset_id, 'val_dataset_id': val_dataset_id,
        'train_list': train_list, 'val_list': val_list
    }
    

def criterion(inputs, target):
    losses = dict()
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, criterion, data_loader, device, num_classes):
    epoch_loss = list()
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            epoch_loss.append(loss.item())
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    print('eval confmat', confmat)
    acc_global, _, _ = confmat.compute()
    epoch_acc = acc_global.item()

    return epoch_acc, sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, num_classes):
    epoch_loss = list()
    model.train()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        epoch_loss.append(loss.item())
        confmat.update(target.flatten(), output['out'].argmax(1).flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    confmat.reduce_from_all_processes()
    acc_global, _, _ = confmat.compute()
    epoch_acc = acc_global.item()

    return epoch_acc, sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0


def handler(context):
    print('Start train handler.')
    if not isinstance(context, dict):
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        raise Exception(message)

    try:
        if ABEJA_TRAINING_RESULT_DIR:
            utils.mkdir(ABEJA_TRAINING_RESULT_DIR)

        utils.init_distributed_mode(parameters)

        device_name = parameters.DEVICE if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)

        trainval_info = get_trainval_dataset_index(context['datasets'], parameters.EARLY_STOPPING_TEST_SIZE)

        dataset = AbejaDataset(
            root=None, dataset_id=trainval_info['train_dataset_id'],
            transforms=get_transform(train=True),
            prefetch=parameters.USE_ON_MEMORY,
            use_cache=parameters.USE_CACHE,
            indices = trainval_info['train_list'])
        dataset_test = AbejaDataset(
            root=None, dataset_id=trainval_info['val_dataset_id'],
            transforms=get_transform(train=False),
            prefetch=parameters.USE_ON_MEMORY,
            use_cache=parameters.USE_CACHE,
            indices = trainval_info['val_list'])
        num_classes = dataset.num_class()
        
        if len(dataset) <= 0 or len(dataset_test) <= 0:
            raise Exception(
                "Training or Test dataset size is too small. "
                "Please add more dataset or set EARLY_STOPPING_TEST_SIZE properly"
            )
        drop_last = False if len(dataset) < parameters.BATCH_SIZE else True
 
        if parameters.DISTRIBUTED:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=parameters.BATCH_SIZE,
            sampler=train_sampler, num_workers=parameters.NUM_DATA_LOAD_THREAD,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=utils.collate_fn, drop_last=drop_last)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=parameters.NUM_DATA_LOAD_THREAD,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=utils.collate_fn)

        model = create_model(
            num_classes=num_classes,
            model_name=parameters.SEGMENTATION_MODEL,
            pretrained=parameters.PRETRAINED,
            finetuning=parameters.FINE_TUNING)
        model.to(device)

        if parameters.DISTRIBUTED:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if parameters.RESUME:
            checkpoint = torch.load(parameters.RESUME, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        model_without_ddp = model
        if parameters.DISTRIBUTED:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[parameters.GPU])
            model_without_ddp = model.module

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
        print(parameters.parameters)

        # save params
        save_param = {'SEGMENTATION_MODEL': parameters.SEGMENTATION_MODEL,'NUM_CLASSES': num_classes}
        with open(os.path.join(ABEJA_TRAINING_RESULT_DIR,'parameters.json'), 'w') as f:
            json.dump(save_param,f)

        start_time = time.time()
        t_epoch_start = time.time()
        for epoch in range(parameters.EPOCHS):
            if parameters.DISTRIBUTED:
                train_sampler.set_epoch(epoch)
            average_epoch_train_acc, average_epoch_train_loss = train_one_epoch(
                model, criterion, optimizer, data_loader, lr_scheduler,
                device, epoch, parameters.PRINT_FREQ, num_classes)
            average_epoch_val_acc, average_epoch_val_loss = evaluate(
                model, criterion, data_loader_test, device=device, num_classes=num_classes)

            t_epoch_finish = time.time()
            print('-------------')
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_TRAIN_Acc:{:.4f} '
                  '|| Epoch_VAL_Loss:{:.4f} || Epoch_VAL_Acc:{:.4f}'.format(
                epoch + 1, average_epoch_train_loss, average_epoch_train_acc,
                average_epoch_val_loss, average_epoch_val_acc))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            statistics(epoch + 1, average_epoch_train_loss, average_epoch_train_acc,
                       average_epoch_val_loss, average_epoch_val_acc)

            writer.add_scalar('main/loss', average_epoch_train_loss, epoch + 1)
            writer.add_scalar('main/acc', average_epoch_train_acc, epoch + 1)
            if (epoch + 1) % 10 == 0:
                writer.add_scalar('test/loss', average_epoch_val_loss, epoch + 1)
                writer.add_scalar('test/acc', average_epoch_val_acc, epoch + 1)
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'parameters': parameters.parameters
                    },
                    os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model_{}.pth'.format(epoch)))
            writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # save final model
        torch.save(model.to('cpu').state_dict(), os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pth'))

        # remove checkpoints files
        print('removing checkpoint files')
        rm_file_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model_*.pth')
        remove_files = glob.glob(rm_file_path)
        for f in remove_files:
            os.remove(f)
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        raise e
    writer.close()


if __name__ == '__main__':
    context = dict()
    context['datasets'] = {"data": os.environ.get('DATASET_ID')}
    handler(context)



