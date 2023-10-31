import torch
from tools.datasets import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets

    
data_route = {'cifar': '../data.cifar10/',
              'imagenet': '/datasets/imagenet/'}

cifar10_strings = ['vgg16','resnet18','densenet','effnet', 'resnet32_cifar', 'resnet20_cifar', 'mae']
imagenet_strings = ['vgg16_bn', 'resnet50', 'vit_small_patch16_224', 'vit_base_patch16_224', 'deit_base_patch16_224', 'deit_small_patch16_224', 'deit_tiny_patch16_224']

def dataset_loader(model,batch_size=100,num_workers=12, args=None):
    if args.dataset == "cifar" and model in cifar10_strings:
        print("Loading CIFAR-10 with batch size "+str(batch_size))
        train_loader,test_loader,calib_loader = get_cifar10_loaders(data_route['cifar'],batch_size,num_workers,args.calib_size)
    elif args.dataset == "imagenet" and model in imagenet_strings:
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if any(st in model for st in ["vit", "deit"]):
            args.mean = [0.5,] * 3
            args.std = [0.5,] * 3
        else:
            args.mean = IMAGENET_DEFAULT_MEAN
            args.std = IMAGENET_DEFAULT_STD
        if "384" in model:
            train_loader,test_loader = get_trainval_imagenet_dali_loader(args, batch_size, 384, 384)
            calib_loader = get_calib_imagenet_dali_loader(args, batch_size, 384, 384, calib_size=args.calib_size)
        else:
            train_loader,test_loader = get_trainval_imagenet_dali_loader(args, batch_size)
            calib_loader = get_calib_imagenet_dali_loader(args, batch_size, calib_size=args.calib_size)
        
        # train_loader = get_imagenet_iter_torch("train", image_dir=data_route["imagenet"], batch_size=batch_size)
        # test_loader = get_imagenet_iter_torch("val", image_dir=data_route["imagenet"], batch_size=batch_size)
    else:
        raise ValueError(f'Model not implemented for {model} :P')
    
    return train_loader,test_loader, calib_loader


def get_imagenet_iter_torch(type, image_dir, batch_size=128, num_threads=16, device_id=0, crop=224, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)
    return dataloader



import os, glob
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    print("ImportError: Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, testsize=-1, args=None):
    if testsize != -1:
        labels = []
        files = []
        # import pdb; pdb.set_trace()
        for i, l in enumerate(sorted(os.listdir(data_dir))):
            ps = glob.glob(os.path.join(data_dir, l, "*.JPEG"))
            files += ps
            labels += [i] * len(ps)
        labels = labels[::len(files) // testsize][:-1]
        files = files[::len(files) // testsize][:-1]
        print(is_training, len(files))
        images, labels = fn.readers.file(files=files,
                                        labels=labels,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in  the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
    images = fn.resize(images,
                        device=dali_device,
                        size=size,
                        mode="not_smaller",
                        interp_type=types.INTERP_CUBIC)
    # images = fn.jitter(images, 
    #                    interp_type=types.INTERP_CUBIC)

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[d * 255 for d in args.mean],
                                      std=[d * 255 for d in args.std],
                                      mirror=False)
    labels = labels.gpu()
    return images, labels


def get_trainval_imagenet_dali_loader(args, batchsize=32, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    args.testsize = -1
    args.val_testsize = -1 #args.calib_size
    traindir = os.path.join(data_route["imagenet"], 'train')
    
    pipe = create_dali_pipeline(batch_size=batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True,
                                testsize=args.testsize,
                                args=args)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    val_loader = get_val_imagenet_dali_loader(args, batchsize, crop_size, val_size)
    return train_loader, val_loader


def get_calib_imagenet_dali_loader(args, val_batchsize=32, crop_size=224, val_size=256, calib_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    valdir = os.path.join(data_route["imagenet"], 'train')
    pipe = create_dali_pipeline(batch_size=val_batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False,
                                testsize=calib_size,
                                args=args)
    pipe.build()
    loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.FILL, auto_reset=True)
    return loader


def get_val_imagenet_dali_loader(args, val_batchsize=32, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    valdir = os.path.join(data_route["imagenet"], 'val')
    pipe = create_dali_pipeline(batch_size=val_batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False,
                                testsize=args.val_testsize,
                                args=args)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return val_loader

