import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import albumentations
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
import sys
import cv2

from Dataset import Dataset
from torch.utils.data import DataLoader
#import train as trn
#from Augmentations import Augmentations
import utils

#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../Datasets/Segmentation_Data')
parser.add_argument('--output_dir', default='./Keras/trained_model/satellite')
parser.add_argument('--backbone', default='efficientnetb3')
parser.add_argument('-m',   '--mode', choices=['t', 'es', 'ct','cf', 'v', 'ea'], default='t', required='true')

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albumentations.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albumentations.RandomCrop(height=320, width=320, always_apply=True),
        albumentations.IAAAdditiveGaussianNoise(p=0.2),
        albumentations.IAAPerspective(p=0.5),
        albumentations.OneOf(
            [
                albumentations.CLAHE(p=1),
                albumentations.RandomBrightness(p=1),
                albumentations.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.IAASharpen(p=1),
                albumentations.Blur(blur_limit=3, p=1),
                albumentations.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.RandomContrast(p=1),
                albumentations.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albumentations.Lambda(mask=round_clip_0_1)
    ]
    
    return albumentations.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albumentations.PadIfNeeded(384, 480)
    ]
    return albumentations.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
    ]
    return albumentations.Compose(_transform)
    
if __name__ == '__main__':
    
    args = parser.parse_args()
    CLASSES = ['built-up']
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    BACKBONE = args.backbone
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VAL = 4
    LR = 0.00008
    EPOCHS = 40
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    
    # uncomment when wnat a classification head..!
    '''aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    #activation='sigmoid',      # activation function, default is None
    classes=n_classes,                 # define number of output labels
    )'''
    
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, 
        activation=ACTIVATION#,
        #aux_params=aux_params
        )
    # print(model)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # define optomizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=LR),
    ])
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )
    
    #Define Loss type
    mode = smp.losses.constants.BINARY_MODE if n_classes == 1 else smp.losses.constants.MULTICLASS_MODE
    print("Classification Mode: {}".format(mode))
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = smp.utils.losses.DiceLoss()
    
    #focal_loss = smp.utils.losses.BinaryFocalLoss() if n_classes == 1 else smp.utils.losses.CategoricalFocalLoss()
    loss = dice_loss #+ (1 * focal_loss)
    
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    # Use this mode for evaluating threshold's tpr and fpr
    if (args.mode == 'v'):
        #print(model)
        
        imgs_dir = os.path.join(args.data_dir, 'images')
        segm_dir = os.path.join(args.data_dir, 'segmentations')
            
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load best weights
        model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
        
        dataset = Dataset(
            imgs_dir, 
            segm_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        i = 0
        tpr_vals = []
        fpr_vals = []
        thr = 0.98
        
        with tqdm(dataset, file=sys.stdout) as iterator:
            for it in iterator:
                image, gt_mask,f_name = dataset[i]
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                image = image.squeeze()
                image = np.transpose(image,(1,2,0))
                gt_mask = gt_mask.squeeze()
                
                pred_mask = model.forward(x_tensor)
                pr_mask = pred_mask.squeeze()
                pr_mask = pr_mask.detach().squeeze().cpu().numpy()
                
                pred = (pr_mask>thr)
                i = i+1
                
                #continue
                if(np.sum((gt_mask == 1)) == 0):#For some gt masks, there isn't a single pixel of 1, which makes tp + fn = 0. Ignore those
                    print("Ground truth {} is invalid, skipping".format(f_name))
                    continue
                
                tp = np.sum((pred == 1) & (gt_mask == 1))
                fn = np.sum((pred == 0) & (gt_mask == 1))
                fp = np.sum((pred == 1) & (gt_mask == 0))
                tn = np.sum((pred == 0) & (gt_mask == 0))
                
                tpr = tp/(tp+fn)
                fpr = fp/(tn+fp)
                tpr_vals.append(tpr)
                fpr_vals.append(fpr)
                
        np.save("tpr_array_at_{}.npy".format(thr),tpr_vals)
        np.save("fpr_array_at_{}.npy".format(thr),fpr_vals)
        print(sum(tpr_vals)/len(tpr_vals))
        print(sum(fpr_vals)/len(fpr_vals))
        
        
    if (args.mode=='t' or args.mode=='ct'):# This mode is used for training or conituning training.
        #//////////////////////// DATASET PATH \\\\\\\\\\\\\\\\\\\\\\\\
        x_train_dir = os.path.join(args.data_dir, 'images')
        y_train_dir = os.path.join(args.data_dir, 'segmentations')
        
        x_valid_dir = os.path.join(args.data_dir, 'test_images')
        y_valid_dir = os.path.join(args.data_dir, 'test_segmentations')
        
        # Set device: `cuda` or `cpu`
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if (args.mode=='t'):
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            else:
                print("Warining Model path {} already exists".format(args.output_dir))
        
        if (args.mode=='ct'):
            assert os.path.exists(args.output_dir)
            model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
            print('Loaded pre-trained DeepLabV3+ model!')

        # Dataset for train images
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            classes=CLASSES, 
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        # Dataset for validation images
        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir,
            classes=CLASSES, 
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )
        
        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )
        
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []
        print("Total Epochs: {}".format(EPOCHS))
        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            torch.save(model, os.path.join(args.output_dir,'best_model.h5'))
            
    if(args.mode=='ea'):# This mode is used for saving predictions against images in specified folder. The name of pred image is same as inp.
        imgs_dir = os.path.join(args.data_dir, 'Images_divide')
        pred_dir = os.path.join(args.data_dir, 'vis')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load best weights
        model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
        dataset = Dataset(
            imgs_dir, 
            imgs_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        i = 0
        with tqdm(dataset, file=sys.stdout) as iterator:
            for it in iterator:
                image, _,f_name = dataset[i]
                i = i+1
                save_path = os.path.join(pred_dir,f_name)
                if(os.path.exists(save_path)):
                    print("Path already exists, skipping")
                    continue
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                image = image.squeeze()
                image = np.transpose(image,(1,2,0))
                pred_mask = model(x_tensor)
                pr_mask = pred_mask.round().squeeze()
                pr_mask = pr_mask.detach().squeeze().cpu().numpy()
                '''
                if mode == smp.losses.constants.BINARY_MODE:
                    pr_mask = np.expand_dims(pr_mask, 0)
                
                #print(pr_mask.shape)
                pr_mask = np.transpose(pr_mask,(1,2,0))
                pr_img = np.zeros((pr_mask.shape[0],pr_mask.shape[1],3))
                for j in range(pr_mask.shape[2]):
                    pr_img = pr_img+ utils.MasktoRGB(pr_mask[...,j],Dataset.colors[j])'''
                    
                plt.imsave(save_path,pr_mask.astype(np.uint8))
            
    if(args.mode=='cf'):# This mode is used to predict values b/w 0 and 1 instead of just binary values 0,1, to estimate confidences
        imgs_dir = os.path.join(args.data_dir, 'Images')
        pred_dir = os.path.join(args.data_dir, 'confidences')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load best weights
        model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
        dataset = Dataset(
            imgs_dir, 
            imgs_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        i = 0
        with tqdm(dataset, file=sys.stdout) as iterator:
            for it in iterator:
                image, _,f_name = dataset[i]
                name = f_name.split(".")[0] + ".tif"
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                image = image.squeeze()
                image = np.transpose(image,(1,2,0))
                pred_mask = model(x_tensor)
                pr_mask = pred_mask.squeeze()
                pr_mask = pr_mask.detach().squeeze().cpu().numpy()
                
                save_path = os.path.join(pred_dir,name)
                Image.fromarray(pr_mask).save(save_path)
                i = i+1
                
    if (args.mode=='es'):# This mode simply visualizes test data along with their generated mask(images aren't saved)
                
        x_test_dir = os.path.join(args.data_dir, 'test_images')
        y_test_dir = os.path.join(args.data_dir, 'test_segmentations')
        
        
        # Set device: `cuda` or `cpu`   
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir, 
            classes=CLASSES, 
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        # load best weights
        best_model = torch.load(os.path.join(args.output_dir,'best_model.h5'), map_location=DEVICE)
        
        for i in range(len(test_dataset)):
            image, gt_mask, name = test_dataset[i]
            gt_mask = np.transpose(gt_mask,(1,2,0))
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            image = image.squeeze()
            image = np.transpose(image,(1,2,0))
            
            pred_mask = best_model(x_tensor)
            pr_mask = pred_mask.round().squeeze()
            pr_mask = pr_mask.detach().squeeze().cpu().numpy()
            
            if mode == smp.losses.constants.BINARY_MODE:
                pr_mask = np.expand_dims(pr_mask, 0)
        
            #print(pr_mask.shape)
            pr_mask = np.transpose(pr_mask,(1,2,0))
            pr_img = np.zeros((pr_mask.shape[0],pr_mask.shape[1],3))
            gt_img = np.zeros((gt_mask.shape[0],pr_mask.shape[1],3))
            for j in range(pr_mask.shape[2]):
                pr_img = pr_img+ utils.MasktoRGB(pr_mask[...,j],Dataset.colors[j])
                gt_img = gt_img+ utils.MasktoRGB(gt_mask[...,j],Dataset.colors[j])
                
            utils.visualize(
                image=utils.denormalize(image.squeeze()),
                gt_mask=gt_img.astype(np.uint8),
                pr_mask=pr_img.astype(np.uint8),
            )
               
