import os
import argparse
import matplotlib.pyplot as plt
import math
import albumentations
import numpy as np
import segmentation_models_pytorch as smp
import torch
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import multiprocessing as mlt
import glob


#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--input_imgs', default='.\*.tiff')
parser.add_argument('--model_path', default='.\Models\Keras\trained_model\DeepLabV3+\DHA_Built_vs_Unbuilt_temporal2')
parser.add_argument('--output_dir', default='.\Outputs')


target_size = (256, 256)
padding_pixels = 32
padding_value = 0

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

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

def process(args):
    img_path, device, model, output_dir, pos = args

    file_name = os.path.split(img_path)[-1].split('.')[0]

    sub_dirs = os.path.split(img_path[pos:])[0]
    out_path = os.path.join(output_dir,sub_dirs)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, file_name + "_preds.png")
    chk_path = save_path.split('.')[0]+'.tif'
    if os.path.exists(save_path) or os.path.exists(chk_path):
        print("Prediction for file {} already exists, skipping..!".format(file_name))
        return
    img = np.array(Image.open(img_path))
    img = cv2.copyMakeBorder(img, padding_pixels, padding_pixels, padding_pixels, padding_pixels,
                             cv2.BORDER_CONSTANT, value=padding_value)
    src_im_height = img.shape[0]
    src_im_width = img.shape[1]

    cols = (math.ceil(src_im_height/target_size[0]))
    rows = (math.ceil(src_im_width/target_size[1]))

    print("Total {} patches for the given image {}".format(rows*cols, file_name))
    combined_image = np.zeros((cols*target_size[0], rows*target_size[1]), dtype=np.uint8) * 255
    #useful_portion = np.zeros((src_im_height-2*padding_pixels, src_im_width-2*padding_pixels), dtype=np.uint8)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    x1, y1, idx = 0, 0, 0
    while y1 < src_im_height:
        y2 = y1 + target_size[0] + 2 * padding_pixels
        while x1 < src_im_width:
            x2 = x1 + target_size[1] + 2 * padding_pixels
            img_crop = img[y1: y2, x1: x2]
            pad_bottom = y2 - src_im_height if y2 > src_im_height else 0
            pad_right = x2 - src_im_width if x2 > src_im_width else 0

            if pad_bottom > 0 or pad_right > 0:
                img_crop = cv2.copyMakeBorder(img_crop, 0, pad_bottom, 0, pad_right,
                                              cv2.BORDER_CONSTANT, value=padding_value)

            sample = preprocessing(image=img_crop)
            image = sample['image']
            image = np.transpose(image, (2, 0, 1)).astype('float32')
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            pred_mask = model(x_tensor)
            pr_mask = pred_mask.squeeze()
            pr_mask = pr_mask.detach().squeeze().cpu().numpy()
            mask = pr_mask.round()
            sizex, sizey = target_size
            patch = mask[padding_pixels:sizex + padding_pixels, padding_pixels:sizey + padding_pixels]
            combined_image[y1: y1+target_size[1], x1: x1+target_size[0]] = patch

            x1 += target_size[0]
            idx += 1
            print("Patch {} done for file {}".format(idx, file_name))
            #break
        x1 = 0
        y1 += target_size[1]
        #break

    #useful_portion = combined_image[:src_im_height-2*padding_pixels, :src_im_width-2*padding_pixels]
    plt.imsave(save_path, combined_image[:src_im_height-2*padding_pixels, :src_im_width-2*padding_pixels])

    print_line = "Processing completed for {}..!".format(file_name)
    print(print_line)


    
if __name__ == '__main__':
    
    args = parser.parse_args()
    print()
    print("Evaluation start...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best weights
    print("Loading Model...")
    model = torch.load(os.path.join(args.model_path, 'best_model.h5'), map_location=DEVICE)

    imgs = [file for file in glob.glob(args.input_imgs) if file.endswith('.tif')]

    assert len(imgs) > 0, "The number of images equal to zero"

    num_processes = 1
    print("Running on {} images using {} parallel processes".format(len(imgs), num_processes))

    pos = len(args.input_imgs.split('*')[0])

    args = [[img, DEVICE, model, args.output_dir, pos] for img in imgs]

    if num_processes > 1:
        p = mlt.Pool(num_processes)
        (p.map(process, args))
        p.close()
    else:
        for arg in args:
            process(arg)
    print("Completed")