import argparse
import random
import os
import numpy as np

from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default="DHA_GEID_Current/images", help= "Directory with the SIGNS dataset")
parser.add_argument('--masks_dir', default="DHA_GEID_Current/segmentations", help= "Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default="Datasets/Land_Use_DHA/built_vs_unbuilt_LR", help="Where to write the new data")



def save(filename, output_dir):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    #image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.images_dir), "Couldn't find the dataset at {}".format(args.images_dir)
    assert os.path.isdir(args.masks_dir), "Couldn't find the dataset at {}".format(args.masks_dir)
    
    #print(args.data_dir)
    #print(args.output_dir)

    #classes = os.listdir(args.data_dir)
    # Define the data directories
    #for class_name in classes:
    #class_dir = os.path.join(args.data_dir)
    images_name = os.listdir(args.images_dir)
    images_name.sort()
    masks_name = os.listdir(args.masks_dir)
    masks_name.sort()
    
    #print(images_name)
    #print(masks_name)
    random.seed(230)
    c = list(zip(images_name, masks_name))
    random.shuffle(c)
    
    images_name, masks_name = zip(*c)
    
    images_path = [os.path.join(args.images_dir, f) for f in images_name if f.endswith('.png')]
    masks_path = [os.path.join(args.masks_dir, f) for f in masks_name if f.endswith('.png')]
    
    #images_path.sort()
    #masks_path.sort()
    
    train_index = int(0.8 * len(images_path))
    #val_index   = int(0.9 * len(class_images_path))
    train_filenames_img = images_path[:train_index]
    train_filenames_msk = masks_path[:train_index]
    #val_filenames   = class_images_path[train_index:val_index]
    test_filenames_img  = images_path[train_index:]
    test_filenames_msk = masks_path[train_index:]
    filenames = {'images': train_filenames_img,'segmentations': train_filenames_msk,'test_images': test_filenames_img,'test_segmentations': test_filenames_msk}
    
    #filenames = {'segmentations2': train_filenames_msk,'test_segmentations2': test_filenames_msk}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))
        
    for split in ['images','segmentations','test_images','test_segmentations']:
        print(split)
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            #print('here')
            os.makedirs(output_dir_split)
            
        print("Processing data, saving preprocessed data to {}".format(split))
        for filename in tqdm(filenames[split]):
            save(filename, output_dir_split)
            #print(filename)

    print("Done building dataset")
