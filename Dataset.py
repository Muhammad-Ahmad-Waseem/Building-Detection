import os
import numpy as np
import cv2
import keras

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    '''
    CLASSES = ['commercial', 'residential','background']
    colors = [
        (0, 255, 251), # Commercial
        (234, 113, 245), # Residential
        (255, 255, 255) # background
        ]
    CLASSES = ['built-up', 'underconst', 'vegetation','background']
    colors = [
        (164, 113, 88), # Built-up
        (255, 127, 0), # UnderConst
        (38, 137, 23), # Vegetation
        (255, 255, 255) # background
        ]'''
    CLASSES = ['built-up','background']
    colors = [
        (164, 113, 88), # Built-up
        (255, 255, 255) # background
        ]
        
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_ids = np.sort(os.listdir(images_dir))
        self.mask_ids = np.sort(os.listdir(masks_dir))
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #print("Mask shape after Loading:{}".format(mask.shape))
        # convert RGB mask to index 
        one_hot_map = []
        for color in self.colors:
            class_map = np.all(np.equal(mask, color), axis=-1)
            one_hot_map.append(class_map)
        
        one_hot_map = np.stack(one_hot_map, axis=-1)
        one_hot_map = one_hot_map.astype('float32')
        
        mask = np.argmax(one_hot_map, axis=-1)
        
        #Label conflicting edges as last class(background)
        min_mask = np.argmin(one_hot_map, axis=-1)
        mask[(mask==min_mask)]=(len(self.colors))-1
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        #print("Mask Shape after process 1: {}".format(mask.shape))
        
        #add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        #print("Mask Shape after process 2: {}".format(mask.shape))
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #print("Mask Shape after process 3: {}".format(mask.shape))
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #print("Mask Shape after process 4: {}".format(mask.shape))    
        image = np.transpose(image, (2, 0, 1)).astype('float32') 
        mask = np.transpose(mask, (2, 0, 1))
        #print("Mask Shape returned: {}".format(mask.shape))
        #print("Changed")
        #print(mask.shape)
        return image,mask,self.image_ids[i] 
        
    def __len__(self):
        return len(self.image_ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)