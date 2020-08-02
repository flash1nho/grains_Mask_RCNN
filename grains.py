from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.utils import Dataset

import numpy as np

from os import listdir
from xml.etree import ElementTree

class GrainsMaskRCNNConfig(Config):
    # define the name of the configuration
    NAME = 'grains'
    # number of classes (background + any grains)
    NUM_CLASSES = 1 + 3

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512

    STEPS_PER_EPOCH = 131

    LEARNING_RATE = 0.006

    DETECTION_MIN_CONFIDENCE = 0.9

    MAX_GT_INSTANCES = 10

    TRAIN_ROIS_PER_IMAGE = 200

class GrainsDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "grains_2mm")
        self.add_class("dataset", 2, "grains_2_5mm")
        self.add_class("dataset", 3, "grains_3mm")
        
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            image_id = filename[:-4]
            
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:
                continue
            
            # setting image file
            img_path = images_dir + filename
            
            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3])# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()

        for box in root.findall('.//object'):
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            if name == 'grains_2mm' or name == 'grains_2_5mm' or name == 'grains_3mm':
                boxes.append(coors)

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height # load the masks for an image

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, w, h = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]

            if box[4] == 'grains_2mm':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('grains_2mm'))
            elif box[4] == 'grains_2_5mm':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('grains_2_5mm'))
            elif box[4] == 'grains_3mm':
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('grains_3mm'))

        return masks, np.asarray(class_ids, dtype='int32') # load an image reference

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']
