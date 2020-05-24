from grains import GrainsDataset
from grains import GrainsMaskRCNNConfig
from Mask_RCNN.mrcnn import model as modellib
import time

# prepare train set
train_set = GrainsDataset()
train_set.load_dataset('datasets', is_train = True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))# prepare test/val set

test_set = GrainsDataset()
test_set.load_dataset('datasets', is_train = False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

config = GrainsMaskRCNNConfig()

model = modellib.MaskRCNN(mode = "training", config = config, model_dir='models')

model.load_weights('Mask_RCNN/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=5, layers='heads')

model_path = 'models/mask_rcnn_grains.h5'
model.keras_model.save_weights(model_path)
