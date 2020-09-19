import argparse
from keras.preprocessing.image import img_to_array
from Mask_RCNN.mrcnn import model as modellib
from Mask_RCNN.mrcnn import visualize
from grains import GrainsMaskRCNNConfig
import skimage

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image...")
args = vars(ap.parse_args())

config = GrainsMaskRCNNConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='models')

model_path = 'models/mask_rcnn_grains.h5'
model.load_weights(model_path, by_name=True)

image = skimage.io.imread(args['image'])
results = model.detect([image], verbose=1) # Display results
result = results[0]

visualize.save_image(
  image,
  'roi',
  result['rois'],
  result['masks'],
  result['class_ids'],
  result['scores'],
  ['__background__', 'grains'],
  filter_classs_names = ['grains'],
  scores_thresh = 0.9,
  mode = 0
)
