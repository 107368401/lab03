import os
import re
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd

def load_xml(xml_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    folder = root[0].text
    filename = root[1].text
    size = [int(root[4][0].text), int(root[4][1].text), int(root[4][2].text)]
    object_name = root[6][0].text
    bbox = [int(root[6][4][0].text), int(root[6][4][1].text), int(root[6][4][2].text), int(root[6][4][3].text)]
    return folder, filename, size, object_name, bbox

# data_dir = './data'
pattern = re.compile('(.+\/)?(\w+)\/([^_]+)_.+xml')
# all_files = glob(os.path.join(data_dir, 'train/*xml'))
all_files = glob(os.path.join('train/*xml'))
all_files = [re.sub(r'\\', r'/', file) for file in all_files]
print(len(all_files))

frames = []
xmins = []
xmaxs = []
ymins = []
ymaxs = []
class_ids = []
i = 0
for entry in all_files:
    r = re.match(pattern, entry)
    if r:
        folder, filename, size, object_name, bbox = load_xml(entry)
        file_name, file_extension = os.path.splitext(filename)
        if not file_name in entry:
            # print(filename, entry)
            continue
        frames.append(filename)
        xmins.append(bbox[0])
        xmaxs.append(bbox[1])
        ymins.append(bbox[2])
        ymaxs.append(bbox[3])
        class_ids.append(0)
        if i == 18000:
            frames = [re.sub(r'xml', r'jpg', frame) for frame in frames]
            train_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs, 'ymin': ymins, 'ymaxs': ymaxs, 'class_id': class_ids})
            train_labels = train_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymaxs', 'class_id']]
            train_labels.to_csv("train_labels.csv", index=False)
            frames = []
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            class_ids = []
        i += 1
frames = [re.sub(r'xml', r'jpg', frame) for frame in frames]
val_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs, 'ymin': ymins, 'ymaxs': ymaxs, 'class_id': class_ids})
val_labels = val_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymaxs', 'class_id']]
val_labels.to_csv("val_labels.csv", index=False)


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

# 1. Set the model configuration parameters
img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
# img_height = 240 # Height of the input images
# img_width = 320 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 6 # Number of classes including the background class
# n_classes = 1 # Number of classes including the background class
min_scale = 0.08 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]

# II. Build or load the model
# II.1 Create a new model
K.clear_session() # Clear previous models from memory.
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=None,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# III. Set up the data generators for the training
train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

train_images_path = './sample/data/udacity_driving_datasets/'
train_labels_path = './sample/data/udacity_driving_datasets/train_labels.csv'
val_images_path = './sample/data/udacity_driving_datasets/'
val_labels_path = './sample/data/udacity_driving_datasets/val_labels.csv'
# train_images_path = './data/train'
# train_labels_path = './train_labels.csv'
# val_images_path = './data/train'
# val_labels_path = './val_labels.csv'

train_dataset.parse_csv(images_path=train_images_path,
                        labels_path=train_labels_path,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_path=val_images_path,
                      labels_path=val_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer=None,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

batch_size = 16

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5), # Randomly change brightness between 0.5 and 2 with probability 0.5
                                         flip=0.5, # Randomly flip horizontally with probability 0.5
                                         translate=((5, 50), (3, 30), 0.5), # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                         scale=(0.75, 1.3, 0.5), # Randomly scale between 0.75 and 1.3 with probability 0.5
                                         max_crop_and_resize=False,
                                         full_crop_and_resize=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=False,
                                     full_crop_and_resize=False,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     diagnostics=False)

n_train_samples = train_dataset.get_n_samples()
n_val_samples = val_dataset.get_n_samples()

# IV. Run the training
epochs = 10

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(n_train_samples/batch_size),
                              epochs=epochs,
                              callbacks=[ModelCheckpoint('ssd7_weights_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           period=1),
                                           EarlyStopping(monitor='val_loss',
                                                         min_delta=0.001,
                                                         patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5,
                                                             patience=0,
                                                             epsilon=0.001,
                                                             cooldown=0)],
                              validation_data=val_generator,
                              validation_steps=ceil(n_val_samples/batch_size))

model_name = 'ssd7'
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))
print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()

# V. Make predictions
