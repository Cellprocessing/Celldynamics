
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
use_GPU = core.use_gpu()
initial_model = "cyto"
test_dir = "F:/spyder/project/cellpose_seg/test"

model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)
diam_labels = model.diam_labels.copy()

Channel_to_use_for_training = "Green"  # @param ["Grayscale", "Blue", "Green", "Red"]

# @markdown ###If you have a secondary channel that can be used for training, for instance nuclei, choose it here:

Second_training_channel = "Red"  # @param ["None", "Blue", "Green", "Red"]

if Channel_to_use_for_training == "Grayscale":
    chan = 0
elif Channel_to_use_for_training == "Blue":
    chan = 3
elif Channel_to_use_for_training == "Green":
    chan = 2
elif Channel_to_use_for_training == "Red":
    chan = 1

if Second_training_channel == "Blue":
    chan2 = 3
elif Second_training_channel == "Green":
    chan2 = 2
elif Second_training_channel == "Red":
    chan2 = 1
elif Second_training_channel == "None":
    chan2 = 0

output = io.load_train_test_data(test_dir,image_filter="img", mask_filter='masks')
#print(np.shape(output))
test_data, test_labels = output[:2]

#print(np.shape(test_data))
# run model on test images
masks = model.eval(test_data,
                   channels=[chan, chan2],
                   diameter=diam_labels)[0]



# print(np.shape(test_labels[0]))
# print(np.shape(masks[0]))
#print(test_labels,masks)
# check performance using ground truth labels


ap = metrics.average_precision(np.array(test_labels[0],dtype=np.int16), masks[0],threshold=[0.5])[0]
print('')
print(f'>>> average precision at iou threshold 0.5 = {ap.mean():.3f}')


"""

0.5

average precision at iou threshold 0.5 = 0.942
average precision at iou threshold 0.5 = = 0.921


0.7

average precision at iou threshold 0.5 = 0.796
average precision at iou threshold 0.5 = 0.781
 




0.9
average precision at iou threshold 0.5 = 0.201
average precision at iou threshold 0.5 = 0.305


"""



