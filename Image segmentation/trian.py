
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob
from natsort import natsorted


# url = 'https://drive.google.com/uc?id=1HXpLczf7TPCdI1yZY5KV3EkdWzRrgvhQ'
# utils.download_url_to_file(url, 'human_in_the_loop.zip')

# start logger (to see training across epochs)
logger = io.logger_setup()

train_dir = "F:/spyder/project/cellpose_seg/train"  # @param {type:"string"}
test_dir = "F:/spyder/project/cellpose_seg/test"  # @param {type:"string"}
# Define where the patch file will be saved
base = "/content"

# model name and path
# @markdown ###Name of the pretrained model to start from and new model name:
from cellpose import models

initial_model = "cyto"  # @param ['cyto','nuclei','tissuenet','livecell','cyto2','CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4','scratch']
model_name = "CP_tissuenet"  # @param {type:"string"}

# other parameters for training.
# @markdown ###Training Parameters:
# @markdown Number of epochs:
n_epochs = 500 # @param {type:"number"}
use_GPU = core.use_gpu()


Channel_to_use_for_training = "Green"  # @param ["Grayscale", "Blue", "Green", "Red"]

# @markdown ###If you have a secondary channel that can be used for training, for instance nuclei, choose it here:

Second_training_channel = "Red"  # @param ["None", "Blue", "Green", "Red"]

# @markdown ###Advanced Parameters

Use_Default_Advanced_Parameters = True  # @param {type:"boolean"}
# @markdown ###If not, please input:
learning_rate = 0.001  # @param {type:"number"}
weight_decay = 0.0001  # @param {type:"number"}

if (Use_Default_Advanced_Parameters):
    print("Default advanced parameters enabled")
    learning_rate = 0.001
    weight_decay = 0.0001

# here we check that no model with the same name already exist, if so delete
model_path = train_dir + 'models/'
if os.path.exists(model_path + '/' + model_name):
    print("!! WARNING: " + model_name + " already exists and will be deleted in the following cell !!")

if len(test_dir) == 0:
    test_dir = None

# Here we match the channel to number
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

if initial_model == 'scratch':
    initial_model = 'None'



# DEFINE CELLPOSE MODEL (without size model)
model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

# set channels
channels = [chan, chan2]

# get files
output = io.load_train_test_data(train_dir, test_dir,image_filter="img", mask_filter='masks')
train_data, train_labels, _, test_data, test_labels, _ = output

new_model_path = model.train(train_data, train_labels,
                              test_data=test_data,
                              test_labels=test_labels,
                              channels=channels,
                              save_path=train_dir,
                              n_epochs=n_epochs,
                              learning_rate=learning_rate,
                              weight_decay=weight_decay,
                              nimg_per_epoch=8,
                              model_name=model_name)

# diameter of labels in training images
diam_labels = model.diam_labels.copy()






#####################################################################################################################
output = io.load_train_test_data(test_dir,image_filter="img", mask_filter='masks')
#print(np.shape(output))
test_data, test_labels = output[:2]
#print(np.shape(test_data))
# run model on test images
print(new_model_path)
model.load_pred(new_model_path)




#print(np.shape(test_labels),np.shape(masks))
#print(test_labels,masks)
# check performance using ground truth labels

ap_list=[]
for i in range(len(test_labels)):
    #print("iii:",i)
    masks = model.eval(test_data[i],
                       channels=[chan, chan2],
                       diameter=diam_labels)[0]
    ap = metrics.average_precision(np.array(test_labels[i],dtype=np.int16), masks)[0]
    ap_list.append(ap[0])
#print('average precision at iou threshold 0.5 =',np.mean(ap_list))
#ap = metrics.average_precision(test_labels, masks)[0]
print('')
print(f'>>> average precision at iou threshold 0.5 = {np.mean(ap_list):.3f}')



'''
pretrain


self.n_epochs:500, self.batch_size:8,nclasses:3,learning_rate:0.001  tfs
>>> average precision at iou threshold 0.5 = 0.718

self.n_epochs:500, self.batch_size:8,nclasses:3,learning_rate:0.001  not tfs
>>> average precision at iou threshold 0.5 = 0.638


'''
