
# coding: utf-8

# # Random forest machine learning classification of leaf microCT data
# #### Semi-automated segmentation of 3D microCT data into multiple classes: (1) background, (2) veins, (3) mesophyll cells, (4) bundle sheath tissue, and (5) intercellular airspace
# 
# #### Open questions / To-Do:
# * Incorporate thresholding based on two user provided ranges for the grid and phase reconstructed images
# * Try adding more convolution filters to GenerateFL2 function.  E.g. difference of gaussians, membrane projection filter, etc.
# * Try automated segmentation of palisade and mesophyll tissue
# * Test on other tissue types, e.g. epidermal cells, when they are clearly visible in scans. E.g. in conifers and cycads
# * What happens when there is embolism in the vascular tissue?
# * Develop module for quantifying leaf traits (e.g. mesophyll surface area, porosity, thickness, leaf volume, etc.)
# 
# #### Last edited by: J. Mason Earles 
# #### Date: 12/05/2017

# In[1]:

# Import libraries
import os
import sklearn as skl
import skimage.io as io
from skimage import img_as_int, img_as_ubyte, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import RFLeafSeg # reload(RFLeafSeg)
from scipy import misc
from skimage.util import invert
from skimage import transform
from sklearn.metrics import confusion_matrix


# ### Image Loading and Pre-processing

# In[2]:

# Set path to tiff stacks
filepath = '../forBeth/'


# In[3]:

# Read gridrec, phaserec, and label tif stacks
gridrec_stack = io.imread(filepath + 'V_champ_gridrec.tif')
phaserec_stack = io.imread(filepath + 'V_champ_phaserec.tif')
label_stack = np.rollaxis(io.imread(filepath + 'label_stack_wBS.tif'),2,0)


# In[4]:

# # Generate/load local thickness stack
# # Uncomment as needed

# # Generate binary thresholded image for input to local thickness function
# # Set grid and phase threshold values for segmenting low and high absorption regions
# # I typically use Fiji to subjectively and visually determine the 'best' value for each
# Th_grid = -22.92 # grid rec threshold value
# Th_phase = 0.37 # phase rec threshold value
# GridPhase_Bin = RFLeafSeg.Threshold_GridPhase(gridrec_stack, phaserec_stack,Th_grid,Th_phase)

# # Invert image
# GridPhase_Bin_invert = invert(GridPhase_Bin)

# # Downsample to 25%
# GridPhase_Bin_invert = transform.rescale(GridPhase_Bin_invert, 0.25)

# local_thick = RFLeafSeg.local_thickness(GridPhase_Bin_invert)
# local_thick_upscale = transform.rescale(local_thick, 4, mode='reflect')

# # Write as a tif file
# io.imsave('local_thick_upscale.tif', local_thick_upscale)

# Load local thickness stack, if already generated
LocalThickness_CellVeins = io.imread(filepath + 'local_thick_upscale.tif')


# In[5]:

# Match array dimensions to correct for resolution loss due to downsampling when generating local thickness
gridrec_stack, local_thick_upscale = RFLeafSeg.match_array_dim(gridrec_stack,LocalThickness_CellVeins)
phaserec_stack, local_thick_upscale = RFLeafSeg.match_array_dim(phaserec_stack,LocalThickness_CellVeins)
label_stack = RFLeafSeg.match_array_dim(label_stack,LocalThickness_CellVeins)[0]


# In[24]:

# Plot thresholded image for QC
# %matplotlib inline
# fig = plt.figure(figsize = (15,15))
# io.imshow(GridPhase_Bin[100,:,:])


# In[25]:

# Plot some of the images to make sure everything looks correct
for i in range(0,4):
    io.imshow(label_stack[i,:,:])
    io.show()

for i in [55,99,160,248]:    
    io.imshow(LocalThickness_CellVeins[i,:,:])
    io.show()
    
for i in [55,99,160,248]:    
    io.imshow(phaserec_stack[i,:,:], cmap='gray')
    io.show()

print(label_stack.shape)
print(gridrec_stack.shape)


# In[22]:

# Check shapes of stacks to make ensure that they match
print(gridrec_stack.shape)
print(phaserec_stack.shape)
print(label_stack.shape)
print(LocalThickness_CellVeins.shape)


# In[6]:

# Define image subsets for training and testing
gridphase_train_slices_subset = [99,248] # 99 for training of Vitis champ.
gridphase_test_slices_subset = [55,160] # 55, 160 and 248 for testing of Vitis champ.
label_train_slices_subset = [1,3] # corresponding slice from the label stack
label_test_slices_subset = [0,2] # corresponding slice from the label stack


# In[7]:

# Generate training and testing feature layer array
FL_train_transverse = RFLeafSeg.GenerateFL2(gridrec_stack, phaserec_stack, LocalThickness_CellVeins, gridphase_train_slices_subset, "transverse")
FL_test_transverse = RFLeafSeg.GenerateFL2(gridrec_stack, phaserec_stack, LocalThickness_CellVeins, gridphase_test_slices_subset, "transverse")


# In[8]:

# Load and encode label image vectors
Label_train_transverse = RFLeafSeg.LoadLabelData(label_stack, label_train_slices_subset, "transverse")
Label_test_transverse = RFLeafSeg.LoadLabelData(label_stack, label_test_slices_subset, "transverse")


# In[9]:

# Check the dimensions of the feature array and label vector to ensure that they are the same dimensions
print(FL_train_transverse.shape)
print(Label_train_transverse.shape)


# ### Train model

# In[13]:

# Define Random Forest classifier parameters and fit model
rf_transverse = RandomForestClassifier(n_estimators=50, verbose=True, oob_score=True, n_jobs=4, warm_start=False) #, class_weight="balanced")
rf_transverse = rf_transverse.fit(FL_train_transverse, Label_train_transverse)


# In[ ]:

# Save model to disk # This can be a pretty large file -- ~2 Gb
# import pickle
# filename = 'RF_Vitus_champ_model.sav'
# pickle.dump(rf_transverse, open(filename, 'wb'))

#load the model from disk
#rf = pickle.load(open(filename, 'rb'))


# ### Examine prediction metrics on training dataset

# In[14]:

# Print out of bag precition accuracy
print('Our out-of-bag (OOB) prediction of accuracy for is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))


# In[15]:

# Print feature layer importance
# See RFLeafSeg module for corresponding feature layer types
feature_layers = range(0,len(rf_transverse.feature_importances_))
for fl, imp in zip(feature_layers, rf_transverse.feature_importances_):
    print('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp))
#print('none')


# ### Predict single slices from test dataset

# In[16]:

# Make prediction on test set
class_prediction_transverse = rf_transverse.predict(FL_test_transverse)
class_prediction_transverse_prob = rf_transverse.predict_proba(FL_test_transverse)


# In[34]:

# Generate confusion matrix for transverse section
cf_mat1 = pd.crosstab(Label_test_transverse, class_prediction_transverse, rownames=['Actual'], colnames=['Predicted'])


# In[30]:

# Generate normalized confusion matrix for transverse section
pd.crosstab(Label_test_transverse, class_prediction_transverse, rownames=['Actual'], colnames=['Predicted'], normalize='index')


# In[35]:

class_prediction_transverse2 = np.copy(class_prediction_transverse)
class_prediction_transverse2[class_prediction_transverse==1] = 2
#class_prediction_transverse2[class_prediction_transverse2==3] = 4
Label_test_transverse2 = np.copy(Label_test_transverse)
Label_test_transverse2[Label_test_transverse==1] = 2
#Label_test_transverse2[Label_test_transverse2==3] = 4
pd.crosstab(Label_test_transverse2, class_prediction_transverse2, rownames=['Actual'], colnames=['Predicted'], normalize='index')
cf_mat2 = pd.crosstab(Label_test_transverse2, class_prediction_transverse2, rownames=['Actual'], colnames=['Predicted'])
print(cf_mat2)


# In[51]:

# Total accuracy
print(float(np.diag(cf_mat2).sum())/float(cf_mat2.sum().sum()))

# Class precision
print(np.diag(cf_mat2)/np.sum(cf_mat2,1), "precision")

# Class recall
print(np.diag(cf_mat2)/np.sum(cf_mat2,0), "recall")


# In[21]:

get_ipython().magic(u'matplotlib inline')
# fig = plt.figure(figsize = (15,15))
# io.imshow(GridPhase_Bin[100,:,:])class_prediction_transverse_prob.shape


# In[19]:

# Reshape arrays for plotting images of class probabilities, predicted classes, observed classes, and feature layer of interest
prediction_transverse_prob_imgs = class_prediction_transverse_prob.reshape((
    -1,
    label_stack.shape[1],
    label_stack.shape[2],
    5),
    order="F")
prediction_transverse_imgs = class_prediction_transverse.reshape((
    -1,
    label_stack.shape[1],
    label_stack.shape[2]),
    order="F")
observed_transverse_imgs = Label_test_transverse.reshape((
    -1,
    label_stack.shape[1],
    label_stack.shape[2]),
    order="F")
FL_transverse_imgs = FL_test_transverse.reshape((
    -1,
    label_stack.shape[1],
    label_stack.shape[2],
    36),
    order="F")


# In[20]:

# Plot images of class probabilities, predicted classes, observed classes, and feature layer of interest
get_ipython().magic(u'matplotlib inline')
for i in range(0,prediction_transverse_imgs.shape[2]):
    io.imshow(prediction_transverse_prob_imgs[i,:,:,3], cmap="RdYlBu")
    io.show()
    io.imshow(observed_transverse_imgs[i,:,:])
    io.show()
    io.imshow(prediction_transverse_imgs[i,:,:])
    io.show()
    io.imshow(phaserec_stack[260,:,:], cmap="gray")
    io.show()
    io.imshow(FL_transverse_imgs[0,:,:,26], cmap="gray")
    io.show()b


# ### Predict all slices in 3D microCT stack

# In[ ]:

# Predict all slices in 3D microCT stack
RFPredictCTStack_out = RFLeafSeg.RFPredictCTStack(rf_transverse,gridrec_stack, phaserec_stack, LocalThickness_CellVeins,"transverse")


# ### Calculate performance metrics

# In[28]:

# Performance metrics
test_slices = (55,160)
label_slices = (0,2)

# Generate absolute confusion matrix
confusion_matrix = pd.crosstab(RFPredictCTStack_out[test_slices,:,:].ravel(order="F"),
                               label_stack[label_slices,:,:].ravel(order="F"),
                               rownames=['Actual'], colnames=['Predicted'])

# Generate normalized confusion matrix
confusion_matrix_norm = pd.crosstab(RFPredictCTStack_out[test_slices,:,:].ravel(order="F"),
                               label_stack[label_slices,:,:].ravel(order="F"),
                               rownames=['Actual'], colnames=['Predicted'])

# Total accuracy
print(np.diag(confusion_matrix).sum()/RFPredictCTStack_out[test_slices,:,:].sum())

# Class precision
print(np.diag(confusion_matrix)/np.sum(confusion_matrix,1), "precision")

# Class recall
print(np.diag(confusion_matrix)/np.sum(confusion_matrix,0), "recall")


# ### Save segmented and classified stack as TIFF file

# In[ ]:

# Save classified stack
io.imsave('Vitis_champ_Predicted_wBS.tif',img_as_int(RFPredictCTStack_out/6))


# In[ ]:



