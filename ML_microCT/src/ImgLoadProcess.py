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
from tabulate import tabulate
import pickle
from PIL import Image


#image loading
def Load_images(fp,gr_name,pr_name,ls_name):
    # Set path to tiff stacks
    #filepath = '../images/'
    print("***LOADING IMAGE STACKS***")
    # Read gridrec, phaserec, and label tif stacks
    gridrec_stack = io.imread(fp + gr_name)
    phaserec_stack = io.imread(fp + pr_name)
    #Optional image loading, if you need to rotate images
    #label_stack = np.rollaxis(io.imread(filepath + 'label_stack.tif'),2,0)
    label_stack = io.imread(fp + ls_name)
    #Invert my label_stack, uncomment as needed
    label_stack = invert(label_stack)
    return gridrec_stack, phaserec_stack, label_stack

# Threshold grid and phase images and add the IAS together, invert, downsample and save as .tif stack
def Threshold_GridPhase_invert_down(grid_img, phase_img, Th_grid, Th_phase):
    print("***THRESHOLDING IMAGES***")
    tmp = np.zeros(grid_img.shape)
    tmp[grid_img < Th_grid] = 1
    tmp[grid_img >= Th_grid] = 0
    tmp[phase_img < Th_phase] = 1
    #invert
    tmp_invert = invert(tmp)
    #downsample to 25%
    tmp_invert_ds = transform.rescale(tmp_invert, 0.25)
    print("***SAVING IMAGE STACK***")
    #write as a .tif file un our images folder
    io.imsave('../images/GridPhase_invert_ds.tif',tmp_invert_ds)

# run local thickness, upsample and save as a .tif stack in images folder
def localthick_up_save():
    print("***GENERATING LOCAL THICKNESS STACK***")
    #load thresholded binary downsampled images for local thickness
    GridPhase_invert_ds = io.imread('../images/GridPhase_invert_ds.tif')
    #run local thickness
    local_thick = RFLeafSeg.local_thickness(GridPhase_invert_ds)
    #upsample local_thickness images
    local_thick_upscale = transform.rescale(local_thick, 4, mode='reflect')
    print("***SAVING LOCAL THICKNESS STACK***")
    #write as a .tif file in our images folder
    io.imsave('../images/local_thick_upscale.tif', local_thick_upscale)

def displayImages_displayDims(gr_s,pr_s,ls,lt_s,gp_train,gp_test,label_train,label_test):
    '''
    #plot some images for QC
    for i in [label_test,label_train]:
        io.imshow(ls[i,:,:])
        io.show()
    for i in [gp_train,gp_test]:
        io.imshow(gr_s[i,:,:], cmap='gray')
        io.show()
    for i in [gp_train,gp_test]:
        io.imshow(pr_s[i,:,:], cmap='gray')
        io.show()
    for i in [gp_train,gp_test]:
        io.imshow(lt_s[i,:,:])
        io.show()
    '''
    #check shapes of stacks to ensure they match
    print(gr_s.shape)
    print(pr_s.shape)
    print(ls.shape)
    print(lt_s.shape)

def train_model(gr_s,pr_s,ls,lt_s,gp_train,gp_test,label_train,label_test):
    print("***GENERATING FEATURE LAYERS***")
    #generate training and testing feature layer array
    FL_train_transverse = RFLeafSeg.GenerateFL2(gr_s, pr_s, lt_s, gp_train, "transverse")
    FL_test_transverse = RFLeafSeg.GenerateFL2(gr_s, pr_s, lt_s, gp_test, "transverse")
    # Load and encode label image vectors
    Label_train = RFLeafSeg.LoadLabelData(ls, label_train, "transverse")
    Label_test = RFLeafSeg.LoadLabelData(ls, label_test, "transverse")
    print("***TRAINING MODEL***")
    # Define Random Forest classifier parameters and fit model
    rf_trans = RandomForestClassifier(n_estimators=50, verbose=True, oob_score=True, n_jobs=4, warm_start=False) #, class_weight="balanced")
    rf_trans = rf_trans.fit(FL_train_transverse, Label_train)

    return rf_trans,FL_train_transverse,FL_test_transverse, Label_train, Label_test

def print_feature_layers(rf_t):
    # Print feature layer importance
    # See RFLeafSeg module for corresponding feature layer types
    feature_layers = range(0,len(rf_t.feature_importances_))
    for fl, imp in zip(feature_layers, rf_t.feature_importances_):
        print('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp))

def predict_testset(rf_t,FL_test): #predict single slices from dataset
    # Make prediction on test set
    print("***GENERATING PREDICTED STACK***")
    class_prediction = rf_t.predict(FL_test)
    class_prediction_prob = rf_t.predict_proba(FL_test)

    return class_prediction, class_prediction_prob

def make_conf_matrix(L_test,class_p):
    # Generate confusion matrix for transverse section
    df = pd.crosstab(L_test, class_p, rownames=['Actual'], colnames=['Predicted'])
    print(tabulate(df, headers='keys', tablefmt='pqsl'))

def make_normconf_matrix(L_test,class_p):
    # Generate normalized confusion matrix for transverse section
    df = pd.crosstab(L_test, class_p, rownames=['Actual'], colnames=['Predicted'], normalize='index')
    print(tabulate(df, headers='keys', tablefmt='pqsl'))

def save_trainmodel(rf_t,FL_train,FL_test,Label_train,Label_test):
    #Save model to disk; This can be a pretty large file -- ~2 Gb
    print("***SAVING TRAINED MODEL***")
    filename = '../images/RF_model.sav'
    pickle.dump(rf_t, open(filename, 'wb'))
    print("***SAVING FEATURE LAYER ARRAYS***")
    #save training and testing feature layer array
    io.imsave('../images/FL_train.tif',FL_train)
    io.imsave('../images/FL_test.tif',FL_test)
    print("***SAVING LABEL IMAGE VECTORS***")
    #save label image vectors
    io.imsave('../images/Label_train.tif',Label_train)
    io.imsave('../images/Label_test.tif',Label_test)

def load_trainmodel():
    print("***LOADING TRAINED MODEL***")
    #load the model from disk
    filename = '../images/RF_model.sav'
    rf = pickle.load(open(filename, 'rb'))
    print("***LOADING FEATURE LAYER ARRAYS***")
    FL_tr = io.imread('../images/FL_train.tif')
    FL_te = io.imread('../images/FL_test.tif')
    print("***LOADING LABEL IMAGE VECTORS***")
    Label_tr = io.imread('../images/Label_train.tif')
    Label_te = io.imread('../images/Label_test.tif')
    return rf,FL_tr,FL_te,Label_tr,Label_te

def reshape_arrays(class_prediction_prob,class_prediction,Label_test,FL_test,label_stack):
    print("***RESHAPING ARRAYS***")
    # Reshape arrays for plotting images of class probabilities, predicted classes, observed classes, and feature layer of interest
    prediction_prob_imgs = class_prediction_prob.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2],
        4),
        order="F")
    prediction_imgs = class_prediction.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2]),
        order="F")
    observed_imgs = Label_test.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2]),
        order="F")
    FL_imgs = FL_test.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2],
        36),
        order="F")
    return prediction_prob_imgs,prediction_imgs,observed_imgs,FL_imgs

def check_images(prediction_prob_imgs,prediction_imgs,observed_imgs,FL_imgs,phaserec_stack):
    # Plot images of class probabilities, predicted classes, observed classes, and feature layer of interest
    for i in range(0,prediction_imgs.shape[0]):
#        img1 = Image.open(prediction_prob_imgs[i,:,:,1], cmap="RdYlBu")
        img1 = prediction_prob_imgs[i,:,:,1]
        img1 = Image.fromarray(img1)
        img1.show()
        # io.imshow(observed_imgs[i,:,:])
        # io.show()
        # io.imshow(prediction_imgs[i,:,:])
        # io.show()
        # io.imshow(phaserec_stack[260,:,:], cmap="gray")
        # io.show()
        # io.imshow(FL_imgs[0,:,:,26], cmap="gray")
        # io.show()
