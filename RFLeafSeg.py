# Functions for Random Forest 3D Leaf Segmentation Algorithm


# Import libraries and modules
import numpy as np
import skimage.io as io
from skimage.filters import median, sobel, hessian, gabor, gaussian, scharr
import VarianceFilter
from sklearn.preprocessing import LabelEncoder
import scipy as sp
from skimage.segmentation import clear_border
import scipy.ndimage as spim
import scipy.spatial as sptl
from tqdm import tqdm
from numba import jit
from skimage.morphology import cube, ball, disk
from skimage import transform
import cv2


# Variance filter
def winVar(img, wlen):
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen,wlen), borderType=cv2.BORDER_REFLECT)
                       for x in (img, img*img))
    return wsqrmean - wmean*wmean


# Filter parameters; Label encoder setup
disk_size=5
gauss_sd_list = [2,4,8,16,32,64]
gauss_length = 2*len(gauss_sd_list)
hess_range = [4,64]
hess_step = 4
num_feature_layers = 36 # grid and phase recon; plus gaussian blurs; plus hessian filters


# Import label encoder
labenc = LabelEncoder()


# Generate feature layers based on grid/phase stacks and local thickness stack
# Requires five user inputs: 
    # 1) grid recon stack (assumes transverse section)
    # 2) phase recon stack (assumes transverse section)
    # 3) local thickness stack (assumes transverse section)
    # 4) list of sub-slices for training/testing
    # 5) section of interest (i.e. transverse, paradermal, or longitudinal)
def GenerateFL2(gridimg_in, phaseimg_in, localthick_cellvein_in, sub_slices, section): 
    # Define image dimensions (img_dim1, img_dim2), number of slices (num_slices), and rotation parameters (rot_i, rot_j, num_rot)
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    
    # Rotate stacks to correct section view and select subset of slices
    gridimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
    phaseimg_in_rot = np.rot90(phaseimg_in, k=num_rot, axes=(rot_i,rot_j))
    gridimg_in_rot_sub = gridimg_in_rot[sub_slices,:,:]
    phaseimg_in_rot_sub = phaseimg_in_rot[sub_slices,:,:]   
    
    # Define distance from lower/upper image boundary
    dist_edge = np.ones(gridimg_in.shape)
    dist_edge[:,(0,1,2,3,4,gridimg_in.shape[1]-4,gridimg_in.shape[1]-3,gridimg_in.shape[1]-2,gridimg_in.shape[1]-1),:] = 0
    dist_edge = transform.rescale(dist_edge, 0.25)
    dist_edge_FL = spim.distance_transform_edt(dist_edge)
    dist_edge_FL = np.multiply(transform.rescale(dist_edge_FL,4),4)
    
    # Define empty numpy array for feature layers (FL)
    FL = np.empty((len(sub_slices),img_dim1,img_dim2,num_feature_layers), dtype=np.float64)
    
    # Populate FL array with feature layers using custom filters, etc.
    for i in range(0,len(sub_slices)):
        FL[i,:,:,0] = gridimg_in_rot_sub[i,:,:] 
        FL[i,:,:,1] = phaseimg_in_rot_sub[i,:,:]
        FL[i,:,:,2] = gaussian(FL[i,:,:,0],8)
        FL[i,:,:,3] = gaussian(FL[i,:,:,1],8)
        FL[i,:,:,4] = gaussian(FL[i,:,:,0],64)
        FL[i,:,:,5] = gaussian(FL[i,:,:,1],64)
        FL[i,:,:,6] = winVar(FL[i,:,:,0],9)
        FL[i,:,:,7] = winVar(FL[i,:,:,1],9)
        FL[i,:,:,8] = winVar(FL[i,:,:,0],18)
        FL[i,:,:,9] = winVar(FL[i,:,:,1],18)
        FL[i,:,:,10] = winVar(FL[i,:,:,0],36)
        FL[i,:,:,11] = winVar(FL[i,:,:,1],36)
        FL[i,:,:,12] = winVar(FL[i,:,:,0],72)
        FL[i,:,:,13] = winVar(FL[i,:,:,1],72)
        FL[i,:,:,14] = LoadCTStack(localthick_cellvein_in, sub_slices, section)[i,:,:] # > 5%
        FL[i,:,:,15] = dist_edge_FL[i,:,:]
        FL[i,:,:,16] = gaussian(FL[i,:,:,0],4)
        FL[i,:,:,17] = gaussian(FL[i,:,:,1],4)
        FL[i,:,:,18] = gaussian(FL[i,:,:,0],32)
        FL[i,:,:,19] = gaussian(FL[i,:,:,1],32)
        FL[i,:,:,20] = sobel(FL[i,:,:,0])
        FL[i,:,:,21] = sobel(FL[i,:,:,1])
        FL[i,:,:,22] = gaussian(FL[i,:,:,20],8)
        FL[i,:,:,23] = gaussian(FL[i,:,:,21],8)
        FL[i,:,:,24] = gaussian(FL[i,:,:,20],32)
        FL[i,:,:,25] = gaussian(FL[i,:,:,21],32) 
        FL[i,:,:,26] = gaussian(FL[i,:,:,20],64)
        FL[i,:,:,27] = gaussian(FL[i,:,:,21],64)
        FL[i,:,:,28] = gaussian(FL[i,:,:,20],128)
        FL[i,:,:,29] = gaussian(FL[i,:,:,21],128)
        FL[i,:,:,30] = winVar(FL[i,:,:,20],32)
        FL[i,:,:,31] = winVar(FL[i,:,:,21],32)
        FL[i,:,:,32] = winVar(FL[i,:,:,20],64)
        FL[i,:,:,33] = winVar(FL[i,:,:,21],64)
        FL[i,:,:,34] = winVar(FL[i,:,:,20],128)
        FL[i,:,:,35] = winVar(FL[i,:,:,21],128)


    # Collapse training data to two dimensions
    FL_reshape = FL.reshape((-1,FL.shape[3]), order="F")
    return FL_reshape


# Load labeled data stack
def LoadLabelData(gridimg_in,sub_slices,section):
    # Define image dimensions
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    
    # Load training label data
    labelimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
    labelimg_in_rot_sub = labelimg_in_rot[sub_slices,:,:]

    # Collapse label data to a single dimension
    img_label_reshape = labelimg_in_rot_sub.ravel(order="F")
    
    # Encode labels as categorical variable
    img_label_reshape = labenc.fit_transform(img_label_reshape)
    return(img_label_reshape)

# Load generic CT stack
def LoadCTStack(gridimg_in,sub_slices,section):
    # Define image dimensions
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    
    # Load training label data
    labelimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
    labelimg_in_rot_sub = labelimg_in_rot[sub_slices,:,:]

    return(labelimg_in_rot_sub)

# Get dimensions of CT stack
def GetStackDims(labelimg_in,section):
    if(section=="transverse"):
        img_dim1 = labelimg_in.shape[1]
        img_dim2 = labelimg_in.shape[2]
        num_slices = labelimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = labelimg_in.shape[1]
        img_dim2 = labelimg_in.shape[0]
        num_slices = labelimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = labelimg_in.shape[0]
        img_dim2 = labelimg_in.shape[2]
        num_slices = labelimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    return([num_slices,img_dim1,img_dim2])


# Use random forest model to predict entire CT stack on a slice-by-slice basis
def RFPredictCTStack(rf_transverse,gridimg_in, phaseimg_in, localthick_cellvein_in, section):  
    # Define distance from lower/upper image boundary
    dist_edge = np.ones(gridimg_in.shape)
    dist_edge[:,(0,1,2,3,4,gridimg_in.shape[1]-4,gridimg_in.shape[1]-3,gridimg_in.shape[1]-2,gridimg_in.shape[1]-1),:] = 0
    dist_edge = transform.rescale(dist_edge, 0.25)
    dist_edge_FL = spim.distance_transform_edt(dist_edge)
    dist_edge_FL = np.multiply(transform.rescale(dist_edge_FL,4),4)
    
    # Define numpy array for storing class predictions
    RFPredictCTStack_out = np.empty(gridimg_in.shape, dtype=np.float64)
    
    # Define empty numpy array for feature layers (FL)
    FL = np.empty((gridimg_in.shape[1],gridimg_in.shape[2],num_feature_layers), dtype=np.float64)
    
    for j in range(0,gridimg_in.shape[0]):
        # Populate FL array with feature layers using custom filters, etc.
        FL[:,:,0] = gridimg_in[j,:,:]
        FL[:,:,1] = phaseimg_in[j,:,:]
        FL[:,:,2] = gaussian(FL[:,:,0],8)
        FL[:,:,3] = gaussian(FL[:,:,1],8)
        FL[:,:,4] = gaussian(FL[:,:,0],64)
        FL[:,:,5] = gaussian(FL[:,:,1],64)
        FL[:,:,6] = winVar(FL[:,:,0],9)
        FL[:,:,7] = winVar(FL[:,:,1],9)
        FL[:,:,8] = winVar(FL[:,:,0],18)
        FL[:,:,9] = winVar(FL[:,:,1],18)
        FL[:,:,10] = winVar(FL[:,:,0],36)
        FL[:,:,11] = winVar(FL[:,:,1],36)
        FL[:,:,12] = winVar(FL[:,:,0],72)
        FL[:,:,13] = winVar(FL[:,:,1],72)
        FL[:,:,14] = LoadCTStack(localthick_cellvein_in,j,section)[:,:]
        FL[:,:,15] = dist_edge_FL[j,:,:]
        FL[:,:,16] = gaussian(FL[:,:,0],4)
        FL[:,:,17] = gaussian(FL[:,:,1],4)
        FL[:,:,18] = gaussian(FL[:,:,0],32)
        FL[:,:,19] = gaussian(FL[:,:,1],32)
        FL[:,:,20] = sobel(FL[:,:,0])
        FL[:,:,21] = sobel(FL[:,:,1])
        FL[:,:,22] = gaussian(FL[:,:,20],8)
        FL[:,:,23] = gaussian(FL[:,:,21],8)
        FL[:,:,24] = gaussian(FL[:,:,20],32)
        FL[:,:,25] = gaussian(FL[:,:,21],32)
        FL[:,:,26] = gaussian(FL[:,:,20],64)
        FL[:,:,27] = gaussian(FL[:,:,21],64)
        FL[:,:,28] = gaussian(FL[:,:,20],128)
        FL[:,:,29] = gaussian(FL[:,:,21],128)
        FL[:,:,30] = winVar(FL[:,:,20],32)
        FL[:,:,31] = winVar(FL[:,:,21],32)
        FL[:,:,32] = winVar(FL[:,:,20],64)
        FL[:,:,33] = winVar(FL[:,:,21],64)
        FL[:,:,34] = winVar(FL[:,:,20],128)
        FL[:,:,35] = winVar(FL[:,:,21],128)
        # Collapse training data to two dimensions
        FL_reshape = FL.reshape((-1,FL.shape[2]), order="F")
        class_prediction_transverse = rf_transverse.predict(FL_reshape)
        RFPredictCTStack_out[j,:,:] = class_prediction_transverse.reshape((
            gridimg_in.shape[1],
            gridimg_in.shape[2]),
            order="F")
    return(RFPredictCTStack_out)


# Calculate local thickness; from Porespy library
def local_thickness(im):
    if im.ndim == 2:
        from skimage.morphology import square
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt, decimals=0))
    im_new = sp.zeros_like(im, dtype=float)
    for r in tqdm(sizes):
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
        #Trim outer edge of features to remove noise
    if im.ndim == 3:
        im_new = spim.binary_erosion(input=im, structure=ball(1))*im_new
    if im.ndim == 2:
        im_new = spim.binary_erosion(input=im, structure=disc(1))*im_new
    return im_new


# Match array dimensions
def match_array_dim(stack1,stack2):
    if stack1.shape[0] > stack2.shape[0]:
        stack1 = stack1[0:stack2.shape[0],:,:]
    else:
        stack2 = stack2[0:stack1.shape[0],:,:]
    if stack1.shape[1] > stack2.shape[1]:
        stack1 = stack1[:,0:stack2.shape[1],:]
    else:
        stack2 = stack2[:,0:stack1.shape[1],:]
    if stack1.shape[2] > stack2.shape[2]:
        stack1 = stack1[:,:,0:stack2.shape[2]]
    else:
        stack2 = stack2[:,:,0:stack2.shape[2]]

    return stack1, stack2
