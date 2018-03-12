import vtk
from skimage import io
import numpy as np
import smooth_stl

def displayPixelvalues(stack):
    pixelVals = np.unique(stack)
    for i in range(0,len(pixelVals)):
        print('Class '+str(i)+' has a pixel value of: '+str(pixelVals[i]))

def main():
    filepath = '../results/test3_10/'
    filename = 'post_processed_fullstack.tif'
    stack = io.imread(filepath+filename)
    stl_classes = []
    print("\nDisplayed below are your dataset's class numbers and corresponding pixel values.")
    print("\nTo select which classes you would like to convert to a 2D mesh (.stl files)\nyou must manually complete the following steps:")
    print("1) Navigate to your custom results folder and open corresponding \nfull stack prediction using ImageJ or FIJI.")
    print("2) Move reticle over image and note pixel values (range 0-255, displayed \non the 'Developer Menu').\nRecord values for all desired pixel classes.\n")
    displayPixelvalues(stack)
    catch = str(raw_input("\nEnter class numbers for which you would like to generate an '.stl' file,\nin order separated by commas:\nExamples: '0,1,2,3' or '2,6'\n"))
    for z in catch.split(','):
        z.strip()
        stl_classes.append(z)
    print(stl_classes)
    smooth_stl.tif_to_stl(filepath,filename,stl_classes)


if __name__ == '__main__':
    main()
