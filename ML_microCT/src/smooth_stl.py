# Import libraries
import numpy as np
from skimage import io
import vtk
from tqdm import tqdm

def tif_to_stl(filepath,filename,stl_classes):
    # Set input filepath and filename
    # input = '/Users/mattjenkins1/Desktop/Davis_2017/mach_lrn/ML_microCT/results/test1/fullstack_prediction.tif'
    input = filepath+filename
    for i in range(0,len(stl_classes)):
        # Set output filepath and filename
        hold = int(stl_classes[i])
        output = filepath+'class'+str(hold)+'_mesh.stl'
        print('READING TIFF STACK into VTK')
        # Read TIFF file into VTK
        readerVolume = vtk.vtkTIFFReader()
        readerVolume.SetFileName(input)
        readerVolume.Update()
        print('Threshold one class at a time')
        # Threshold material of interest based value at index position (e.g. [2] = veins for this leaf)
        index = np.unique(io.imread(input))[hold]
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(readerVolume.GetOutputPort())
        threshold.ThresholdBetween(index-1,index+1)  # keep only veins
        threshold.ReplaceInOn()
        threshold.SetInValue(0)  # set all values below 400 to 0
        threshold.ReplaceOutOn()
        threshold.SetOutValue(1)  # set all values above 400 to 1
        threshold.Update()
        print('MARHCING CUBES')
        # Use marching cubes to generate STL file from TIFF file
        contour = vtk.vtkDiscreteMarchingCubes()
        contour.SetInputConnection(threshold.GetOutputPort())
        contour.GenerateValues(1, 1, 1)
        contour.Update()
        print('SMOOTHING MESH')
        # Smooth the mesh
        #for possible functions check out http://davis.lbl.gov/Manuals/VTK-4.5/classvtkSmoothPolyDataFilter.html#p9
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(contour.GetOutputPort())
        smooth.SetNumberOfIterations(1000)
        smooth.BoundarySmoothingOn()
        smooth.Update()
        print('Decimate the mesh')
        # Decimate the mesh; this removes vertices and fills holes
        # You might also give this a try
        # Could help for smoothing
        # https://www.vtk.org/doc/nightly/html/classvtkDecimatePro.html
        dec = vtk.vtkDecimatePro()
        dec.SetInputConnection(smooth.GetOutputPort())
        dec.SetTargetReduction(0.2) # Tries to reduce dataset to 80% of it's original size
        dec.PreserveTopologyOn() # Tries to preserve topology
        dec.Update()
        print('WRITING STL FILE')
        # Write STL file
        writer = vtk.vtkSTLWriter()
        # use this line when NOT using decimate
        # writer.SetInputConnection(smooth.GetOutputPort()) # Change "smooth" to "dec", for example, if you want to output the decimated STL file
        # use this line when using decimate
        writer.SetInputConnection(dec.GetOutputPort()) # Change "smooth" to "dec", for example, if you want to output the decimated STL file
        writer.SetFileTypeToBinary()
        writer.SetFileName(output)
        writer.Write()
