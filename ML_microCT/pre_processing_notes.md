# Image Pre-Processing
last edited by: Matt Jenkins
03.20.2018
### Following is for 6 classes (epidermis, IAS, background, palisade mesophyll, spongy mesophyll, vein/vascular bundle)
#### 0 - Before starting:
- Make grid-phase stack
        - Open cropped grid reconstruction stack and cropped phase reconstruction (orient with palisade mesophyll on TOP of image)
            - save all grid reconstruction ROIs to manager for use on phase reconstruction as well
        - With threshold tool (cmd+shift+T), threshold grid and phase reconstructions at desired levels, respectively
        - Use ‘Process->Image calculator’ to add thresholded grid and phase reconstructions
        - Resulting thresholded stack is used for image pre-processing
    - In- In resulting thresholded stack, navigate to number of first train or test slice
        - Duplicate this image TWICE
        - In one of the duplicate, we will call ‘dupA’, highlight all area and fill with white (value = 255) using color picker
            - This is going to be final product, keep this open
        - In other duplicate, ‘dupB’ do steps 1-6
#### 1 - Outline leaf area including epidermis on ‘dubB’; save to ROI manager
- On ‘dupA’ fill this ROI with value 100 using color picker
#### 2 - Outline leaf area NOT including epidermis on ‘dubB’; save to ROI manager
- On ‘dupA’ fill this ROI with value 125 using color picker
#### 3 - Move only abaxial (bottom) toggle positions from ROI in step 2 to bring them to level of lower palisade mesophyll; save to ROI manager
- On ‘dupA’ fill the ROI with value 150 using color picker
#### 4 - Outline veins; save all to ROI
#### 5 - On ‘dupB’ highlight ROI from step 2; then use ‘Edit->Clear outside’
- Use ‘Process->Image calculator’ to add ‘dupA’ to ‘dupB’
#### 6 - On resulting image from step 5 select all ROIs corresponding to veins and fill with value 200 using color picker
#### 7 - Then, invert (cmd+shift+I) resulting image
- On this image the following values will correspond to the following anatomy:
Background = 255 (black)
Intercellular air space (IAS) = 0 (white)
Palisade mesophyll = 150
Spongy mesophyll = 125
Vein/vascular bundle = 200
Epidermis = 100

### *Do not forget to save all ROIs in an organized way that you can reference in the future, allowing trivial use of more or less classes
### **When finished manually segmenting all training and testing slices, must save all .tif images as a .tif stack, noting index of each slice
