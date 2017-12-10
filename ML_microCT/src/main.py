import os
import RFLeafSeg
import ImgLoadProcess
import skimage.io as io


#Uncomment if you relocate all RFLeafSeg and other functions into this file....consolidate
#if __name__=="__main__":
selection = "1"
while selection != "7":
    print("1. Image loading and pre-processing")
    print("2. Train model")
    print("3. Examine prediction metrics on training dataset")
    print("4. Predict single slices from test dataset")
    print("5. Predict all slices in 3d microCT stack")
    print("6. Calculate performance metrics")
    print("7. Quit")
    selection = str(input("Select an option (type a number, press enter):\n"))

    if selection=="1": #image loading and pre-processing
        selection2 = "1"
        while selection2 != "5":
            print("1. Load image stacks")
            print("2. Generate binary threshold image, invert, downsample and save as a .tif file for input into local thickness function.")
            print("3. Run local thickness algorithm on downsampled tif file, upsample and save as a .tif file.")
            print("4. Load processed local thickness stack. Match Array dimensions.")
            print("5. Go back")
            selection2 = str(input("Select an option (type a number, press enter):\n"))
            if selection2=="1": #load image stacks
                #filepath = input("Enter filepath to .tif stacks, relative to main.py:\n")
                filepath = "../images/"
                #grid_name = input("Enter filename of grid reconstruction .tif stack:\n")
                grid_name = "gridrec.tif"
                #phase_name = input("Enter filename of phase reconstruction .tif stack:\n")
                phase_name = "phaserec.tif"
                #label_name = input("Enter filename of labeled .tif stack:\n")
                label_name = "label_stack.tif"
                gridrec_stack, phaserec_stack, label_stack = ImgLoadProcess.Load_images(filepath,grid_name,phase_name,label_name)
            elif selection2=="2": #generate binary threshold image, invert, downsample and save
                #Th_grid = input("Enter subjective threshold value for grid reconstruction images, determined in FIJI.\n")
                Th_grid = -12.08
                #Th_phase = input("Enter subjective threshold value for phase reconstruction images, determined in FIJI.\n")
                Th_phase = 0.31
                ImgLoadProcess.Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase)
            elif selection2=="3": #run local thickness, upsample, save
                ImgLoadProcess.localthick_up_save()
            elif selection2=="4": #load processed local thickness stack and match array dimensions
                localthick_stack = io.imread('../images/local_thick_upscale.tif')
                # Match array dimensions to correct for resolution loss due to downsampling when generating local thickness
                gridrec_stack, localthick_stack = RFLeafSeg.match_array_dim(gridrec_stack,localthick_stack)
                phaserec_stack, localthick_stack = RFLeafSeg.match_array_dim(phaserec_stack,localthick_stack)
                label_stack, localthick_stack = RFLeafSeg.match_array_dim_label(label_stack,localthick_stack)
            elif selection2=="5": #go back one step
                print("Going back one step...")
            else:
                print("Not a valid choice.")

    elif selection=="2": #train model
        selection3="1"
        while selection3 != "4":
            print("1. Define image subsets for training and testing.")
            print("2. Display some images from each stack and stack dimensions for QC.")
            print("3. Train model.")
            print("4. Go back")
            selection3 = str(input("Select an option (type a number, press enter):\n"))
            if selection3=="1": #define image subsets for training and testing
                #gridphase_train_slices_subset = input("Enter number of slice(s) for training the model, separated by "," only.")
                gridphase_train_slices_subset = [45]
                #gridphase_test_slices_subset = input("Enter number of slice(s) for testing of the model, separated by "," only.")
                gridphase_test_slices_subset = [245]
                #label_train_slices_subset = input("Enter number of label_stack slice(s) that corresponds to training slices, separated by "," only.")
                label_train_slices_subset = [1]
                #label_test_slices_subset = input("Enter number of label_stack slice(s) that corresponds to testing slices, separated by "," only.")
                label_test_slices_subset = [0]
            elif selection3=="2": #plot some images and stack dimensions
                ImgLoadProcess.displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
            elif selection3=="3": #train model
                rf_transverse,FL_train,FL_test = ImgLoadProcess.train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
            elif selection3=="4": #go back one step
                print("Going back one step...")
            else:
                print("Not a valid choice.")
    elif selection=="3": #examine prediction metrics on training dataset
        # Print out of bag precition accuracy
        hold = "1"
        print('Our OOB prediction of accuracy for is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
        print("Would you like to print feature layer importance?")
        hold = str(input("Enter 1 for yes, or 2 for no."))
        if hold == "1":
            ImgLoadProcess.print_feature_layers(rf_transverse,FL_train,FL_test)
        else:
            print("Okay. Going back.")
    elif selection=="4": #predict single slices
        print("You selected option 4")
        #class_prediction, class_prediction_prob = ImgLoadProcess.predict_testset(rf_transverse,FL_train,FL_test)
    elif selection=="5": #predict all slices in 3d stack
        print("You selected option 5")
    elif selection=="6": #performance metrics
        print("You selected option 6")
    elif selection=="7":
        print("Session ended")
    else:
        print("Not a valid choice.")
