#  ML_microCT:
## Random Forest machine learning for the quanitification and visualization of various plant anatomy
last edited by: M. Jenkins
03.20.2018

## Instructions:

#### Check out ML_microCT folder

Contents of folder should be exactly:
‘data_settings’ folder
‘images’ folder
‘src’ folder
‘results’ folder
‘ML_microCT_instr.rtf’ file

#### Prepare your images and (optional) .txt file instructions

1) Populate image folder with:
grid reconstructed (using FIJI) tiff image stack
phase reconstructed (using FIJI) tiff image stack
labeled images (using FIJI) tiff image stack
**note functions within program will add/overwrite other image stacks saved to this folder
2) In ‘data_settings’ folder open ‘input_key.txt’ file
‘input_key.txt’ is a reference key for the architecture or your .txt file of instructions
**note a .txt file is required to execute program in ‘Read from File Mode’
 3) Open a new ‘.txt’ file and enter your instructions, line by line, using key as reference
 4) Save your file a ’.txt’ file in ‘data_settings’ folder

#### Compile and execute program

1) open terminal and change present working drive to ‘src’ folder location
2) enter the following command to compile and run program:

        python MLmicroCT.py

3) Program main menu will be displayed
4) Choose:
- 'Manual Mode' for an interactive mode that asks for user-input throughout
- 'Read from File Mode’ to execute program using preset instructions on a .txt file
- ‘Quit’ to quit program

### Manual Mode Instructions:
1) Choose 1 for ‘Image loading and pre-processing’ menu
**note options 2-3 must be run once per dataset
- Choose 1 for ‘Load image stacks’
enter requested information
-  Optional: Choose 2 for ‘Generate binary…’
enter requested information, determined subjectively in FIJI
- Optional: Choose 3 for ‘Run local thickness…’
- Choose 4 for ‘Load processed…’
- Choose 5 to ‘Go back’ one step
2) Choose 2 for ‘Train model’ menu
**note options 2, 4, and 5 must be run at least once per dataset
- Choose 1 for ‘Define image…’
enter requested information
- Optional: Choose 2 for ‘Display some images…’
- Choose 3 for ‘Train model’
this step will take a few minutes
- Optional: Choose 4 to ‘Save trained model…’
- Optional: Choose 5 to ‘Load trained model…’
- Choose option 6 to ‘Go back’ one step
3) Choose 3 for ‘Examine prediction metrics on training dataset’, this step may be repeated
OOB prediction accuracy is printed
- Choose yes or no to see or not see feature layer importance
4) Choose 4 for ‘Predict single slices from test dataset’ and following menu
**note options 2, 3 should be run at least once per dataset for quality control
- Choose 1 to ‘Predict single slices…’
- Optional: Choose 2 to ‘Generate confusion matrices’
- Optional: Choose 3 to ‘Plot images’
- Choose 4 to ‘Go back’ one step
5) Choose 5 for ‘Predict all slices in 3d microCT stack’ and following menu
- Choose 1 to ‘Predict full stack’
this step takes a few minutes
- Choose 2 to ‘Write stack as .tif file’
- Choose 3 to ‘Go back’ one step
6) Choose 6 for ‘Calculate performance metrics’ and following menu
**note this step is incomplete and will do nothing
7) Choose 7 for ‘Go back’ to go back one step
**These instructions are not complete. Updated instructions for Post-processing are coming soon.

### Read from File Mode Instructions:
1) Enter number of batches you’d like to run; usually 1 unless you want to process two scans without interruption
2) Enter exact filename of your .txt file with instructions, file should be in ‘data_settings’ folder
this step will be repeated until filename for all batches is defined
program will execute all steps for each batch, saving everything with tag in form ‘B#___’ where ___==rest of file name
See ‘results’ folder for all relevant results
**These instructions are not complete. Updated instructions for Post-processing are coming soon.
