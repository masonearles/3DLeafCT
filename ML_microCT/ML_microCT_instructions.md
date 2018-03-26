#  ML_microCT:
## Random Forest machine learning for the quantification and visualization of various plant anatomical components
last edited by: M. Jenkins
03.26.2018

## Instructions:

#### Check out ML_microCT folder

<<<<<<< HEAD
Contents of folder should be exactly:
‘settings’ folder
=======
Contents of folder should be named exactly as:
‘data_settings’ folder
>>>>>>> ffe9d7bb6c92ced285a0dd88f255b3cc2e0afe1a
‘images’ folder
‘src’ folder
‘results’ folder
'jupyter' folder
‘ML_microCT_instructions.md’ file
'pre_processing.md' file

#### Prepare your images and (optional) .txt file instructions

1) Populate image folder with:
grid reconstructed (using FIJI) tiff image stack
phase reconstructed (using FIJI) tiff image stack
labeled images (using FIJI) tiff image stack
<<<<<<< HEAD
2) In 'settings’ folder open ‘input_key.txt’ file
‘input_key.txt’ is a reference key for the architecture or your .txt file of instructions (optional)
**note a .txt file is required to execute program in ‘Read from File Mode’
=======
**note functions within program will add/overwrite other image stacks saved to this folder***
2) In ‘data_settings’ folder open ‘input_key.txt’ file
‘input_key.txt’ is a reference key for the architecture or your .txt file of instructions
**note a .txt file is required to execute program in ‘Read from File Mode’***
>>>>>>> ffe9d7bb6c92ced285a0dd88f255b3cc2e0afe1a
 3) Open a new ‘.txt’ file and enter your instructions, line by line, using key as reference
 4) Save your file a ’.txt’ file in ‘settings’ folder

#### Compile and execute program

1) Open terminal and change present working drive to ‘src’ folder location
2) Enter the following command to compile and run program:

        python MLmicroCT.py

3) Program main menu will be displayed
4) Choose:
- 'Manual Mode' for an interactive mode that asks for user-input throughout
- 'Read from File Mode’ to execute program using preset instructions on .txt file(s)
- ‘Quit’ to exit program

### Manual Mode Instructions:
<<<<<<< HEAD

You'll be automatically prompted to make or designate a folder for data related to current scan.
Folder will exist or be created in 'results/' directory.

1) Choose 1 for ‘Image loading and pre-processing’
**options 1, 2 and 3 must be run once per dataset
**fastest route: 1, 4, 5 (requires previous running of 2 and 3)
- Choose 1 for ‘Load image stacks’ and enter requested information
- Optional: Choose 2 for ‘Generate binary…’ enter requested information, determined subjectively in FIJI (see 'pre_processing.md')
- Optional: Choose 3 for ‘Run local thickness…’
- Choose 4 for ‘Load processed…’
- Choose 5 to ‘Go back’ one step
2) Choose 2 for ‘Train model’
**options 1 and 3 must be run at least once per dataset
**fastest route: 1, 4, 5 (requires previous running of 2 and 3)
- Choose 1 for ‘Define image subsets...’ enter requested information
- Optional: Choose 2 for ‘Display stack dimensions for QC’
- Choose 3 for ‘Train model’ this step will take a few minutes
- Optional: Choose 4 to ‘Save trained model…’
- Optional: Choose 5 to ‘Load trained model…’
- Choose option 6 to ‘Go back’ one step
3) Choose 3 for ‘Examine prediction metrics on training dataset’ to see OOB prediction of accuracy
- Choose yes (1) to see/save feature layer importance to your results folder, or no (2) to skip
4) Choose 4 for ‘Predict single slices from test dataset’
**options 1 and 2 must be run at least once per dataset
**fastest route: 1, 3
=======
1) Choose 1 for ‘Image loading and pre-processing’ menu
**note options 2-3 must be run once per dataset***
- Choose 1 for ‘Load image stacks’
enter requested information
-  Optional: Choose 2 for ‘Generate binary…’
enter requested information, determined subjectively in FIJI
- Optional: Choose 3 for ‘Run local thickness…’
- Choose 4 for ‘Load processed…’
- Choose 5 to ‘Go back’ one step
2) Choose 2 for ‘Train model’ menu
**note options 2, 4, and 5 must be run at least once per dataset***
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
**note options 2, 3 should be run at least once per dataset for quality control***
>>>>>>> ffe9d7bb6c92ced285a0dd88f255b3cc2e0afe1a
- Choose 1 to ‘Predict single slices…’
- Optional: Choose 2 to ‘Generate confusion matrices’ this will also save confusion matrices to your results folder
- Choose 4 to ‘Go back’ one step
5) Choose 5 for ‘Predict all slices in 3d microCT stack’
**option 1 must be run at least once per dataset
**fastest route: 2, 3 (requires previous running of 1 with save)
- Choose 1 to ‘Predict full stack’ this step takes a few minutes, then optionally save this prediction (saving is a good idea)
- Optional: Choose 2 to ' Load existing full stack prediction' enter requested information
- Choose 3 to ‘Go back’ one step
<<<<<<< HEAD
6) Optional: Choose 6 for 'Post-processing'
- Optional: Choose 1 to 'Correct false predictions' then enter requested information, post-process then optionally save (saving is a good idea)
- Optional: Choose 2 to 'Build 3D mesh...' then enter requested information (requires completed and saved stack from 'Correct false predictions' step)
- Optional: Choose 3 for 'Trait measurement'
**This step is incomplete and will do nothing
7) Optional: Choose 7 for ‘Calculate performance metrics’
**requires presence of full stack prediction stack in results folder
Follow instructions for confusion matrix and normalized confusion matrix for full stack prediction and (optionally) post-processed full stack. Will also save absolute precision scores to 'PerformanceMetrics.txt' in your results folder.
8) Choose 8 for ‘Go back’ to go back one step

**These instructions are not complete. Updated instructions are posted periodically.

### Read from File Mode Instructions:
1) Enter exact filename(s) of your .txt file(s), following instructions. File(s) should be in 'settings’ folder.
Program will execute all desired steps.
See ‘results/’ directory for your folder and all relevant results.
Performance metrics can only be accessed in 'manual mode'

**These instructions are not complete. Updated instructions are posted periodically.
=======
6) Choose 6 for ‘Calculate performance metrics’ and following menu
**note this step is incomplete and will do nothing***
7) Choose 7 for ‘Go back’ to go back one step
**These instructions are not complete. Updated instructions for Post-processing are coming soon.***

### Read from File Mode Instructions:
1) Enter number of batches you’d like to run; usually 1 unless you want to process two scans without interruption
2) Enter exact filename of your .txt file with instructions, file should be in ‘data_settings’ folder
this step will be repeated until filename for all batches is defined
program will execute all steps for each batch, saving everything with tag in form ‘B#___’ where ___==rest of file name
See ‘results’ folder for all relevant results
**These instructions are not complete. Updated instructions for Post-processing are coming soon.***
>>>>>>> ffe9d7bb6c92ced285a0dd88f255b3cc2e0afe1a
