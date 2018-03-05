# RandomForestMicroCT
### Random forest segmentation for 3D microCT images
X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

## Changes on this Branch:
### Command Line Version:
Once installed, ML_microCT can be run from the command line. This version includes both a 'manual' mode with user input at multiple points throughout image segmentation process as well as a file I/O method that runs the entire segmentation process without interruption.

Introduced 'batch run' capability for file I/O method that allows one to segment multiple stacks, using multiple instruction files, without any interruption. Note, batch run will always be slightly risky as the program may abort at any point, so this is not necessarily a recommended method. Improvements forthcoming.

Now, every time a new scan is segmented all results and data related to this scan will be exported to a new folder with a user-generated name.

#### See 'ML_microCT_inst.rtf' for detailed instructions on running from command line:

### Post-processing Beta:
Post-processing now integrated with 'manual' mode only in 'MLmicroCT.py' script. Removes falsely predicted epidermis, false IAS and mesophyll predictions that fall outside the epidermis, false background predictions that fall inside the epidermis; still relies on hardcoded values for epidermis, background, IAS and palisade/spongy mesophyll--interactives are in the works. Improvements forthcoming, including post-processing integration with 'batch-mode'.

Once you have a fully post-processed stack, you can generate a 2D mesh in .stl format. Then smooth this 2D surface and visualize segmented classes as separate, complementary volumes in 3D space. Currently requires 'smooth.py' file and changing lots of hardcoded values.
