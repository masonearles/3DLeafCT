# 3DLeafCT
### Developing a machine learning pipeline for fast segmentation of 3D leaf microCT images

X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

<br> ![Alt text](imgs_readme/Nymphaea_Peelback_Panel.jpg?raw=true "Nymphaea Peelback Panel") <br>

### We use these segmented images to calculate novel leaf geometric traits that drive photosynthesis and transpiration, such as CO<sub>2</sub> and H<sub>2</sub>O diffusion path length,

<br> ![Alt text](imgs_readme/3DRendering_Tortuosity.jpg?raw=true "3D Rendering Tortuosity") <br>

### And for parameterizing abstracted 3D models of CO<sub>2</sub> diffusion and photosynthetic reaction (color corresponds with CO<sub>2</sub> concentration throughout the leaf).<br>

<br> ![Alt text](imgs_readme/CO2_Simulation.gif?raw=true "CO2 Simulation") <br>

### Plus, they provide great tools for educational and public outreach: http://3dleafatlas.org
