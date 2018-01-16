# 3DLeafCT
### Developing a machine learning pipeline for fast segmentation of 3D leaf X-ray microCT images

X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

In the image below, the upper left block is a microCT section through a leaf (~200 micrometers thick). The dark regions are plant tissue and the light regions are intercellular airspace. The challenge is to segment the leaf into background, tissue, airspace, and veins. The particularly tricky aspect is to segment out the vein network, which can take hours to a day to segment by hand in large images with highly reticulated veins. Ultimately, we want to generate 3D segmented volumes as shown below.

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/Nymphaea_Peelback_Panel.jpg" width = 600></a></br>

#### We use these segmented images to calculate novel leaf geometric traits that drive photosynthesis and transpiration, such as CO<sub>2</sub> and H<sub>2</sub>O diffusion path length,

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/3DRendering_Tortuosity.jpg" width = 600></a></br>

#### And for parameterizing abstracted 3D models of CO<sub>2</sub> diffusion and photosynthetic reaction.<br>

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/CO2_Diffusion_Reaction.png" width = 600></a></br>

#### Plus, they provide great tools for educational and public outreach: http://3dleafatlas.org
