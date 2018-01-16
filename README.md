# 3DLeafCT
### Developing a machine learning pipeline for fast segmentation of 3D leaf X-ray microCT images

X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

In the image below, the upper left block is a microCT section through a leaf (~200 micrometers thick). The dark regions are plant tissue and the light regions are intercellular airspace. The challenge is to segment the leaf into background, tissue, airspace, and veins. The particularly tricky aspect is to segment out the vein network, which can take hours to a day by hand in large images with highly reticulated veins. 

Ultimately, we use these segmented images to generate 3D volumes for geometric measurement and model simulation as shown below.

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/Nymphaea_Peelback_Panel.jpg" width = 600></a></br>

For example, I have made novel measurements of leaf geometric traits that underlie photosynthesis and transpiration, such as CO<sub>2</sub> and H<sub>2</sub>O diffusion path length, tortuosity, and intercellular airspace connectivity.

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/3DRendering_Tortuosity.jpg" width = 600></a></br>

Moreover, I have used these 3D geometric traits to parameterize 3D finite element models of CO<sub>2</sub> diffusion and photosynthetic reaction.<br>

<br><a href="url"><img src="https://github.com/masonearles/3DLeafCT/blob/master/imgs_readme/CO2_Diffusion_Reaction.png" width = 650></a></br>

3D models of leaves also provide great tools for educational and public outreach: http://3dleafatlas.org
