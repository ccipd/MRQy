# ImageQC

ImageQC is a tool for the quality control of computed tomography (CT) and magnetic resonance imaging (MRI) modalities.

## Description


This tool takes some CT or MRI slices with (_.dcm_, _.nii_, _.ima_ or _.mha_) file format as the input. Then uses two Python scripts (_QC.py_ and _QCF.py_) to generate some criteria for the quality assessment. Finally, the calculated measures which are saved in a  _.tsv_ file and  also the _.png_ thumbnail of input images are fed to the bunch of _.js_ scripts to create the user interface (_index.html_) output. The schematic framework of the tool is as follows:



![imageqc_pipeline](https://user-images.githubusercontent.com/50635618/66402652-3343a600-e9b3-11e9-897e-68ebca4a93bc.png)


## Prerequisites

The required Python packages are listed in the following

### General packages:

1. os, 2. argparse, 3. numpy, 4. datetime, 5. time, 6. matplotlib, 7. Scipy, 8. skimage, and 9. cv2

### Specific packages:

Based on your input files format you can install one of the following packages: 
1. medpy (for _.mha_ files), 2. pydicom (for _.dcm_ files), and 3. nibabel (for _.nii_ files)


## Running

To test that the code is working fine please try
```
E:\Python_Codes\Github>python QC.py --help

```
The output should be 
```
usage: QC.py [-h] [-o OUTDIR] [inputdir [inputdir ...]]

positional arguments:
  inputdir              input foldername consists of *.mha files. For example:
                        'E:\Data\Rectal\RectalCancer_Multisite\UH'

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        a subfolder on the Data directory of the UserInterface
                        i.e. E:\Python_Codes\Github\UserInterface\Data\Output
```
Standard usage is to run ``` QC.py –o output directory “input directory” ``` i.e. 

```
python QC.py -o E:\Python_Codes\Github\UserInterface\Data\Output "E:\Data\Rectal\RectalCancer_Multisite\UH"

```
There is no need to make a subfolder in the Data directory, just specify its name in the command like above code line.
Every action will be print in the output console. The thumbnail images in the format of _.png_ will be saved on E:\Python_Codes\Github\UserInterface\Data with its original filename as its subfolder's name. Afterward, double click index.html (on e.g. E:\Python_Codes\Github\UserInterface) to open front end user interface, select the respective results.tsv file from the E:\Python_Codes\Github\UserInterface\Data\Output directory.

## Basic Information 

### Measurements

The measures of the ImageQC tool are listed in the following table

| Measure |  Description  |  Formula |
|---------|------------| ---------------------|
|   __VR_x__, __V_y__, __VR_z__ | voxel resolution in x, y, z| _ |
|   __Rows__, __Columns__  | Rows, Columns| _ |
|   __Number__  | number of slice| _ |
|   __Mean__  |  mean of foreground| _ |
|   __Range__  | range of foreground| _ |
|   __CV__  | foreground coefficient of variation| ![](http://www.sciweavers.org/download/Tex2Img_1570566359.jpg) |
|   __CPP__  | contrast per pixel of foreground| mean(conv2(foreground image, filter)), filter: 3 * 3  with 8 in center and -1 others|
|   __PSNR__  | peak signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566645.jpg), MSE: mean squared error between foreground and median filter 5 * 5  over the foreground   |
|   __SNR1__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566763.jpg)|
|   __SNR2__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566845.jpg), patch: 5 * 5  square patch with center the maximum intensity value of the image    |
|   __SNR3__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566876.jpg)  |
|   __SNR4__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566995.jpg)    |
|   __SNR5__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570567015.jpg)   |
|   __CNR__  | contrast to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570567041.jpg)| 
|   __CVP__  | coefficient of variation of patch| ![](http://www.sciweavers.org/download/Tex2Img_1570567065.jpg)|
|   __EFC__  | entropy focus criterion| Shannon entropy of the foreground image voxel intensites|
|   __FBER__  | foreground-background energy ratio $\ \ $| ![](http://www.sciweavers.org/download/Tex2Img_1570567104.jpg), MED: median|

Reference for the last two measures is  https://mriqc.readthedocs.io/en/stable/.

### User Interface

The following figures show the user interface of the tool (index.html). 

![ui1](https://user-images.githubusercontent.com/50635618/66433517-7a02c180-e9ee-11e9-8949-5849e89ecaf7.png)
![ui2](https://user-images.githubusercontent.com/50635618/66433515-796a2b00-e9ee-11e9-9ffb-44c130f81533.png)
![ui3](https://user-images.githubusercontent.com/50635618/66433516-796a2b00-e9ee-11e9-9139-07ed57ee342c.png)

