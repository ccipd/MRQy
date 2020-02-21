# MRQy

MRQy tool is a new quality assessment and evaluation tool for magnetic resonance imaging (MRI) data.

## Description


This tool takes MRI datasets in the file formats (_.dcm_, _.nii_, _.nii.gz_ or _.mha_) as the input. Two Python scripts (_QC.py_ and _QCF.py_) are used to generate several tag and noise/information measurements for quality assessment. These scripts save the calculated measures in a  _.tsv_ file as well generate _.png_ thumbnails for all images in a subject volume. These are then fed to the bunch of _.js_ scripts to create the user interface (_index.html_) output. The schematic framework of the tool is as follows:



![imageqc_pipeline](https://user-images.githubusercontent.com/50635618/66402652-3343a600-e9b3-11e9-897e-68ebca4a93bc.png)


## Prerequisites

The current version of the tool has been tested on the Python vresion 3.7.4. The required Python packages are listed in the following

### General packages:

1. os, 2. argparse, 3. numpy, 4. datetime, 5. time, 6. matplotlib, 7. scipy, 8. itertools, 9. pandas, 10. warnings, and 11. skimage

### Specific packages:

Based on your input files format you may have to install one or more of the following packages: 
1. medpy (for _.mha_, _.nii_, and _.nii.gz_ files) and 2. pydicom (for _.dcm_ files).

2. The t-SNE and UMAP plots work with the sklearn and umap (umap-learn) packeges respectively. 


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
|   __Manufacturer__ | Manufacturer| _ |
|   __MFS__ | Magnetic Field Strength| _ |
|   __VR_x__, __V_y__, __VR_z__ | voxel resolution in x, y, z| _ |
|   __Rows__, __Columns__  | Rows, Columns| _ |
|   __TR__, __TE__  | Repetition Time, Echo Time| _ |
|   __Number__  | number of slice| _ |
|   __Mean__  |  mean of foreground| _ |
|   __Range__  | range of foreground| _ |
|   __Variance__  | range of foreground| _ |
|   __CV__  | foreground coefficient of variation| ![](http://www.sciweavers.org/download/Tex2Img_1570566359.jpg) |
|   __CPP__  | contrast per pixel of foreground| mean(conv2(foreground image, filter)), filter: 3 * 3  with 8 in center and -1 others|
|   __PSNR__  | peak signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566645.jpg), MSE: mean squared error between foreground and median filter 5 * 5  over the foreground   |
|   __SNR1__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566763.jpg)|
|   __SNR2__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566845.jpg), patch: 5 * 5  square patch with center the maximum intensity value of the image    |
|   __SNR3__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566876.jpg)  |
|   __SNR4__  | signal to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570566995.jpg)    |
|   __CNR__  | contrast to noise ratio| ![](http://www.sciweavers.org/download/Tex2Img_1570567041.jpg)| 
|   __CVP__  | patch coefficient of variation| ![](http://www.sciweavers.org/download/Tex2Img_1570567065.jpg)|
|   __EFC__  | entropy focus criterion| Shannon entropy of the foreground image voxel intensites|
|   __FBER__  | foreground-background energy ratio $\ \ $| ![](http://www.sciweavers.org/download/Tex2Img_1570567104.jpg), MED: median|

Reference for the last two measures is  https://mriqc.readthedocs.io/en/stable/.

### User Interface

The following figures show the user interface of the tool (index.html). 

![ui1](https://user-images.githubusercontent.com/50635618/66433517-7a02c180-e9ee-11e9-8949-5849e89ecaf7.png)
![ui1](https://user-images.githubusercontent.com/50635618/75050129-633ea200-5499-11ea-81b6-4140ac6458a3.PNG)
![ui2](https://user-images.githubusercontent.com/50635618/66433515-796a2b00-e9ee-11e9-9ffb-44c130f81533.png)
![ui3](https://user-images.githubusercontent.com/50635618/66433516-796a2b00-e9ee-11e9-9139-07ed57ee342c.png)

### Feedback and usage

Please report and issues, bugfixes, ideas for enhancements via the "Issues" tab

If you do use the tool in your own work, please drop us a line to let us know.
