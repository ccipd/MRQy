# ImageQC

ImageQC is a tool for the quality control of computed tomography (CT) and magnetic resonance imaging (MRI) modalities.

## Description


This tool takes some CT or MRI slices with (_.dcm_, _.nii_, _.IMA_ or _.mha_) file format as the input. Then uses two Python scripts (_QC.py_ and _QCF.py_) to generate some criteria for the quality assessment. Finally, the calculated measures which are saved in a _.tsv_ file and _.png_ thumbnail of input images are fed to the bunch of _.js_ scripts to create the user interface (_index.html_) output. The schematic framework of the tool is as follows:



![imageqc_pipeline](https://user-images.githubusercontent.com/50635618/66402652-3343a600-e9b3-11e9-897e-68ebca4a93bc.png)


## Prerequisites
The required Python packages are listed in the following

### General packages:

1. os, 2. argparse, 3. numpy, 4. datetime, 5. time, 6. matplotlib, 7. Random, 8. Scipy, and 9. skimage

### Specific packages:

Based on your input files format you can install one of the following packages: 
1. medpy (for _.mha_ files), 2. pydicom (for _.dcm_ files), and 3. nibabel (for _.nii_)



