# ImageQC
ImageQC is a tool for the quality control of computed tomography (CT) and magnetic resonance imaging (MRI) modalities.

## Description 
This tool takes some CT or MRI slices with (.dcm, .nii, or .mha) file format as the input. Then uses two Python scripts (QC.py and QCF.py) to generate some criteria for the quality assessment. Finally, the calculated measures which are saved in a .tsv file and .png thumbnail of input images are fed to the bunch of .js scripts to create the user interface (index.html) output. The schematic framework of the tool is as follows:
![imageqc_pipeline](https://user-images.githubusercontent.com/50635618/66402652-3343a600-e9b3-11e9-897e-68ebca4a93bc.png)



| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |
