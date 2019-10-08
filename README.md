# ImageQC

ImageQC is a tool for the quality control of computed tomography (CT) and magnetic resonance imaging (MRI) modalities.

## Description

This tool takes some CT or MRI slices with (_.dcm_, _.nii_, or _.mha_) file format as the input. Then uses two Python scripts (_QC.py_ and _QCF.py_) to generate some criteria for the quality assessment. Finally, the calculated measures which are saved in a _.tsv_ file and _.png_ thumbnail of input images are fed to the bunch of _.js_ scripts to create the user interface (_index.html_) output. The schematic framework of the tool is as follows:

![imageqc_pipeline](https://user-images.githubusercontent.com/50635618/66402652-3343a600-e9b3-11e9-897e-68ebca4a93bc.png)



| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |
