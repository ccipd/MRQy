![Picture3](https://user-images.githubusercontent.com/50635618/76575042-70e1b900-6494-11ea-8b39-cb4f189fb13c.png)

## Description


This tool takes MRI datasets in the file formats (_.dcm_, _.nii_, _.nii.gz_ or _.mha_) as the input. \
Two Python scripts (_QC.py_ and _QCF.py_) are used to generate several tags and noise/information measurements for quality assessment. These scripts save the calculated measures in a  _.tsv_ file as well as generate _.png_ thumbnails for all images in a subject volume. These are then fed to the bunch of _.js_ scripts to create the user interface (_index.html_) output. The schematic framework of the tool is as follows.



![Picture1](https://user-images.githubusercontent.com/50635618/76675455-07df6b80-6590-11ea-85f7-13b71a9a1ec3.png)





## Prerequisites

The current version of the tool has been tested on the Python vresion 3.7.4. The required Python packages are listed in the following figure.

![Picture7](https://user-images.githubusercontent.com/50635618/76580525-a2638000-64a6-11ea-8a37-38e95c4693c3.png)


## Running

To test that the code is working fine please try
```
D:\Downloads\MRQy-master>python QC.py --help

```
The output should be 
```
usage: QC.py [-h] output_folder_name [inputdir [inputdir ...]]

positional arguments:
  output_folder_name  the subfolder name on the
                      '...\UserInterface\Data\output_folder_name' directory.
  inputdir            input foldername consists of *.mha (*.nii or *.dcm)
                      files. For example: 'E:\Data\Rectal\input_data_folder'

optional arguments:
  -h, --help          show this help message and exit
  
```
Standard usage is to run ``` QC.py output_folder_name “input directory” ``` i.e. 

```
python QC.py output_folder_name "E:\Data\Rectal\RectalCancer_Multisite\input_data_folder"

```
There is no need to make a subfolder in the Data directory, just specify its name in the command like above code line.\
Every action will be printed in the output console. \
The thumbnail images in the format of _.png_ will be saved on "...\UserInterface\Data\output_folder_name" with its original filename as its subfolder's name. Afterward, double click "index.html" (on e.g. "D:\Downloads\MRQy-master\UserInterface") to open front end user interface, select the respective _results.tsv_ file from the e.g. "D:\Downloads\MRQy-master\UserInterface\Data\output_folder_name" directory.

## Basic Information 

### Measurements

The measures of the MRQy tool are listed in the following table.

![Picture1](https://user-images.githubusercontent.com/50635618/76733243-cb9a3f80-6736-11ea-8100-a1bdb6f60d3f.png)


### User Interface

The following figures show the user interface of the tool (index.html). 

![ui1](https://user-images.githubusercontent.com/50635618/75050129-633ea200-5499-11ea-81b6-4140ac6458a3.PNG)
![ui2](https://user-images.githubusercontent.com/50635618/75050201-8a956f00-5499-11ea-8aa7-19babc98cb70.PNG)


## Feedback and usage

Please report and issues, bugfixes, ideas for enhancements via the "Issues" tab

If you do use the tool in your own work, please drop us a line to let us know.
