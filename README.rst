.. figure:: https://user-images.githubusercontent.com/50635618/77593997-b1492a00-6ecb-11ea-939c-c8962f371e5a.png
   :alt: Picture1

   Picture1

Front-end View
--------------

.. figure:: https://user-images.githubusercontent.com/50635618/77496601-b6519f00-6e21-11ea-8f52-16f33d4c66cc.gif
   :alt: gbm_mrqy

   gbm_mrqy

Backend View
------------

.. figure:: https://user-images.githubusercontent.com/50635618/77506445-43095680-6e3c-11ea-9376-7be6f7cdc5d8.gif
   :alt: gbm_mrqy

   gbm_mrqy

Table of Contents
=================

-  `Description <#description>`__
-  `Prerequisites <#prerequisites>`__
-  `Running <#running>`__
-  `Basic Information <#basic-information>`__

   -  `Measurements <#measurements>`__
   -  `User Interface <#user-interface>`__

-  `Feedback and usage <#feedback-and-usage>`__

Description
-----------

| This tool takes MRI datasets in the file formats (*.dcm*, *.nii*,
  *.nii.gz* or *.mha*) as the input.
| Two Python scripts (*QC.py* and *QCF.py*) are used to generate several
  tags and noise/information measurements for quality assessment. These
  scripts save the calculated measures in a *.tsv* file as well as
  generate *.png* thumbnails for all images in a subject volume. These
  are then fed to *.js* scripts to create the user interface
  (*index.html*) output. A schematic illustrating the framework of the
  tool is as follows.

.. figure:: https://user-images.githubusercontent.com/50635618/76675455-07df6b80-6590-11ea-85f7-13b71a9a1ec3.png
   :alt: Picture1

   Picture1

Prerequisites
-------------

| The current version of the tool has been tested on the Python 3.6+
| You must have
  `pipenv <https://pipenv-fork.readthedocs.io/en/latest/basics.html>`__
  installed on your environment to run MRQy locally. It will pull all
  the dependencies listed in the diagram.

.. figure:: https://user-images.githubusercontent.com/50635618/76580525-a2638000-64a6-11ea-8a37-38e95c4693c3.png
   :alt: Picture7

   Picture7

You can also likely install the python requirements using something
like:

pip3 install -r requirements.txt

Running
-------

For local development, test that the code is functional

::

   MRQy % pipenv shell
   (mrqy) MRQy% pipenv install .
   (mrqy) MRQy% python -m mrqy.QC --help

The output should be

::

   usage: QC.py [-h] output_folder_name [inputdir [inputdir ...]]

   positional arguments:
     output_folder_name  the subfolder name on the
                         '...\UserInterface\Data\output_folder_name' directory.
     inputdir            input foldername consists of *.mha (*.nii or *.dcm)
                         files. For example: 'E:\Data\Rectal\input_data_folder'

   optional arguments:
     -h, --help          show this help message and exit
     

Standard usage is to run ``QC.py output_folder_name “input directory”``
i.e. 

::

   python QC.py output_folder_name "E:\Data\Rectal\RectalCancer_Multisite\input_data_folder"

| There is no need to make a subfolder in the Data directory, just
  specify its name in the command as in the above code.
| Every action will be printed in the output console.
| The thumbnail images in the format of *.png* will be saved in
  “…:raw-latex:`\UserInterface`:raw-latex:`\Data`:raw-latex:`\output`\_folder_name”
  with its original filename as the subfolder name. Afterward, double
  click “index.html” (on
  e.g. “D::raw-latex:`\Downloads`:raw-latex:`\MRQy`-master:raw-latex:`\UserInterface`”)
  to open front-end user interface, and select the respective
  *results.tsv* file from the correct location
  e.g. “D::raw-latex:`\Downloads`:raw-latex:`\MRQy`-master:raw-latex:`\UserInterface`:raw-latex:`\Data`:raw-latex:`\output`\_folder_name”
  directory.

Contribution guidelines
-----------------------

Testing
~~~~~~~

::

   MRQy % pipenv shell
   (mrqy) MRQy% pipenv install .
   (mrqy) MRQY% pipenv run -m pytest tests/

Building on Travis
~~~~~~~~~~~~~~~~~~

The recommended path is to follow the `Forking
Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow>`__.
Create a Travis CI build for your github fork to validate your fork
before pushing a merge request to master.

Basic Information
-----------------

Measurements
~~~~~~~~~~~~

The measures of the MRQy tool are listed in the following table.

.. figure:: https://user-images.githubusercontent.com/50635618/76733243-cb9a3f80-6736-11ea-8100-a1bdb6f60d3f.png
   :alt: Picture1

   Picture1

User Interface
~~~~~~~~~~~~~~

The following figures show the user interface of the tool (index.html).

|C1| |C2| |C3|

Feedback and usage
------------------

Please report and issues, bugfixes, ideas for enhancements via the
“`Issues <https://github.com/ccipd/MRQy/issues>`__” tab.

Detailed usage instructions and an example of using MRQy to analyze TCIA
datasets are in the `Wiki <https://github.com/ccipd/MRQy/wiki>`__.

| You can cite this in any associated publication as:
| Sadri, AR, Janowczyk, A, Zou, R, Verma, R, Beig, N, Antunes, J,
  Madabhushi, A, Tiwari, P, Viswanath, SE, “Technical Note: MRQy — An
  open-source tool for quality control of MR imaging data”, Med. Phys.,
  2020, 47: 6029-6038. https://doi.org/10.1002/mp.14593

ArXiv: https://arxiv.org/abs/2004.04871

If you do use the tool in your own work, please drop us a line to let us
know.

.. |C1| image:: https://user-images.githubusercontent.com/50635618/78467306-3ce76580-76d9-11ea-8dbd-d43f82cd29a6.PNG
.. |C2| image:: https://user-images.githubusercontent.com/50635618/78467302-3bb63880-76d9-11ea-84ff-ce44f5f8a822.PNG
.. |C3| image:: https://user-images.githubusercontent.com/50635618/78467305-3ce76580-76d9-11ea-96a8-7574042c14c6.PNG
