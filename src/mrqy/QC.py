"""
Created on Sun Feb 10 11:21:31 2019, Last update on Tue July 13 10:42:18 PM 2021

@author: Amir Reza Sadri ars329@case.edu
"""
import sys
import os
import numpy as np
import argparse
import datetime
import mrqy.QCF as QCF                  # import QCF
import time
from medpy.io import load               # for .mha, .nii, or .nii.gz files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom                          # for .dcm files
from itertools import accumulate
import pandas as pd
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE
import umap
import scipy
from scipy.io import loadmat
import warnings        
warnings.filterwarnings("ignore")       # remove all warnings like conversion thumbnails

nfiledone = 0
csv_report = None
first = True
headers = []


def patient_name(root):
    
    # Starting message
    print('MRQy is starting....')
    
    # Gathering relevant file paths based on extensions. MRQy supports .dcm, .mha, .nii, .gz and .mat files
    files = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(root) 
                for filename in filenames 
                if filename.endswith('.dcm') 
                or filename.endswith('.mha')
                or filename.endswith('.nii')
                or filename.endswith('.gz')
                or filename.endswith('.mat')]
    
    # Separating files based on their extensions
    mats = [i for i in files if i.endswith('.mat')]
    dicoms = [i for i in files if i.endswith('.dcm')]
    mhas = [i for i in files 
            if i.endswith('.mha')
            or i.endswith('.nii')
            or i.endswith('.gz')]
    
    # Extracting subject identifiers from the different files
    mhas_subjects = [os.path.basename(scan)[:os.path.basename(scan).index('.')] for scan in mhas]
    dicom_subjects = []
    mat_subjects = [os.path.basename(scan)[:os.path.basename(scan).index('.')] for scan in mats]
    
    if folders_flag == "False":
        # Processing individual DICOM files
        for i in dicoms:
            dicom_subjects.append(pydicom.dcmread(i).PatientID) 
        # Counting occurences of each patient ID
        duplicateFrequencies = {}
        for i in dicom_subjects:
            duplicateFrequencies[i] = dicom_subjects.count(i)
        
        # Extracting unique patient IDs and their counts
        subjects_id = []
        subjects_number = []
        for i in range(len(duplicateFrequencies)):
              subjects_id.append(list(duplicateFrequencies.items())[i][0])
              subjects_number.append(list(duplicateFrequencies.items())[i][1])
        ind = [0] + list(accumulate(subjects_number))
        # Splitting DICOM files within folders
        splits = [dicoms[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
    
    elif folders_flag == "True":
        # Processing DICOM files within folders
        dicom_subjects = [d for d in os.listdir(root) if os.path.isdir(root + os.sep + d)]
        subjects_number = []
        for i in range(len(dicom_subjects)):
            # Counting DICOM files per subject
            subjects_number.append(
                len([os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(root + os.sep + dicom_subjects[i]) 
            for filename in filenames 
            if filename.endswith('.dcm')]))
        subjects_id  = dicom_subjects
        ind = [0] + list(accumulate(subjects_number))
        # Splitting DICOM files based on folders (subjects)
        splits = [dicoms[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

    # Combining all subject identifiers
    subjects = subjects_id + mhas_subjects + mat_subjects
    # Displaying the total number of identified subjects
    print('The number of patients is {}'.format(len(subjects)))
    # Returning varous lists containing file paths, subjects, and DICOM splits
    return files, subjects, splits, mhas, mhas_subjects, mats, mat_subjects


def volume_dicom(scans, name):
    
    # Selecting a portion of scans based on size
    scans = scans[int(0.005 *len(scans)*(100 - middle_size)):int(0.005 *len(scans)*(100 + middle_size))]
    # Reading metadata from the first DICOM file
    inf = pydicom.dcmread(scans[0])
    
    # Modyfying attributes if they axist
    if hasattr(inf, 'MagneticFieldStrength'):
        if inf.MagneticFieldStrength > 10:
            inf.MagneticFieldStrength = inf.MagneticFieldStrength/10000
    else:
        inf.MagneticFieldStrength = ''
    if hasattr(inf, 'Manufacturer') == False:
        inf.Manufacturer = ''
    if  hasattr(inf, 'RepetitionTime') == False:
            inf.RepetitionTime = 0
    if  hasattr(inf, 'EchoTime') == False:
            inf.EchoTime = 0
    
    # Determining name value based on folders_flag
    if folders_flag == "False":
        name_value = inf.PatientID
    elif folders_flag == "True":
        name_value = name
    
    # Creating a dictionnary of DICOM metadata attributes and their values    
    tags = {
             'ID': name_value,                                  # Patient / Subject ID
             'MFR': inf.Manufacturer,                           # Manufacturer name from the file header
             'VRX': format(inf.PixelSpacing[0], '.2f'),         # Voxel resolution in x plane
             'VRY': format(inf.PixelSpacing[1], '.2f'),         # Voxel resolution in y plane
             'VRZ': format(inf.SliceThickness, '.2f'),          # Voxel resolution in z plane
             'MFS': inf.MagneticFieldStrength,                  # Magnetic fiels strength from the file header
             'ROWS': int(inf.Rows),                             # Rows value of the volume
             'COLS': int(inf.Columns),                          # Columns value of the volume
             'TR': format(inf.RepetitionTime, '.2f'),           # Repetition time value of the volume
             'TE': format(inf.EchoTime, '.2f'),                 # Echo time value of the volume
             'NUM': len(scans)                                  # Number of slice images in each volume
    }
    
    # Fetching additional attributes if available
    tag_values = []
    # Checking for args variable 
    if args.t != 0:
        for de in tag_list:
            # if hasattr(inf, de) == False or inf.data_element(de).value == '':
            if hasattr(inf, de) == False:
                value = ''
            else:
                value = inf.data_element(de).value
            tag_values.append(value)
        res_dct = dict(zip(iter(tag_names), iter(tag_values)))
        tags.update(res_dct)
    
    # Reading and sorting DICOM files, creating a 3D image volume    
    slices = [pydicom.read_file(s) for s in scans]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    # PL = pd.DataFrame([s.pixel_array for s in slices], columns=['images'])
    # images = PL['images'].to_numpy().astype(np.int64)
    images = np.stack([s.pixel_array for s in slices])
    images = images.astype(np.int64)
    
    # Returning the image volume and associated tags
    return images, tags


def volume_notdicom(scan, name):
    # Loading image data and header
    image_data, image_header = load(scan)
    # Extracting 2D images from the 3D image data
    images = [image_data[:,:,i] for i in range(np.shape(image_data)[2])]
    # Return image, name and image header
    return images, name, image_header      


def volume_mat(mat_scan, name):
    # Loading volume data 
    v1 = loadmat(mat_scan)['vol']
    # Creating a dictionnary for the ID
    tags = {'ID': name}
    # Return the loaded volume data and the tags
    return v1, tags


def saveThumbnails_dicom(v, output):
    # Check if saving masks is enabled
    if save_masks_flag!='False':
        ffolder = output + '_foreground_masks'
        # Create a directory for foreground masks
        os.makedirs(ffolder + os.sep + v[1]['ID'])
    elif save_masks_flag=='False':
        ffolder = output
    # Create a directory for images
    os.makedirs(output + os.sep + v[1]['ID'])
    # Save images as thumbnails
    for i in range(0, len(v[0]), sample_size):
        plt.imsave(output + os.sep + v[1]['ID'] + os.sep + v[1]['ID'] + '(%d).png' % i, v[0][i], cmap = cm.Greys_r)
    # Print the number of saved images and the directory
    print('The number of %d images are saved to %s' % (len(v[0]),output + os.sep + v[1]['ID']))
    return ffolder + os.sep + v[1]['ID']


def saveThumbnails_mat(v, output):
    # Check if saving masks is enabled
    if save_masks_flag!='False':
        ffolder = output + '_foreground_masks'
        # Create a directory for foreground masks
        os.makedirs(ffolder + os.sep + v[1]['ID'])
    elif save_masks_flag=='False':
        ffolder = output
    # Create a directory for images 
    os.makedirs(output + os.sep + v[1]['ID'])
    # Save image as thumbnails
    for i in range(np.shape(v[0])[2]):
        plt.imsave(output + os.sep + v[1]['ID']+ os.sep + v[1]['ID'] + '(%d).png' % int(i+1), v[0][:,:,i], cmap = cm.Greys_r)
        # Print the number of saved images and the directory
    print('The number of %d images are saved to %s' % (np.shape(v[0])[2],output + os.sep + v[1]['ID']))
    return ffolder + os.sep + v[1]['ID']


def saveThumbnails_nondicom(v, output):
    # Create a directory for images
    os.makedirs(output + os.sep + v[1])
    # Save images as thumbnails, with a rotation
    for i in range(len(v[0])):
        plt.imsave(output + os.sep + v[1] + os.sep + v[1] + '(%d).png' % int(i+1), scipy.ndimage.rotate(v[0][i],270), cmap = cm.Greys_r)
        # print('image number %d out of %d is saved to %s' % (int(i+1), len(v[0]),output + os.sep + v[1]))
    print('The number of %d images are saved to %s' % (len(v[0]),output + os.sep + v[1]))


def worker_callback(s,fname_outdir):
    # Access global variables
    global csv_report, first, nfiledone
    # Check that the first file is being processed
    if nfiledone  == 0:
        # Open the CSV report file in append mode or overwrite mode
        csv_report = open(fname_outdir + os.sep + "results" + ".tsv" , overwrite_flag, buffering=1)
        first = True

    if first and overwrite_flag == "w": 
        first = False
        # Write comment lines for headers
        csv_report.write("\n".join(["#" + s for s in headers])+"\n")
        # Write comment line indicating the dataset and its output fields
        csv_report.write("#dataset:"+"\t".join(s["output"])+"\n")
    
    # Write data to the CSV report file                     
    csv_report.write("\t".join([str(s[field]) for field in s["output"]])+"\n")
    csv_report.flush()      # Flush the buffer to ensure writing immediately
    nfiledone += 1          # Increment the count of processed files
    print('The results are updated.')
    

def tsv_to_dataframe(tsvfileaddress):
    # Read the CSV file into a pandas dataframe, skipping the first two rows and using the third row as headers
    return pd.read_csv(tsvfileaddress, sep='\t', skiprows=2, header=0)


def data_whitening(dframe):
    # Fill missing values with N/A in the dataframe
    dframe = dframe.fillna('N/A')
    # Create a copy of the dataframe and select columns excluding object type
    df = dframe.copy()
    df = df.select_dtypes(exclude=['object'])
    # Apply whitening transformation to the selected numeric data
    ds = whiten(df)
    # Returns the transformed dataset
    return ds


def tsne_umap(dataframe, per):
    # Apply data whitening to the dataframe
    ds = data_whitening(dataframe)
    # Create a copy of the transformed dataframe for UMAP
    ds_umap = ds.copy()
    # Perform t-SNE on the transformed dataset
    tsne = TSNE(n_components=2, random_state=0, perplexity = per)   # per : must be less than n subjects. default = 30. Usually between 5 and 50
    tsne_obj = tsne.fit_transform(ds)
    # Add t-SNE components to the original DataFrame as 'x' and 'y'
    dataframe['x'] = tsne_obj[:,0].astype(float)
    dataframe['y'] = tsne_obj[:,1].astype(float)
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ds_umap)
    # Add UMAP components to the original DataFrame as 'u' and 'v'
    dataframe['u'] = embedding[:,0]
    dataframe['v'] = embedding[:,1]


def cleanup(final_address, per):
    # Read data from the final TSV file into a dataframe
    df = tsv_to_dataframe(final_address)
    # Apply t-SNE and UMAP dimensionality reduction on the dataframe
    tsne_umap(df, per)
    # Read only the header row from the final TSV file
    hf = pd.read_csv(final_address, sep='\t',  nrows=1)
    # Write the header row back to the same CSV file (overwriting the existing file)
    hf.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'w')
    # Append the modified dataframe to the CSV file (below the header)
    df.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'a')
    # Return the modified dataframe
    return df


def print_msg_box(msg, indent=1, width=None, title=None):
    # Split the message into lines
    lines = msg.split('\n')
    # Define the space with indentation
    space = " " * indent
    # Determine the width based on the maximum line length if width is not specified
    if not width:
        width = max(map(len, lines))
    # Create the top border of the box
    box = f'╔{"═" * (width + indent * 2)}╗\n'  
    # Add title section if title exists
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n' 
    # Add message lines to the box
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    # Add the bottom border of the box
    box += f'╚{"═" * (width + indent * 2)}╝' 
    # Print the box
    print(box)   
    
    
if __name__ == '__main__':
    # Record the start time for runtime measurement
    start_time = time.time() 
    # Add the start time information to the headers list
    headers.append(f"start_time:\t{datetime.datetime.now()}")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('output_folder_name',
                        help = "the subfolder name on the '...\\UserInterface\\Data\\output_folder_name' directory.",
                        type=str)
    parser.add_argument('inputdir',
                        help = "input foldername consists of *.mha (*.nii or *.dcm) files. For example: 'E:\\Data\\Rectal\\input_data_folder'",
                        nargs = "*")
    parser.add_argument('-r', help="folders as name", default=False)
    parser.add_argument('-s', help="save foreground masks", default=False)
    parser.add_argument('-b', help="number of samples", default=1, type = int)
    parser.add_argument('-u', help="percent of middle images", default=100)
    parser.add_argument('-t', help="the address of the user-specified tags list (*.txt)", default=0)
    parser.add_argument('-c', help="if yes the ch computes objects", default=False)
    
    args = parser.parse_args() 
    root = args.inputdir[0]
    
    # Setting flags based on parsed arguments
    if args.r == 0:
        folders_flag = "False"
    else: 
        folders_flag = args.r
    if args.s == 0:
        save_masks_flag = "False" 
    else: 
        save_masks_flag = args.s
    if args.b != 1:
        sample_size = args.b
    else: 
        sample_size = 1
    if args.u != 100:
        middle_size = int(args.u)
    else: 
        middle_size = 100
    if args.t != 0:
        tag_names = [line.strip() for line in open(args.t, "r")]
        tag_list = [line.strip().replace(" ", "") for line in open(args.t, "r")] 
    if args.c == 0:
        ch_flag = "False"
    else: 
        ch_flag = args.c 
    
    # print(os.getcwd())
    print_forlder_note = os.getcwd() + os.sep + 'UserInterface' 
    # print_forlder_note = os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir))+ os.sep + 'UserInterface' 
    
    # print(print_forlder_note)
    fname_outdir = print_forlder_note + os.sep + 'Data' + os.sep + args.output_folder_name
    
    overwrite_flag = "w"        
    headers.append(f"outdir:\t{os.path.realpath(fname_outdir)}") 
    patients, names, dicom_spil, nondicom_spli, nondicom_names, mat_spli, mat_names = patient_name(root)

    # Determine data types and set corresponding flags
    if len(dicom_spil) > 0 and len(nondicom_spli) > 0 and len(mat_spli) > 0:
        dicom_flag = True
        nondicom_flag = True
        mat_flag = True
    if len(dicom_spil) > 0 and len(nondicom_spli) > 0 and len(mat_spli) == 0:
        dicom_flag = True
        nondicom_flag = True
        mat_flag = False
    if len(dicom_spil) > 0 and len(nondicom_spli) == 0 and len(mat_spli) == 0:
        dicom_flag = True
        nondicom_flag = False
        mat_flag = False
    if len(dicom_spil) == 0 and len(nondicom_spli) > 0 and len(mat_spli) == 0:
        dicom_flag = False
        nondicom_flag = True
        mat_flag = False
    if len(dicom_spil) == 0 and len(nondicom_spli) > 0 and len(mat_spli) > 0:
        dicom_flag = False
        nondicom_flag = True
        mat_flag = True
    if len(dicom_spil) == 0 and len(nondicom_spli) == 0 and len(mat_spli) > 0:
        dicom_flag = False
        nondicom_flag = False
        mat_flag = True
    if len(dicom_spil) > 0 and len(nondicom_spli) == 0 and len(mat_spli) > 0:
        dicom_flag = True
        nondicom_flag = False
        mat_flag = True
    if len(dicom_spil) == 0 and len(nondicom_spli) == 0 and len(mat_spli) == 0:
        print('The input folder is empty or includes unsupported files format!')
    
    # Process each data type (DICOM, non-DICOM, MAT) if available
    for i in range(len(names)):
        if dicom_flag:
            for j in range(len(dicom_spil)):
                v = volume_dicom(dicom_spil[j], names[j])
                folder_foregrounds = saveThumbnails_dicom(v,fname_outdir)
                s = QCF.BaseVolume_dicom(fname_outdir, v,j+1,folder_foregrounds, sample_size, ch_flag)
                worker_callback(s,fname_outdir)
            dicom_flag = False
            
        if nondicom_flag:
            for l,k in enumerate(nondicom_spli):
                v = volume_notdicom(k, nondicom_names[l])
                saveThumbnails_nondicom(v,fname_outdir)
                s = QCF.BaseVolume_nondicom(fname_outdir, v,l+1, sample_size, ch_flag)
                worker_callback(s,fname_outdir)
            nondicom_flag = False
        
        if mat_flag:
            for j in range(len(mat_spli)):
                v = volume_mat(mat_spli[j], mat_names[j])
                folder_foregrounds = saveThumbnails_mat(v,fname_outdir)
                s = QCF.BaseVolume_mat(fname_outdir, v,j+1,folder_foregrounds)
                worker_callback(s,fname_outdir)
            mat_flag = False
    
    # Create the path for the result TSV file
    address = fname_outdir + os.sep + "results" + ".tsv" 
            
    if len(names) < 6:
        # t-SNE and UMPA cannot be performed if we have less than 6 images
        print('Skipped the t-SNE and UMAP computation because of insufficient data. The UMAP and t-SNE process need at least 6 input data.')
        df = tsv_to_dataframe(address)
    else:
        
        df = cleanup(address, 30)
        df = df.drop(['Name of Images'], axis=1)
        df = df.rename(columns={"#dataset:Patient": "Patient", 
                                "x":"TSNEX","y":"TSNEY", "u":"UMAPX", "v":"UMAPY" })
        df = df.fillna('N/A')
    
    # Save processed data as IQM.csv
    df.to_csv(fname_outdir + os.sep +'IQM.csv',index=False)
    print("The IQMs data are saved in the {} file. ".format(fname_outdir + os.sep + "IQM.csv"))
    
    # Print execution information
    print("Done!")
    print("MRQy program took", format((time.time() - start_time)/60, '.2f'), \
          "minutes for {} subjects and the overal {} MRI slices to run.".format(len(names),len(patients)))
    
    # Provide guidance for viewing the final results in the MRQy interface
    msg = "Please go to the '{}' directory and open up the 'index.html' file.\n".format(print_forlder_note) + \
    "Click on 'View Results' and select '{}' file.\n".format(fname_outdir + os.sep + "results.tsv")   
    print_msg_box(msg, indent=3, width=None, title="To view the final MRQy interface results:")
