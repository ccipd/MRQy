"""
Created on Sun Feb 10 11:21:31 2019, Last update on Sun Mar 15 00:01:05 2020

@author: Amir Reza Sadri
"""

import os
import numpy as np
import argparse
import datetime
import QCF
import time
from medpy.io import load    # for .mha, .nii, or .nii.gz files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom               # for .dcm files
from itertools import accumulate
import pandas as pd
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE
import umap
import scipy
import warnings        
warnings.filterwarnings("ignore")    # remove all warnings like conversion thumbnails

nfiledone = 0
csv_report = None
first = True
headers = []


def patient_name(root):
    print('MRQy is starting....')
    files = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(root) 
            for filename in filenames 
            if filename.endswith('.dcm') 
            or filename.endswith('.mha')
            or filename.endswith('.nii')
            or filename.endswith('.gz')]
    dicoms = [i for i in files if i.endswith('.dcm')]
    mhas = [i for i in files 
            if i.endswith('.mha')
            or i.endswith('.nii')
            or i.endswith('.gz')]
    mhas_subjects = [os.path.basename(scan)[:os.path.basename(scan).index('.')] for scan in mhas]
    dicom_subjects = []
    for i in dicoms:
        dicom_subjects.append(pydicom.dcmread(i).PatientID)  
    duplicateFrequencies = {}
    for i in dicom_subjects:
        duplicateFrequencies[i] = dicom_subjects.count(i)
    subjects_id = []
    subjects_number = []
    for i in range(len(duplicateFrequencies)):
         subjects_id.append(list(duplicateFrequencies.items())[i][0])
         subjects_number.append(list(duplicateFrequencies.items())[i][1])
    ind = [0] + list(accumulate(subjects_number))
    splits = [dicoms[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
    subjects = subjects_id + mhas_subjects
    print('The number of patients is {}'.format(len(subjects)))
    return files, subjects, splits, mhas, mhas_subjects


def volume_dicom(scans):   
    inf = pydicom.dcmread(scans[0])
    if hasattr(inf, 'MagneticFieldStrength'):
        if inf.MagneticFieldStrength > 10:
            inf.MagneticFieldStrength = inf.MagneticFieldStrength/10000
    else:
        inf.MagneticFieldStrength = ''
        
    if hasattr(inf, 'Manufacturer') == False:
        inf.Manufacturer = ''
        
    tags = {
             'ID': inf.PatientID,
             'Manufacturer': inf.Manufacturer,
             'VR_x': format(inf.PixelSpacing[0], '.2f'),
             'VR_y': format(inf.PixelSpacing[1], '.2f'),
             'VR_z': format(inf.SliceThickness, '.2f'),
             'MFS': inf.MagneticFieldStrength,
             'Rows': int(inf.Rows),
             'Columns': int(inf.Columns),
             'TR': format(inf.RepetitionTime, '.2f'),
             'TE': format(inf.EchoTime, '.2f'),
             'Number': len(scans)
    }
    
    slices = [pydicom.read_file(s) for s in scans]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    images = np.stack([s.pixel_array for s in slices])
    images = images.astype(np.int64)
    return images, tags

def volume_notdicom(scan, name):
    image_data, image_header = load(scan)
    images = [image_data[:,:,i] for i in range(np.shape(image_data)[2])]
    return images, name, image_header      
    
def saveThumbnails_dicom(v, output):
    os.makedirs(output + os.sep + v[1]['ID'])
    for i in range(len(v[0])):
        plt.imsave(output + os.sep + v[1]['ID'] + os.sep + v[1]['ID'] + '(%d).png' % i, v[0][i], cmap = cm.Greys_r)
        # print('image number %d out of %d is saved to %s' % (int(i+1), len(v[0]),output + os.sep + v[1]['ID']))
    print('The number of %d images are saved to %s' % (len(v[0]),output + os.sep + v[1]['ID']))
    
def saveThumbnails_nondicom(v, output):
    os.makedirs(output + os.sep + v[1])
    for i in range(len(v[0])):
        plt.imsave(output + os.sep + v[1] + os.sep + v[1] + '(%d).png' % int(i+1), scipy.ndimage.rotate(v[0][i],270), cmap = cm.Greys_r)
        # print('image number %d out of %d is saved to %s' % (int(i+1), len(v[0]),output + os.sep + v[1]))
    print('The number of %d images are saved to %s' % (len(v[0]),output + os.sep + v[1]))

def worker_callback(s,fname_outdir):
    global csv_report, first, nfiledone
    if nfiledone  == 0:
        csv_report = open(fname_outdir + os.sep + "results" + ".tsv" , overwrite_flag, buffering=1)
        first = True

    if first and overwrite_flag == "w": 
        first = False
        csv_report.write("\n".join(["#" + s for s in headers])+"\n")
        csv_report.write("#dataset:"+"\t".join(s["output"])+"\n")
                         
    csv_report.write("\t".join([str(s[field]) for field in s["output"]])+"\n")
    csv_report.flush()
    nfiledone += 1
    print('The results are updated.')
    


def tsv_to_dataframe(tsvfileaddress):
    return pd.read_csv(tsvfileaddress, sep='\t', skiprows=2, header=0)


def data_whitening(dframe):
    dframe = dframe.fillna('N/A')
    df = dframe.copy()
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[0], axis=1)
    ds = whiten(df)
    return ds

def tsne_umap(dataframe, per):
    ds = data_whitening(dataframe)
    ds_umap = ds.copy()
    tsne = TSNE(n_components=2, random_state=0, perplexity = per)
    tsne_obj = tsne.fit_transform(ds)
    dataframe['x'] = tsne_obj[:,0].astype(float)
    dataframe['y'] = tsne_obj[:,1].astype(float)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ds_umap)
    dataframe['u'] = embedding[:,0]
    dataframe['v'] = embedding[:,1]


def cleanup(final_address, per):
    df = tsv_to_dataframe(final_address)
    tsne_umap(df, per)
    hf = pd.read_csv(final_address, sep='\t',  nrows=1)
    hf.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'w')
    df.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'a')

def print_msg_box(msg, indent=1, width=None, title=None):
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝' 
    print(box)   
    
    

if __name__ == '__main__':
    start_time = time.time()
    headers.append(f"start_time:\t{datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('output_folder_name',
                        help = "the subfolder name on the '...\\UserInterface\\Data\\output_folder_name' directory.",
                        type=str)
    parser.add_argument('inputdir',
                        help = "input foldername consists of *.mha (*.nii or *.dcm) files. For example: 'E:\\Data\\Rectal\\input_data_folder'",
                        nargs = "*")

    args = parser.parse_args() 
    root = args.inputdir[0]
    print_forlder_note = os.getcwd() + os.sep + 'UserInterface' 
    fname_outdir = print_forlder_note + os.sep + 'Data' + os.sep + args.output_folder_name
    overwrite_flag = "w"        
    headers.append(f"outdir:\t{os.path.realpath(fname_outdir)}") 
    patients, names, dicom_spil, nondicom_spli, nondicom_names = patient_name(root)

    if len(dicom_spil) > 0 and len(nondicom_spli) > 0:
        dicom_flag = True
        nondicom_flag = True
    if len(dicom_spil) > 0 and len(nondicom_spli) == 0:
        dicom_flag = True
        nondicom_flag = False
    if len(dicom_spil) == 0 and len(nondicom_spli) > 0:
        dicom_flag = False
        nondicom_flag = True
    if len(dicom_spil) == 0 and len(nondicom_spli) == 0:
        print('The input folder is empty!')
    
    for i in range(len(names)):
        if dicom_flag:
            for j in range(len(dicom_spil)):
                v = volume_dicom(dicom_spil[j])
                saveThumbnails_dicom(v,fname_outdir)
                s = QCF.BaseVolume_dicom(fname_outdir, v,j+1)
                worker_callback(s,fname_outdir)
            dicom_flag = False
            
        if nondicom_flag:
            for l,k in enumerate(nondicom_spli):
                v = volume_notdicom(k, nondicom_names[l])
                saveThumbnails_nondicom(v,fname_outdir)
                s = QCF.BaseVolume_nondicom(fname_outdir, v,l+1)
                worker_callback(s,fname_outdir)
            nondicom_flag = False

    address = fname_outdir + os.sep + "results" + ".tsv" 
    cleanup(address, 30)
    
    
    print("Done!")
    print("MRQy program took", format((time.time() - start_time)/60, '.2f'), \
          "minutes for {} subjects and the overal {} MRI slices to run.".format(len(names),len(patients)))
    
    msg = "Please go to the '{}' directory and open up the 'index.html' file.\n".format(print_forlder_note) + \
    "Click on 'View Results' and select '{}' file.\n".format(fname_outdir + os.sep + "results.tsv") 
          
    print_msg_box(msg, indent=3, width=None, title="To view the final MRQy interface results:")
    
    