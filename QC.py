# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:24:51 2023

By: Amir R. Sadri (amir@picturehealth.com)
"""


import datetime
import time
import os
import sys
import re
import argparse
import yaml
import pydicom 
import nibabel as nib
import numpy as np
from itertools import accumulate
from collections import Counter
import matplotlib.pyplot as plt
import inspect
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from skimage import exposure as ex
import pandas as pd
import matplotlib.cm as cm
from scipy.signal import convolve2d as conv2
from skimage.filters import median
from skimage.morphology import square
from collections import defaultdict
import warnings        
warnings.filterwarnings("ignore")  


def patient_name(root, folders_flag=False):
    files = []
    for dirpath, _, filenames in os.walk(root):
        # Check if the directory contains any .dcm or .nii.gz files
        if any(filename.endswith('.dcm') for filename in filenames) or any(filename.endswith('.nii.gz') for filename in filenames):
            for filename in filenames:
                if filename.endswith('.dcm') or filename.endswith('.nii.gz'):
                    files.append(os.path.join(dirpath, filename))
    
    dicom_files = [i for i in files if i.endswith('.dcm')]
    nifti_files = [i for i in files if i.endswith('.nii.gz')]

    subjects_id = []
    subjects_number = []
    subject_count = defaultdict(int)
    if folders_flag == False:
        dicom_subjects = [pydicom.dcmread(i).PatientID for i in dicom_files]
        nifti_subjects = [nib.load(i).header['db_name'].decode('utf-8').strip() for i in nifti_files]
        subjects_id = dicom_subjects + nifti_subjects

        duplicateFrequencies = Counter(subjects_id)

        subjects_id = list(duplicateFrequencies.keys())
        subjects_number = list(duplicateFrequencies.values())

        ind = [0] + list(accumulate(subjects_number))
        splits = [files[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

    elif folders_flag == True:
        subjects = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for i in range(len(subjects)):
            subject_files = [os.path.join(dirpath, filename)
                             for dirpath, _, filenames in os.walk(os.path.join(root, subjects[i]))
                             for filename in filenames
                             if filename.endswith('.dcm') or filename.endswith('.nii.gz')]
            
            original_subject_id = subjects[i]
            subject_count[original_subject_id] += 1
            
            if subject_count[original_subject_id] > 1:
                subjects[i] = f"{original_subject_id}_{subject_count[original_subject_id]}"
            
            subjects_number.append(len(subject_files))
        subjects_id  = subjects
        ind = [0] + list(accumulate(subjects_number))
        splits = [files[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

    print('The number of participants is {}'.format(len(subjects_id)))
    return files, subjects_id, splits


def clean_value(number):
    if isinstance(number, (int, float)):
        number = '{:.2f}'.format(number)
        if number.replace(".", "", 1).isdigit():
            number = float(number)
            if number % 1 == 0:
                number = int(number)
            else:
                number = round(number, 2)
    number = np.array(number)
    return number

def extract_tags(inf, tag_data):
    non_tag_value = 'NA'
    pre_tags = pd.DataFrame.from_dict(tag_data, orient='index', columns=['Tag Abbreviation']).reset_index()
    pre_tags = pre_tags.rename(columns={'index': 'Tag Name'})
    pre_tags['Tag Abbreviation'] = pre_tags['Tag Abbreviation'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    tags = pd.DataFrame(columns=['Tag', 'Value'])
    for i, row in pre_tags.iterrows():
        pre_tag = row['Tag Name']
        pre_tag = pre_tag.replace(" ", "")
        try:
            tag_value = inf.get(pre_tag,non_tag_value)
            tag_value = clean_value(tag_value)
        except KeyError:
            tag_value = non_tag_value
        tag = row['Tag Abbreviation']
        mul_tags = tag.split(',')
        for j,k in enumerate(mul_tags):
            if np.iterable(tag_value):
                tag_value_j = tag_value[j] if j < len(tag_value) else non_tag_value
            else:
                tag_value_j = tag_value
            new_row = {'Tag': k, 'Value': tag_value_j}
            tags = pd.concat([tags, pd.DataFrame([new_row])], ignore_index=True)
    return tags


def volume_dicom(scans, name, name_suffix=""):
    institution = os.path.basename(os.path.dirname(os.path.dirname(scans[0])))
    scans = scans[int(0.005 *len(scans)*(100 - middle_size)):int(0.005 *len(scans)*(100 + middle_size))]
    inf = pydicom.dcmread(scans[0])
    tags = extract_tags(inf, tag_data)
    new_row1 = {'Tag': 'NUM', 'Value': len(scans)}
    new_row3 = {'Tag': 'INS', 'Value': institution}
    first_row = {'Tag': 'Participant ID', 'Value': f"{name}{name_suffix}"}
    tags = pd.concat([tags, pd.DataFrame([new_row1, new_row3])], ignore_index=True)
    tags.loc[-1] = first_row
    tags.index = tags.index + 1
    tags = tags.sort_index()
    tags = tags.set_index('Tag')['Value'].to_dict()
    slices = [pydicom.read_file(s) for s in scans]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    images = np.stack([s.pixel_array for s in slices])
    images = images.astype(np.int64)
    return images, tags


def saveThumbnails_dicom(v, output):
    participant = v[1]['Participant ID']
    if save_masks_flag!='False':
        ffolder = output + '_foreground_masks'
        os.makedirs(ffolder + os.sep + participant)
    elif save_masks_flag=='False':
        ffolder = output 
    os.makedirs(output + os.sep + participant)
    for i in range(0, len(v[0]), sample_size):
        plt.imsave(output + os.sep + participant + os.sep + participant + '(%d).png' % i, v[0][i], cmap = cm.Greys_r)
    participant_scan_number = int(np.ceil(len(v[0])/sample_size))
    print(f'-------------- Participant: {participant} --------------')
    print(f'The number of {participant_scan_number} scans were saved to {output + os.sep + participant}')
    return participant_scan_number, ffolder + os.sep + participant


class BaseVolume_dicom(dict):

    def __init__(self,v):
        dict.__init__(self)
        self["warnings"] = [] 
        self["output"] = []
        self["os_handle"] = v[0]
        participant = v[1]['Participant ID']
        self.addToPrintList(0, participant, "Participant", participant)
        
        for i,metric in enumerate(v[1]):
            count = i 
            if i==0:
                self.addToPrintList(count, participant,"Name of Images", os.listdir(fname_outdir + os.sep + participant))
            else:
                value = v[1][metric]
                self.addToPrintList(count, participant, metric, value)
                
        outputs_list = []
        for j in range(1, len(v[0]), sample_size):
            I = v[0][j]
            I = I - np.min(I)  # for CT 
            F, B, c, f, b = foreground(I)
            outputs = {}
            for func in functions:
                name, measure = func(F, B, c, f, b)
                outputs[name] = measure
            outputs_list.append(outputs)
        averages = {}
        for key in outputs_list[0].keys():
            values = [dic[key] for dic in outputs_list]
            averages[key] = np.mean(values) 
            count +=1
            self.addToPrintList(count, participant, key, averages[key])
        
    def addToPrintList(self, count, participant, metric, value):
        self[metric] = value
        self["output"].append(metric)
        if metric != 'Name of Images' and  metric != 'Participant':
            print(f'{count}. The {metric} of the participant {participant} is {value}.')


def foreground(img):
    try:
        h = ex.equalize_hist(img[:,:])*255
        oi = np.zeros_like(img, dtype=np.uint16)
        oi[(img > threshold_otsu(img)) == True] = 1
        oh = np.zeros_like(img, dtype=np.uint16)
        oh[(h > threshold_otsu(h)) == True] = 1
        nm = img.shape[0] * img.shape[1]
        w1 = np.sum(oi)/(nm)
        w2 = np.sum(oh)/(nm)
        ots = np.zeros_like(img, dtype=np.uint16)
        new =( w1 * img) + (w2 * h)
        ots[(new > threshold_otsu(new)) == True] = 1 
        conv_hull = convex_hull_image(ots)
        conv_hull = convex_hull_image(ots)
        ch = np.multiply(conv_hull, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    except Exception: 
        fore_image = img.copy()
        back_image = np.zeros_like(img, dtype=np.uint16)
        conv_hull = np.zeros_like(img, dtype=np.uint16)
        ch = np.multiply(conv_hull, 1)

    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]

def func1(F, B, c, f, b):
    name = 'MEAN'
    measure = np.nanmean(f)
    return name, measure

def func2(F, B, c, f, b):
    name = 'RNG'
    measure = np.ptp(f)
    return name, measure

def func3(F, B, c, f, b):
    name = 'VAR'
    measure = np.nanvar(f)
    return name, measure

def func4(F, B, c, f, b):
    name = 'CV'
    measure = (np.nanstd(f)/np.nanmean(f))*100
    return name, measure

def func5(F, B, c, f, b):
    name = 'CPP'
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(F, filt, mode='same')
    measure = np.nanmean(I_hat)
    return name, measure

def func6(F, B, c, f, b):
    name = 'PSNR'
    I_hat = median(F/np.max(F), square(5))
    measure = psnr(F, I_hat)
    return name, measure

def func7(F, B, c, f, b):
    name = 'SNR1'
    bg_std = np.nanstd(b)
    measure = np.nanstd(f) / (bg_std + 1e-9)
    return name, measure

def func8(F, B, c, f, b):
    name = 'SNR2'
    bg_std = np.nanstd(b)
    measure = np.nanmean(patch(F, 5)) / (bg_std + 1e-9)
    return name, measure 

def func9(F, B, c, f, b):
    name = 'SNR3'
    fore_patch = patch(F, 5)
    std_diff = np.nanstd(fore_patch - np.nanmean(fore_patch))
    if std_diff == 0:
        std_diff = 1e-9
    measure = np.nanmean(fore_patch) / std_diff
    return name, measure

def func10(F, B, c, f, b):
    name = 'SNR4'
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    bg_std = np.nanstd(back_patch)
    if bg_std == 0:
        bg_std = 1e-9
    measure = np.nanmean(fore_patch) / bg_std
    return name, measure


def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    U = np.pad(img, pad_width=5, mode='constant')
    [a,b]  = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return U[a:a+2*h+1,b:b+2*h+1]

def func11(F, B, c, f, b):
    name = 'CNR'
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    measure = np.nanmean(fore_patch-back_patch) / (np.nanstd(back_patch) + 1e-6)
    return name, measure

def func12(F, B, c, f, b):
    name = 'CVP'
    fore_patch = patch(F, 5)
    measure = np.nanstd(fore_patch) / (np.nanmean(fore_patch) + 1e-6 )
    return name, measure

def func13(F, B, c, f, b):
    name = 'CJV'
    measure = (np.nanstd(f) + np.nanstd(b)) / abs(np.nanmean(f) - np.nanmean(b))
    return name, measure

def func14(F, B, c, f, b):
    name = 'EFC'
    n_vox = F.shape[0] * F.shape[1]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (F**2).sum()
    b_max = np.sqrt(abs(cc))
    measure = float((1.0 / abs(efc_max)) * np.sum((F / b_max) * np.log((F + 1e16) / b_max)))
    return name, measure

def func15(F, B, c, f, b):
    name = 'FBER'
    fg_mu = np.nanmedian(np.abs(f) ** 2)
    bg_mu = np.nanmedian(np.abs(b) ** 2)
    if bg_mu < 1.0e-3:
        measure = 0
    measure = float(fg_mu / (bg_mu + 1e-6))
    return name, measure


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



def data_whitening(dframe):
    dframe = dframe.fillna('N/A')
    df = dframe.copy()
    df = df.select_dtypes(exclude=['object'])
    return df


def cleanup(final_address, per):
    df = pd.read_csv(final_address, sep='\t', skiprows=2, header=0)
    hf = pd.read_csv(final_address, sep='\t',  nrows=1)
    hf.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'w')
    df.to_csv(final_address, index = None, header=True, sep = '\t', mode = 'a')
    return df

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
    

  
nfiledone = 0
csv_report = None
first = True
headers = []

if __name__ == '__main__':
    print('MRQy (CT Version) is starting....')
    start_time = time.time() 
    headers.append(f"start_time:\t{datetime.datetime.now()}")
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
    
    args = parser.parse_args() 
    root = args.inputdir[0]
    
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
        
        
    print_forlder_note = os.getcwd() + os.sep + 'UserInterface' 
    fname_outdir = print_forlder_note + os.sep + 'Data' + os.sep + args.output_folder_name
    overwrite_flag = "w"        
    headers.append(f"outdir:\t{os.path.realpath(fname_outdir)}") 
    patients, names, dicom_spil = patient_name(root)
    
    with open('TAGS.yaml', 'rb') as file:
        tag_data = yaml.safe_load(file)
    
    total_tags = 0
    for value in tag_data.values():
        if isinstance(value, list):
            total_tags += len(value)
        else:
            total_tags += 1
        
    functions = [func for name, func in inspect.getmembers(sys.modules[__name__]) if name.startswith('func')]
    functions = sorted(functions, key=lambda f: int(re.search(r'\d+', f.__name__).group()))
    print(f'For each participant, {total_tags} tags will be extracted and {len(functions)} metrics will be computed.')
    
    total_scans = 0
    for i in range(len(names)):
            v = volume_dicom(dicom_spil[i], names[i])   
            participant_scan_number, folder_foregrounds = saveThumbnails_dicom(v,fname_outdir)
            total_scans += participant_scan_number
            s = BaseVolume_dicom(v)
            worker_callback(s,fname_outdir)
    
    address = fname_outdir + os.sep + "results" + ".tsv" 
            
        
    df = cleanup(address, 30)
    df = df.drop(['Name of Images'], axis=1)
    df = df.fillna('N/A')
        
    df.to_csv(fname_outdir + os.sep +'IQM.csv',index=False)  
    print("The IQMs data are saved in the {} file. ".format(fname_outdir + os.sep + "IQM.csv"))
    
    print("Done!")
    print("MRQy (CT Version) program took", format((time.time() - start_time)/60, '.2f'), \
          "minutes for {} subjects and the overal {} CT scans to run.".format(len(names),total_scans))
    
    msg = "Please go to the '{}' directory and open up the 'index.html' file.\n".format(print_forlder_note) + \
    "Click on 'View Results' and select '{}' file.\n".format(fname_outdir + os.sep + "results.tsv") 
          
    print_msg_box(msg, indent=3, width=None, title="To view the final MRQy (CT Version) interface results:")
    


