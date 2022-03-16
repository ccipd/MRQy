# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:45:32 2020

@author: Amir Reza Sadri
"""

"""
Created on Sun Feb 10 11:21:31 2019, Last update on Mon Feb 08 08:24:05 2021

@author: Amir Reza Sadri
"""

import os
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image,convex_hull_object
from skimage import exposure as ex
from skimage.filters import median
from skimage.morphology import square
# from skimage.util import pad   pad is not available in skimage==0.19.2
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

warnings.filterwarnings("ignore")


class BaseVolume_dicom(dict):

    def __init__(self, fname_outdir, v, ol,folder_foregrounds, sample_size, ch_flag):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1]['ID'], v, ol, 170)
        self["outdir"] = fname_outdir
        self.addToPrintList("Name of Images", os.listdir(fname_outdir + os.sep + v[1]['ID']), v, ol, 100)
        for i,j in enumerate(v[1]):
            if i != 0:
                self.addToPrintList(j, v[1][j], v, ol, i)
        # self.addToPrintList("MFR", v[1]['Manufacturer'], v, ol, 1)
        # self.addToPrintList("MFS", v[1]['MFS'], v, ol, 2)
        # self.addToPrintList("VRX", v[1]['VR_x'], v, ol, 3)
        # self.addToPrintList("VRY", v[1]['VR_y'], v, ol, 4)
        # self.addToPrintList("VRZ", v[1]['VR_z'], v, ol, 5)
        # self.addToPrintList("ROWS", v[1]['Rows'], v, ol, 6)
        # self.addToPrintList("COLS", v[1]['Columns'], v, ol, 7)
        # self.addToPrintList("TR", v[1]['TR'], v, ol, 8)
        # self.addToPrintList("TE", v[1]['TE'], v, ol, 9)
        self["os_handle"] = v[0]
        # self.addToPrintList("NUM", v[1]['Number'], v, ol, 10)
        self.addToPrintList("MEAN", vol(v, sample_size, "Mean",folder_foregrounds, ch_flag), v, ol, 11)
        self.addToPrintList("RNG", vol(v, sample_size, "Range",folder_foregrounds, ch_flag), v, ol, 12)
        self.addToPrintList("VAR", vol(v, sample_size, "Variance",folder_foregrounds, ch_flag), v, ol, 13)
        self.addToPrintList("CV", vol(v, sample_size, "CV",folder_foregrounds, ch_flag), v, ol, 14)
        self.addToPrintList("CPP", vol(v, sample_size, "CPP",folder_foregrounds, ch_flag), v, ol, 15)
        self.addToPrintList("PSNR", vol(v, sample_size, "PSNR",folder_foregrounds, ch_flag), v, ol, 16)
        self.addToPrintList("SNR1", vol(v, sample_size, "SNR1",folder_foregrounds, ch_flag), v, ol, 17)
        self.addToPrintList("SNR2", vol(v, sample_size, "SNR2",folder_foregrounds, ch_flag), v, ol, 18)
        self.addToPrintList("SNR3", vol(v, sample_size, "SNR3",folder_foregrounds, ch_flag), v, ol, 19)
        self.addToPrintList("SNR4", vol(v, sample_size, "SNR4",folder_foregrounds, ch_flag), v, ol, 20)
        self.addToPrintList("CNR", vol(v, sample_size, "CNR",folder_foregrounds, ch_flag), v, ol, 21)
        self.addToPrintList("CVP", vol(v, sample_size, "CVP",folder_foregrounds, ch_flag), v, ol, 22)
        self.addToPrintList("CJV", vol(v, sample_size, "CJV",folder_foregrounds, ch_flag), v, ol, 23)
        self.addToPrintList("EFC", vol(v, sample_size, "EFC",folder_foregrounds, ch_flag), v, ol, 24)
        self.addToPrintList("FBER", vol(v, sample_size, "FBER",folder_foregrounds, ch_flag), v, ol, 25)
        
    def addToPrintList(self, name, val, v, ol, il):
        self[name] = val
        self["output"].append(name)
        if name != 'Name of Images' and il != 170:
            print('%s-%s. The %s of the patient with the name of <%s> is %s' % (ol,il,name, v[1]['ID'], val))


class BaseVolume_nondicom(dict):

    def __init__(self, fname_outdir, v, ol, sample_size, ch_flag):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1], v, ol, 170)
        self["outdir"] = fname_outdir
        self.addToPrintList("Name of Images", os.listdir(fname_outdir + os.sep + v[1]), v, ol, 100)
        self.addToPrintList("VRX", format(v[2].get_voxel_spacing()[0], '.2f'), v, ol, 1)
        self.addToPrintList("VRY", format(v[2].get_voxel_spacing()[1], '.2f'), v, ol, 2)
        self.addToPrintList("VRZ", format(v[2].get_voxel_spacing()[2], '.2f'), v, ol, 3)
        self.addToPrintList("ROWS", np.shape(v[0])[1], v, ol, 4)
        self.addToPrintList("COLS", np.shape(v[0])[2], v, ol, 5)
        self["os_handle"] = v[0]
        self.addToPrintList("NUM", len(v[0]), v, ol, 6)
        self.addToPrintList("MEAN", vol(v, sample_size, "Mean",fname_outdir, ch_flag), v, ol, 7)
        self.addToPrintList("RNG", vol(v, sample_size, "Range",fname_outdir, ch_flag), v, ol, 8)
        self.addToPrintList("VAR", vol(v, sample_size, "Variance",fname_outdir, ch_flag), v, ol, 9)
        self.addToPrintList("CV", vol(v, sample_size, "CV",fname_outdir, ch_flag), v, ol, 10)
        self.addToPrintList("CPP", vol(v, sample_size, "CPP",fname_outdir, ch_flag), v, ol, 11)
        self.addToPrintList("PSNR", vol(v, sample_size, "PSNR",fname_outdir, ch_flag), v, ol, 12)
        self.addToPrintList("SNR1", vol(v, sample_size, "SNR1",fname_outdir, ch_flag), v, ol, 13)
        self.addToPrintList("SNR2", vol(v, sample_size, "SNR2",fname_outdir, ch_flag), v, ol, 14)
        self.addToPrintList("SNR3", vol(v, sample_size, "SNR3",fname_outdir, ch_flag), v, ol, 15)
        self.addToPrintList("SNR4", vol(v, sample_size, "SNR4",fname_outdir, ch_flag), v, ol, 16)
        self.addToPrintList("CNR", vol(v, sample_size, "CNR",fname_outdir, ch_flag), v, ol, 17)
        self.addToPrintList("CVP", vol(v, sample_size, "CVP",fname_outdir, ch_flag), v, ol, 18)
        self.addToPrintList("CJV", vol(v, sample_size, "CJV",fname_outdir, ch_flag), v, ol, 19)
        self.addToPrintList("EFC", vol(v, sample_size, "EFC",fname_outdir, ch_flag), v, ol, 20)
        self.addToPrintList("FBER", vol(v, sample_size, "FBER",fname_outdir, ch_flag), v, ol, 21)
        
    def addToPrintList(self, name, val, v, ol, il):
        self[name] = val
        self["output"].append(name)
        if name != 'Name of Images' and il != 170:
            print('%s-%s. The %s of the patient with the name of <%s> is %s' % (ol,il,name, v[1], val))


class BaseVolume_mat(dict):

    def __init__(self, fname_outdir, v, ol,folder_foregrounds, sample_size):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1]['ID'], v, ol, 170)
        self["outdir"] = fname_outdir
        self.addToPrintList("Name of Images", os.listdir(fname_outdir + os.sep + v[1]['ID']), v, ol, 100)
        self.addToPrintList("ROWS", np.shape(v[0])[0], v, ol, 1)
        self.addToPrintList("COLS", np.shape(v[0])[1], v, ol, 2)
        self["os_handle"] = v[0]
        self.addToPrintList("NUM", np.shape(v[0])[2], v, ol, 3)
        self.addToPrintList("MEAN", vol(v, sample_size, "Mean",folder_foregrounds), v, ol, 4)
        self.addToPrintList("RNG", vol(v, sample_size, "Range",folder_foregrounds), v, ol, 5)
        self.addToPrintList("VAR", vol(v, sample_size, "Variance",folder_foregrounds), v, ol, 6)
        self.addToPrintList("CV", vol(v, sample_size, "CV",folder_foregrounds), v, ol, 7)
        self.addToPrintList("CPP", vol(v, sample_size, "CPP",folder_foregrounds), v, ol, 8)
        self.addToPrintList("PSNR", vol(v, sample_size, "PSNR",folder_foregrounds), v, ol, 9)
        self.addToPrintList("SNR1", vol(v, sample_size, "SNR1",folder_foregrounds), v, ol, 10)
        self.addToPrintList("SNR2", vol(v, sample_size, "SNR2",folder_foregrounds), v, ol, 11)
        self.addToPrintList("SNR3", vol(v, sample_size, "SNR3",folder_foregrounds), v, ol, 12)
        self.addToPrintList("SNR4", vol(v, sample_size, "SNR4",folder_foregrounds), v, ol, 13)
        self.addToPrintList("CNR", vol(v, sample_size, "CNR",folder_foregrounds), v, ol, 14)
        self.addToPrintList("CVP", vol(v, sample_size, "CVP",folder_foregrounds), v, ol, 15)
        self.addToPrintList("CJV", vol(v, sample_size, "CJV",folder_foregrounds), v, ol, 16)
        self.addToPrintList("EFC", vol(v, sample_size, "EFC",folder_foregrounds), v, ol, 17)
        self.addToPrintList("FBER", vol(v, sample_size, "FBER",folder_foregrounds), v, ol, 18)
        
    def addToPrintList(self, name, val, v, ol, il):
        self[name] = val
        self["output"].append(name)
        if name != 'Name of Images' and il != 170:
            print('%s-%s. The %s of the patient with the name of <%s> is %s' % (ol,il,name, v[1]['ID'], val))

def vol(v, sample_size, kk, outi_folder, ch_flag):
    switcher={
            'Mean': mean,
            'Range': rang,
            'Variance': variance, 
            'CV': percent_coefficient_variation,
            'CPP': contrast_per_pixle,
            'PSNR': fpsnr,
            'SNR1': snr1,
            'SNR2': snr2,
            'SNR3': snr3,
            'SNR4': snr4,
            'CNR': cnr,
            'CVP': cvp,
            'CJV': cjv,
            'EFC': efc,
            'FBER': fber,
            }
    func=switcher.get(kk)
    M = []
    for i in range(1, len(v[0]), sample_size):
        I = v[0][i]
#        I = I - np.min(I)  # for CT 
        F, B, c, f, b = foreground(I,outi_folder,v,i)
        if np.std(F) == 0:  # whole zero slice, no measure computing
            continue
        measure = func(F, B, c, f, b)
        if np.isnan(measure) or np.isinf(measure):
            continue
            # measure = 0
        # To do (add something)
        M.append(measure)
    return np.mean(M)
       


def foreground(img,save_folder,v,inumber):
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
    
    # if not os.path.isdir(save_folder + os.sep + v[1]['ID']):
    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]



# def vol(v, sample_size, kk,outi_folder, ch_flag):
#     switcher={
#             'Mean': mean,
#             'Range': rang,
#             'Variance': variance, 
#             'CV': percent_coefficient_variation,
#             'CPP': contrast_per_pixle,
#             'PSNR': fpsnr,
#             'SNR1': snr1,
#             'SNR2': snr2,
#             'SNR3': snr3,
#             'SNR4': snr4,
#             'CNR': cnr,
#             'CVP': cvp,
#             'CJV': cjv,
#             'EFC': efc,
#             'FBER': fber,
#             }
#     func=switcher.get(kk)
#     M = []
#     for i in range(1, len(v[0]), sample_size):
#         I = v[0][i]
# #        I = I - np.min(I)  # for CT 
#         F, B, c, f, b = foreground(I,outi_folder,v,i, ch_flag)
#         if np.std(F) == 0:  # whole zero slice, no measure computing
#             continue
#         measure = func(F, B, c, f, b)
#         if np.isnan(measure) or np.isinf(measure):
#             continue
#             # measure = 0
#         # To do (add something)
#         M.append(measure)
#     return np.mean(M)
       

# def foreground(img,save_folder,v,inumber, ch_flag):
#     try:
#         h = ex.equalize_hist(img[:,:])*255
#         oi = np.zeros_like(img, dtype=np.uint16)
#         oi[(img > threshold_otsu(img)) == True] = 1
#         oh = np.zeros_like(img, dtype=np.uint16)
#         oh[(h > threshold_otsu(h)) == True] = 1
#         nm = img.shape[0] * img.shape[1]
#         w1 = np.sum(oi)/(nm)
#         w2 = np.sum(oh)/(nm)
#         ots = np.zeros_like(img, dtype=np.uint16)
#         new =( w1 * img) + (w2 * h)
#         ots[(new > threshold_otsu(new)) == True] = 1 
#         if ch_flag == 'True':
#             conv_hull = convex_hull_object(ots)
#         elif ch_flag == 'False':
#             conv_hull = convex_hull_image(ots)
#         conv_hull = convex_hull_image(ots)
#         ch = np.multiply(conv_hull, 1)
#         fore_image = ch * img
#         back_image = (1 - ch) * img
#     except Exception: 
#         fore_image = img.copy()
#         back_image = np.zeros_like(img, dtype=np.uint16)
#         conv_hull = np.zeros_like(img, dtype=np.uint16)
#         ch = np.multiply(conv_hull, 1)
    
#     # if not os.path.isdir(save_folder + os.sep + v[1]['ID']):
#     if '_foreground_masks' in save_folder + os.sep + v[1]['ID']:
#         plt.imsave(save_folder + os.sep + v[1]['ID'] +'(%d).png' % int(inumber+1), scipy.ndimage.rotate(ch,0), cmap = cm.Greys_r)
#     return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]

def mean(F, B, c, f, b):
    return np.nanmean(f)

def rang(F, B, c, f, b):
    return np.ptp(f)

def variance(F, B, c, f, b):
    return np.nanvar(f)

def percent_coefficient_variation(F, B, c, f, b):
    return (np.nanstd(f)/np.nanmean(f))*100

def contrast_per_pixle(F, B, c, f, b):
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(F, filt, mode='same')
    return np.nanmean(I_hat)

def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))

def fpsnr(F, B, c, f, b):
    I_hat = median(F/np.max(F), square(5))
    return psnr(F, I_hat)

def snr1(F, B, c, f, b):
    return np.nanstd(f) / np.nanstd(b)

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    # U = pad(img, pad_width=h, mode='constant')
    U = np.pad(img, pad_width=5, mode='constant')
    [a,b]  = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return U[a:a+2*h+1,b:b+2*h+1]

def snr2(F, B, c, f, b):
    fore_patch = patch(F, 5)
    return np.nanmean(fore_patch) / np.nanstd(b)

def snr3(F, B, c, f, b):
    fore_patch = patch(F, 5)
    return np.nanmean(fore_patch)/np.nanstd(fore_patch - np.nanmean(fore_patch))

def snr4(F, B, c, f, b):
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    return np.nanmean(fore_patch) / np.nanstd(back_patch)

def cnr(F, B, c, f, b):
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    return np.nanmean(fore_patch-back_patch) / np.nanstd(back_patch)

def cvp(F, B, c, f, b):
    fore_patch = patch(F, 5)
    return np.nanstd(fore_patch) / np.nanmean(fore_patch)

def cjv(F, B, c, f, b):
    return (np.nanstd(f) + np.nanstd(b)) / abs(np.nanmean(f) - np.nanmean(b))

def efc(F, B, c, f, b):
    n_vox = F.shape[0] * F.shape[1]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (F**2).sum()
    b_max = np.sqrt(abs(cc))
    return float((1.0 / abs(efc_max)) * np.sum(
        (F / b_max) * np.log((F + 1e16) / b_max)))

def fber(F, B, c, f, b):
    fg_mu = np.nanmedian(np.abs(f) ** 2)
    bg_mu = np.nanmedian(np.abs(b) ** 2)
    if bg_mu < 1.0e-3:
        return 0
    return float(fg_mu / bg_mu)
