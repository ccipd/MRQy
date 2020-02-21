"""
Created on Sat Feb  8 17:43:12 2020

@author: Amir Reza Sadri
"""

import os
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from skimage import exposure as ex
from skimage.filters import median
from skimage.morphology import square
from skimage.util import pad
import warnings

warnings.filterwarnings("ignore")

sample_size = 1

class BaseVolume_dicom(dict):

    def __init__(self, fname_outdir, v):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1]['ID'], v)
        self["outdir"] = fname_outdir
        self.addToPrintList("Name of Images", os.listdir(fname_outdir + os.sep + v[1]['ID']), v)
        self.addToPrintList("Manufacturer", v[1]['Manufacturer'], v)
        self.addToPrintList("MFS", v[1]['MFS'], v)
        self.addToPrintList("VR_x", v[1]['VR_x'], v)
        self.addToPrintList("VR_y", v[1]['VR_y'], v)
        self.addToPrintList("VR_z", v[1]['VR_z'], v)
        self.addToPrintList("Rows", v[1]['Rows'], v)
        self.addToPrintList("Columns", v[1]['Columns'], v)
        self.addToPrintList("TR", v[1]['TR'], v)
        self.addToPrintList("TE", v[1]['TE'], v)
        self["os_handle"] = v[0]
        self.addToPrintList("Number", v[1]['Number'], v)
        self.addToPrintList("Mean", vol(v, sample_size, "Mean"), v)
        self.addToPrintList("Range", vol(v, sample_size, "Range"), v)
        self.addToPrintList("Variance", vol(v, sample_size, "Variance"), v)
        self.addToPrintList("%CV", vol(v, sample_size, "%CV"), v)
        self.addToPrintList("CPP", vol(v, sample_size, "CPP"), v)
        self.addToPrintList("PSNR", vol(v, sample_size, "PSNR"), v)
        self.addToPrintList("SNR1", vol(v, sample_size, "SNR1"), v)
        self.addToPrintList("SNR2", vol(v, sample_size, "SNR2"), v)
        self.addToPrintList("SNR3", vol(v, sample_size, "SNR3"), v)
        self.addToPrintList("SNR4", vol(v, sample_size, "SNR4"), v)
        self.addToPrintList("CNR", vol(v, sample_size, "CNR"), v)
        self.addToPrintList("CVP", vol(v, sample_size, "CVP"), v)
        self.addToPrintList("EFC", vol(v, sample_size, "EFC"), v)
        self.addToPrintList("FBER", vol(v, sample_size, "FBER"), v)
        
    def addToPrintList(self, name, val, v):
        self[name] = val
        self["output"].append(name)
        print('The %s of the patient with the name of <%s> is %s' % (name, v[1]['ID'], val))


class BaseVolume_nondicom(dict):

    def __init__(self, fname_outdir, v):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1], v)
        self["outdir"] = fname_outdir
        self.addToPrintList("Name of Images", os.listdir(fname_outdir + os.sep + v[1]), v)
        self.addToPrintList("VR_x", format(v[2].get_voxel_spacing()[0], '.2f'), v)
        self.addToPrintList("VR_y", format(v[2].get_voxel_spacing()[1], '.2f'), v)
        self.addToPrintList("VR_z", format(v[2].get_voxel_spacing()[2], '.2f'), v)
        self.addToPrintList("Rows", np.shape(v[0])[1], v)
        self.addToPrintList("Columns", np.shape(v[0])[2], v)
        self["os_handle"] = v[0]
        self.addToPrintList("Number", len(v[0]), v)
        self.addToPrintList("Mean", vol(v, sample_size, "Mean"), v)
        self.addToPrintList("Range", vol(v, sample_size, "Range"), v)
        self.addToPrintList("Variance", vol(v, sample_size, "Variance"), v)
        self.addToPrintList("%CV", vol(v, sample_size, "%CV"), v)
        self.addToPrintList("CPP", vol(v, sample_size, "CPP"), v)
        self.addToPrintList("PSNR", vol(v, sample_size, "PSNR"), v)
        self.addToPrintList("SNR1", vol(v, sample_size, "SNR1"), v)
        self.addToPrintList("SNR2", vol(v, sample_size, "SNR2"), v)
        self.addToPrintList("SNR3", vol(v, sample_size, "SNR3"), v)
        self.addToPrintList("SNR4", vol(v, sample_size, "SNR4"), v)
        self.addToPrintList("CNR", vol(v, sample_size, "CNR"), v)
        self.addToPrintList("CVP", vol(v, sample_size, "CVP"), v)
        self.addToPrintList("EFC", vol(v, sample_size, "EFC"), v)
        self.addToPrintList("FBER", vol(v, sample_size, "FBER"), v)
        
    def addToPrintList(self, name, val, v):
        self[name] = val
        self["output"].append(name)
        print('The %s of the patient with the name of <%s> is %s' % (name, v[1], val))

def vol(v, sample_size, i):
    switcher={
            'Mean': mean,
            'Range': rang,
            'Variance': variance, 
            '%CV': percent_coefficient_variation,
            'CPP': contrast_per_pixle,
            'PSNR': fpsnr,
            'SNR1': snr1,
            'SNR2': snr2,
            'SNR3': snr3,
            'SNR4': snr4,
            'CNR': cnr,
            'CVP': cvp,
            'EFC': efc,
            'FBER': fber,
            }
    func=switcher.get(i)
    M = []
    for i in range(1, len(v[0]), sample_size):
        I = v[0][i]
#        I = I - np.min(I)  # for CT 
        F, B, c, f, b = foreground(I)
        if np.std(F) == 0:  # whole zero slice, no measure computing
            continue
        measure = func(F, B, c, f, b)
        if np.isnan(measure) or np.isinf(measure):
            continue
#            measure = 0
        # To do (add something)
        M.append(measure)
    return np.mean(M)
       

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
        ch = np.multiply(conv_hull, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    except Exception: 
        fore_image = img.copy()
        back_image = np.zeros_like(img, dtype=np.uint16)
        conv_hull = np.zeros_like(img, dtype=np.uint16)
    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]

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
    U = pad(img, pad_width=h, mode='constant')
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
