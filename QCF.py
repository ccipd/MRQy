import os
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from skimage import exposure as ex
import cv2

class BaseVolume(dict):

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
        self.addToPrintList("Mean", Measuremnets(v)[0], v)
        self.addToPrintList("Range", Measuremnets(v)[1], v)
        self.addToPrintList("%CV", Measuremnets(v)[2], v)
        self.addToPrintList("CPP", Measuremnets(v)[3], v)
        self.addToPrintList("PSNR", Measuremnets(v)[4], v)
        self.addToPrintList("SNR1", Measuremnets(v)[5], v)
        self.addToPrintList("SNR2", Measuremnets(v)[6], v)
        self.addToPrintList("SNR3", Measuremnets(v)[7], v)
        self.addToPrintList("SNR4", Measuremnets(v)[8], v) 
        self.addToPrintList("SNR5", Measuremnets(v)[9], v)
        self.addToPrintList("CNR", Measuremnets(v)[10], v)
        self.addToPrintList("CVP", Measuremnets(v)[11], v)
        self.addToPrintList("EFC", Measuremnets(v)[12], v)
        self.addToPrintList("FBER", Measuremnets(v)[13], v)
        
    def addToPrintList(self, name, val, v):
        self[name] = val
        self["output"].append(name)
        print('The %s of the dataset with the name of <%s> is %s' % (name, v[1], val))
        
def Measuremnets(v):
    M = []
    R = []
    CV = []
    CPP = []
    PSNR = []
    SNR1 = []
    SNR2 = []
    SNR3 = []
    SNR4 = []
    SNR5 = []
    CNR = []
    CVP = []
    EFC = []
    FBER = []
    for I in v[0]:
        I = I - np.min(I)
        F, B = foreground(I)
        M.append(mean(F))
        R.append(rang(F))
        CV.append(coefficient_variation(F))
        CPP.append(contrast_per_pixle(F))
        PSNR.append(fpsnr(F, 5))
        SNR1.append(snr(F, B, 5)[0])
        SNR2.append(snr(F, B, 5)[1])
        SNR3.append(snr(F, B, 5)[2])
        SNR4.append(snr(F, B, 5)[3])
        SNR5.append(snr(F, B, 5)[4])
        CNR.append(snr(F, B, 5)[5])
        CVP.append(snr(F, B, 5)[6])
        EFC.append(efc(F))
        FBER.append(fber(I, F))      
    return (np.mean(M) , np.mean(R), np.mean(CV),np.mean(CPP), np.mean(PSNR), 
            np.mean(SNR1), np.mean(SNR2), np.mean(SNR3), np.mean(SNR4), np.mean(SNR5), np.mean(CNR), np.mean(CVP),
            np.mean(EFC), np.mean(FBER))

def foreground(img):
    hist_eq = he(img)
    ots = np.zeros_like(img, dtype=np.uint16)
    ots[(hist_eq > threshold_otsu(hist_eq)) == True] = 1
    ch = convex_hull_image(ots)
    ch = np.multiply(ch, 1)
    fore_image = ch * img
    back_image = (1 - ch) * img
    return fore_image, back_image

def he(img):
    return ex.equalize_hist(img[:,:])*255

def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mean(img):
    return np.mean(img)

def rang(img):
    return np.max(img)-np.min(img)

def coefficient_variation(I):
    m = mean(I)
    s = np.std(I)
    return (s/m)*100

def contrast_per_pixle(I):
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(I, filt, mode='same')
    return np.mean(I_hat)

def fpsnr(I, N):
    I_hat = cv2.medianBlur(np.float32(I), N*N)
    return psnr(I, I_hat)

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    [n,m] = np.shape(img)
    G = np.ones((2 * h + n, 2 * h + m))
    G[h: n + h, h: m + h] = img
    [a,b]  = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return G[a:a+2*h+1,b:b+2*h+1]

def snr(F, B, patch_size):
    fore_patch = patch(F, patch_size)
    back_patch = patch(B, patch_size)
    st = np.std(back_patch)
    bt = np.std(B)
    if bt == 0:
        bt = 50    
    if st == 0:
        st = 10
    ms1 = np.mean(F) / bt
    ms2 = np.mean(fore_patch) / bt
    ms3 = np.mean(fore_patch) / np.std(fore_patch - np.mean(fore_patch))
    ms4 = np.sum(fore_patch - back_patch) /st
    ms5 = np.mean(fore_patch) /np.std(fore_patch)
    ms6 = np.mean(fore_patch - back_patch) / st
    ms7 = np.std(fore_patch) / np.mean(fore_patch)
    return ms1, ms2, ms3, ms4, ms5, ms6, ms7


def efc(img):
    framemask = np.zeros_like(img, dtype=np.uint8)
    n_vox = np.sum(1 - framemask)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (img[framemask == 0]**2).sum()
    b_max = np.sqrt(abs(cc))
    return float((1.0 / abs(efc_max)) * np.sum(
        (img[framemask == 0] / b_max) * np.log(
        (img[framemask == 0] + 1e16) / b_max)))

def fber(img, K):
    fg_mu = np.median(np.abs(img[K > 0]) ** 2)
    airmask = np.ones_like(K, dtype=np.uint8)
    airmask[K > 0] = 0
    bg_mu = np.median(np.abs(img[airmask == 1]) ** 2)
    if abs(bg_mu) < 1.0e-3:
        return 0
    return float(np.abs(fg_mu / bg_mu))