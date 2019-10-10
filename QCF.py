import os
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from skimage import exposure as ex
from skimage.filters import median
from skimage.morphology import square
import warnings

warnings.filterwarnings("ignore")

sample_size = 1 # number of slice's samples for measure computation per patient

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
        self.addToPrintList("SNR5", vol(v, sample_size, "SNR5"), v)
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
            '%CV': coefficient_variation,
            'CPP': contrast_per_pixle,
            'PSNR': fpsnr,
            'SNR1': snr1,
            'SNR2': snr2,
            'SNR3': snr3,
            'SNR4': snr4,
            'SNR5': snr5,
            'CNR': cnr,
            'CVP': cvp,
            'EFC': efc,
            'FBER': fber,
            }
    func=switcher.get(i)
    M = []
    for i in range(1, len(v[0]), sample_size):
        I = v[0][i]
#        I = I - np.min(I)
        F, B = foreground(I)
        M.append(func(I,F,B))
    return np.mean(M)
       
def he(img):
    return ex.equalize_hist(img[:,:])*255

def foreground(img):
    try:
        hist_eq = he(img)
        ots = np.zeros_like(img, dtype=np.uint16)
        ots[(hist_eq > threshold_otsu(hist_eq)) == True] = 1
        ch = convex_hull_image(ots)
        ch = np.multiply(ch, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    except Exception: 
        fore_image = img.copy()
        back_image = 1 - img
    return fore_image, back_image

def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mean(img, foreground, background):
    return np.mean(foreground)

def rang(img, foreground, background):
    return np.ptp(foreground)

def variance(img, foreground, background):
    return np.var(foreground)

def coefficient_variation(img, foreground, background):
    m = np.mean(foreground)
    s = np.std(foreground)
    if m== 0:
        m = 1
    return (s/m)*100

def contrast_per_pixle(img, foreground, background):
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(foreground, filt, mode='same')
    return np.mean(I_hat)

def fpsnr(img, foreground, background):
    I_hat = median(foreground/np.max(foreground), square(5))
    return psnr(foreground, I_hat)

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    [n,m] = np.shape(img)
    G = np.ones((2 * h + n, 2 * h + m))
    G[h: n + h, h: m + h] = img
    [a,b]  = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return G[a:a+2*h+1,b:b+2*h+1]

def snr1(img, foreground, background):
    std_background = np.std(background)
    if std_background == 0:
        std_background = 0.01 
    return np.mean(foreground) / std_background

def snr2(img, foreground, background):
    fore_patch = patch(foreground, 5)
    std_background = np.std(background)
    if std_background == 0:
        std_background = 0.01 
    return np.mean(fore_patch) / std_background

def snr3(img, foreground, background):
    fore_patch = patch(foreground, 5)
    return np.mean(fore_patch) / np.std(fore_patch - np.mean(fore_patch))

def snr4(img, foreground, background):
    fore_patch = patch(foreground, 5)
    back_patch = patch(background, 5)
    std_back_patch = np.std(back_patch)
    if std_back_patch == 0:
        std_back_patch = 0.01 
    return np.sum(fore_patch - back_patch) / std_back_patch

def snr5(img, foreground, background):
    fore_patch = patch(foreground, 5)
    return np.mean(fore_patch) /np.std(fore_patch)

def cnr(img, foreground, background):
    fore_patch = patch(foreground, 5)
    back_patch = patch(background, 5)
    std_back_patch = np.std(back_patch)
    if std_back_patch == 0:
        std_back_patch = 0.01
    return np.mean(fore_patch - back_patch) / std_back_patch

def cvp(img, foreground, background):
    fore_patch = patch(foreground, 5)
    return np.std(fore_patch) / np.mean(fore_patch)

def efc(img, foreground, background):
    framemask = np.zeros_like(img, dtype=np.uint8)
    n_vox = np.sum(1 - framemask)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (img[framemask == 0]**2).sum()
    b_max = np.sqrt(abs(cc))
    if b_max == 0:
        b_max = 1
    return float((1.0 / abs(efc_max)) * np.sum(
        (img[framemask == 0] / b_max) * np.log(
        (img[framemask == 0] + 1e16) / b_max)))

def fber(img, foreground, background):
    fg_mu = np.median(np.abs(img[foreground > 0]) ** 2)
    airmask = np.ones_like(foreground, dtype=np.uint8)
    airmask[foreground > 0] = 0
    bg_mu = np.median(np.abs(img[airmask == 1]) ** 2)
    if abs(bg_mu) < 1.0e-3:
        return 0
    return float(np.abs(fg_mu / bg_mu))
