import os
import argparse
import numpy as np
import datetime
import QCF
import time
from medpy.io import load
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nfiledone = 0
csv_report = None
first = True
headers = []

def patient_name(root):
    return  [i.path for i in os.scandir(root)]


def volume(scan):
    image_data, image_header = load(scan)
    images = [image_data[:,:,i] for i in range(np.shape(image_data)[2])]
    ad = os.path.split(scan)
    return images, os.path.split(ad[1][:-4])[1], image_header

def saveThumbnails(v, output):
    os.makedirs(output + os.sep + v[1])
    for i in range(len(v[0])):
        plt.imsave(output + os.sep + v[1] + os.sep + v[1] + '(%d).png' % int(i+1), v[0][i], cmap = cm.Greys_r)
        print('image number %d out of %d is saved to %s' % (int(i+1), len(v[0]),output + os.sep + v[1]))

def worker_callback(s):
    global csv_report, first, nfiledone
    if nfiledone  == 0:
        
        csv_report = open(args.outdir + os.sep + "results" + ".tsv" , overwrite_flag, buffering=1)
        first = True

    if first and overwrite_flag == "w": 
        first = False
        csv_report.write("\n".join(["#" + s for s in headers])+"\n")
        csv_report.write("#dataset:"+"\t".join(s["output"])+"\n")
                         
    csv_report.write("\t".join([str(s[field]) for field in s["output"]])+"\n")
    csv_report.flush()
    nfiledone += 1
    print('The results is updated.')


if __name__ == '__main__':
    start_time = time.time()
    headers.append(f"start_time:\t{datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputdir',
                        help = "input foldername consists of *.mha files. For example: 'E:\\Data\\Rectal\\RectalCancer_Multisite\\UH'",
                        nargs = "*")
    parser.add_argument('-o','--outdir',
                        help = "a subfolder on the Data directory of the UserInterface i.e. E:\\Python_Codes\\Github\\UserInterface\\Data\\Output",
                        default = "", type=str)

    args = parser.parse_args()
    overwrite_flag = "w"        
    headers.append(f"outdir:\t{os.path.realpath(args.outdir)}")  
    root = args.inputdir[0]
    fname_outdir = args.outdir
    
    patients = patient_name(root)
    
    for i in patients:
        v = volume(i)
        saveThumbnails(v, fname_outdir)
        s = QCF.BaseVolume(fname_outdir, v)
        worker_callback(s)
    print("CT_QC program took", format((time.time() - start_time)/60, '.2f'), "minutes to run.")
            
        

    
    
