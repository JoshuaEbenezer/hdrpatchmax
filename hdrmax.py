from joblib import Parallel,delayed
from utils.hdr_utils import yuv_read
import numpy as np
import cv2
import os
import scipy.ndimage
import joblib
import niqe 
import save_stats
from numba import jit,prange
import argparse

parser = argparse.ArgumentParser(description='Generate HDRPatchMAX features from a single video')
parser.add_argument('--input_file',help='Input video file')
parser.add_argument('--results_file',help='File where features are stored')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--bit_depth', type=int,choices={8,10,12})
parser.add_argument('--color_space',choices={'BT2020','BT709'})

args = parser.parse_args()
C=1
def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='mirror'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean



def Y_compute_lnl(Y):
    if(len(Y.shape)==2):
        Y = np.expand_dims(Y,axis=2)

    maxY = scipy.ndimage.maximum_filter(Y,size=(17,17,1))
    minY = scipy.ndimage.minimum_filter(Y,size=(17,17,1))
    Y_scaled = -1+(Y-minY)* 2/(1e-3+maxY-minY)
    Y_transform =  np.exp(np.abs(Y_scaled)*4)-1
    Y_transform[Y_scaled<0] = -Y_transform[Y_scaled<0]
    return Y_transform



def hdrmax_fromvid(filename,filename_out,width,height,framenos,bit_depth):
    if(os.path.exists(filename)==False):
        print("Input video file does not exist")
        return
    if(os.path.exists(filename_out)):
        print("Output feature file already exists")
        return


    st_time_length = 5

    i = 0


    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    j=0

    i = 0
    C = 1e-3
    for framenum in range(framenos): 
        

    
        # read video file
        Y_pq,_,_ = yuv_read(filename,framenum,height,width,bit_depth)
        dY_pq =  cv2.resize(Y_pq,(width//2,height//2),interpolation=cv2.INTER_CUBIC)
        Y_pq =Y_pq.astype(np.float32)
        dY_pq = dY_pq.astype(np.float32)
            

        # normalize 
        Y = Y_pq/((2**bit_depth)-1)
        dY = dY_pq/((2**bit_depth)-1)

        
        # apply local nonlinearity
        Y_pq_nl = np.squeeze(Y_compute_lnl(Y))
        Y_down_pq_nl = np.squeeze(Y_compute_lnl(dY))

        # find MSCN
        Y_mscn_pq_nl,_,_ = save_stats.compute_image_mscn_transform(Y_pq_nl,C=0.001)
        dY_mscn_pq_nl,_,_ =save_stats.compute_image_mscn_transform(Y_down_pq_nl,C=0.001)

        # find HDRMAX features
        hdrmax_fullscale = save_stats.extract_subband_feats(Y_mscn_pq_nl)
        hdrmax_halfscale = save_stats.extract_subband_feats(dY_mscn_pq_nl)
        hdrmax = np.concatenate((hdrmax_fullscale,hdrmax_halfscale),axis=0)

        feat_sd_list.append(hdrmax)
        spatavg_list.append(hdrmax)

        i=i+1


        # compute rolling standard deviation
        if (i>=st_time_length): 


            # compute rolling standard deviation 
            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []


            i=0
    X1 = np.average(spatavg_list,axis=0)
    X2 = np.average(sd_list,axis=0)
    X = np.concatenate((X1,X2),axis=0)
    train_dict = {"features":X}
    joblib.dump(train_dict,filename_out)
    return




def main():

    args = parser.parse_args()
    vid_stream = open(args.input_file,'r')
    vid_stream.seek(0, os.SEEK_END)
    vid_filesize = vid_stream.tell()

    if(args.bit_depth==10 or args.bit_depth==12):
        multiplier = 3
    elif(args.bit_depth==8):
        multiplier=1.5
    vid_T = int(vid_filesize/(args.height*args.width*multiplier))
    
    hdrmax_fromvid(args.input_file,args.results_file,args.width,args.height,vid_T,args.bit_depth)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

