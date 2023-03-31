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

parser = argparse.ArgumentParser(description='Generate HDR ChipQA features from a single video')
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

@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,step,h,w):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((step-1)/2),cx+int((step-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<h else h-1 for j in range(step)])
    else:
        y_sts = np.arange(cy-int((step-1)/2),cy+int((step-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<w else w-1 for j in range(step)]) 
    return x_sts,y_sts

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,h,step):
    st_kurtosis = np.zeros((len(theta),))
    data = np.zeros((len(theta),step**2))
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data[index,:] =Y3d_mscn[:,y_sts*h+x_sts].flatten() 
        data_mu4 = np.mean((data[index,:]-np.mean(data[index,:]))**4)
        data_var = np.var(data[index,:])
        st_kurtosis[index] = data_mu4/(data_var**2+1e-4)
    idx = (np.abs(st_kurtosis - 3)).argmin()
    
    data_slice = data[idx,:]
    return data_slice


def find_kurtosis_sts(grad_img_buffer,step,cy,cx,rst,rct,theta):

    h = grad_img_buffer[step-1].shape[0]
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(step,-1))
    sts_grad= [find_kurtosis_slice(gradY3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]

    return sts_grad


def Y_compute_lnl(Y):
    if(len(Y.shape)==2):
        Y = np.expand_dims(Y,axis=2)

    maxY = scipy.ndimage.maximum_filter(Y,size=(17,17,1))
    minY = scipy.ndimage.minimum_filter(Y,size=(17,17,1))
    Y_scaled = -1+(Y-minY)* 2/(1e-3+maxY-minY)
    Y_transform =  np.exp(np.abs(Y_scaled)*4)-1
    Y_transform[Y_scaled<0] = -Y_transform[Y_scaled<0]
    return Y_transform


def extract_ptlebrisque_patches(img1, patch_size,ptle):

    Y_mscn,Y_std,_ = save_stats.compute_image_mscn_transform(img1,C=0.001,extend_mode='reflect')
    Y_patches = blockshaped(Y_mscn,patch_size,patch_size)
    Y_std_patches = blockshaped(Y_std,patch_size,patch_size)
    brisque_patch_feats = np.asarray(Parallel(n_jobs=5)(delayed(save_stats.extract_subband_feats)(patch) for patch in Y_patches))
    std_arr = np.average(Y_std_patches,axis=(1,2))
    percentiles = np.percentile(std_arr,[ptle,100-ptle])
    p1 = np.where(std_arr<=percentiles[0])
    p2 = np.where((std_arr>percentiles[0]) & (std_arr<percentiles[1]))
    p3 = np.where(std_arr>=percentiles[1])
    p_list = [p1,p2,p3]

    brisque_ptle_feats = np.hstack([np.average(brisque_patch_feats[p],0) \
        if len(p)>0 else np.average(brisque_patch_feats,0) for p in p_list])
    return brisque_ptle_feats 

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))


def hdrchipqa_fromvid(filename,filename_out,width,height,framenos,bit_depth):
    if(os.path.exists(filename)==False):
        print("Input video file does not exist")
        return
    if(os.path.exists(filename_out)):
        print("Output feature file already exists")
        return


    st_time_length = 5
    t = np.arange(0,st_time_length)
    a=0.5
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)
    cap = cv2.VideoCapture(filename)

#
    theta = np.arange(0,np.pi,np.pi/6)
    ct = np.cos(theta)
    st = np.sin(theta)
    lower_r = int((st_time_length+1)/2)-1
    higher_r = int((st_time_length+1)/2)
    r = np.arange(-lower_r,higher_r)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)






    step = st_time_length
    cy, cx = np.mgrid[step:height-step*4:step*4, step:width-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:height//2-step*4:step*4, step:width//2-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block

    

    grad_img_buffer = np.zeros((st_time_length,height,width))
    graddown_img_buffer =np.zeros((st_time_length,height//2,width//2))


    i = 0



    r1 = len(np.arange(step,height-step*4,step*4)) 
    r2 = len(np.arange(step,width-step*4,step*4)) 
    dr1 = len(np.arange(step,height//2-step*4,step*4)) 
    dr2 = len(np.arange(step,width//2-step*4,step*4)) 
    

    

    
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    j=0

    i = 0
    C = 1e-3
    for framenum in range(framenos): 
        

    
        Y_pq,_,_ = yuv_read(filename,framenum,height,width,bit_depth)
        dY_pq =  cv2.resize(Y_pq,(width//2,height//2),interpolation=cv2.INTER_CUBIC)
        Y_pq =Y_pq.astype(np.float32)
        dY_pq = dY_pq.astype(np.float32)
            

        
        Y = Y_pq/((2**bit_depth)-1)
        dY = dY_pq/((2**bit_depth)-1)
#
        gradient_x = cv2.Sobel(Y,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        
        gradient_x_down = cv2.Sobel(dY,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(dY,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    



        gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag)
        dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down)

        niqe_feats = niqe.compute_niqe_features(Y,C=C)

        full_scale_features = extract_ptlebrisque_patches(Y,20,10)        
        half_scale_features = extract_ptlebrisque_patches(dY,10,10)        
        patchmax = np.concatenate((full_scale_features,half_scale_features))
        
        Y_pq_nl = np.squeeze(Y_compute_lnl(Y))
        Y_down_pq_nl = np.squeeze(Y_compute_lnl(dY))
        Y_mscn_pq_nl,_,_ = save_stats.compute_image_mscn_transform(Y_pq_nl,C=0.001)
        dY_mscn_pq_nl,_,_ =save_stats.compute_image_mscn_transform(Y_down_pq_nl,C=0.001)

        brisque_nl_fullscale = save_stats.extract_subband_feats(Y_mscn_pq_nl)
        brisque_nl_halfscale = save_stats.extract_subband_feats(dY_mscn_pq_nl)
        brisque_nl = np.concatenate((brisque_nl_fullscale,brisque_nl_halfscale),axis=0)

        feat_sd_list.append(np.concatenate((brisque_nl,patchmax)))
        spatavg_list.append(np.concatenate((niqe_feats,patchmax,brisque_nl)))

        grad_img_buffer[i,:,:] =gradY_mscn 
        graddown_img_buffer[i,:,:]=dgradY_mscn 
        i=i+1
#

        # compute ST Gradient chips and rolling standard deviation
        if (i>=st_time_length): 

            # temporal filtering
            grad3d_mscn = spatiotemporal_mscn(grad_img_buffer,avg_window)
            graddown3d_mscn = spatiotemporal_mscn(graddown_img_buffer,avg_window)

            # compute rolling standard deviation 
            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []

            # ST chips
            sts_grad = find_kurtosis_sts(grad3d_mscn,step,cy,cx,rst,rct,theta)
            dsts_grad = find_kurtosis_sts(graddown3d_mscn,step,dcy,dcx,rst,rct,theta)
            sts_grad= unblockshaped(np.reshape(sts_grad,(-1,st_time_length,st_time_length)),r1*st_time_length,r2*st_time_length)
            dsts_grad= unblockshaped(np.reshape(dsts_grad,(-1,st_time_length,st_time_length)),dr1*st_time_length,dr2*st_time_length)
            grad_feats = save_stats.extract_subband_feats(sts_grad)
            dgrad_feats = save_stats.extract_subband_feats(dsts_grad)



            allst_feats = np.concatenate((grad_feats,dgrad_feats),axis=0)
            X_list.append(allst_feats)


            # refresh buffer
            grad_img_buffer = np.zeros((st_time_length,height,width))
            graddown_img_buffer =np.zeros((st_time_length,int(height/2),int(width/2)))
            i=0
    X1 = np.average(spatavg_list,axis=0)
    X2 = np.average(sd_list,axis=0)
    X3 = np.average(X_list,axis=0)
    X = np.concatenate((X1,X2,X3),axis=0)
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
    
    hdrchipqa_fromvid(args.input_file,args.results_file,args.width,args.height,vid_T,args.bit_depth)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

