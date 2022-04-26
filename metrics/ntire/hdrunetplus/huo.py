import os
import os.path as osp
import numpy as np
import data_io as io
import utils 
import cv2
ref_dir = r"E:\NTIRE_single\testtrue\test\hdr"
ref_alignratio_dir = r"E:\NTIRE_single\testtrue\test\alignratio"

fake_dir = r"C:\Users\wanghu\Documents\experiment\img\base\results\ccnet_base\000_Valid_SingleFrame_FirstStage"
fake_alignratio_dir = r"C:\Users\wanghu\Documents\experiment\img\base\results\ccnet_base\000_Valid_SingleFrame_FirstStage\alignratio"

psnr = 0 
npsnr = 0
mpsnr = 0
idx = 0

def huo_uint16_png(image_path,align_ratio=65535):
    """ This function loads a uint16 png image from the specified path and restore its original image range with
    the ratio stored in the specified alignratio.npy respective path.
    Args:
        image_path (str): Path to the uint16 png image
        alignratio_path (str): Path to the alignratio.npy file corresponding to the image
    Returns:
        np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.
    """
    # Load the align_ratio variable and ensure is in np.float32 precision
    #align_ratio = np.load(al.astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio
    
for filename in sorted(os.listdir(ref_dir)):
    image_id = int(filename[:4])
    print(image_id)
    
    ref_hdr_image = io.imread_uint16_png(osp.join(ref_dir, "{:04d}_gt.png".format(image_id)), osp.join(ref_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    fake_hdr_image = huo_uint16_png(osp.join(fake_dir, "{:04d}.png".format(image_id)))
    temp =  utils.calculate_psnr(ref_hdr_image,fake_hdr_image)
    temp_n = utils.normalized_psnr(ref_hdr_image,fake_hdr_image,ref_hdr_image.max())
    temp_m = utils.calculate_tonemapped_psnr(fake_hdr_image,ref_hdr_image)
    psnr+=temp
    npsnr +=temp_n
    mpsnr += temp_m
    idx+=1
print("psnr:",psnr/idx,"npsnr:",npsnr/idx,"mpsnr:",mpsnr/idx)

