import os
import os.path as osp
import numpy as np
import data_io as io
import utils 

ref_dir = r"E:\NTIRE_single\test\hdr"
ref_alignratio_dir = r"E:\NTIRE_single\test\alignratio"

fake_dir = r"C:\Users\wanghu\Documents\experiment\img\results\DT\000_Valid_SingleFrame_FirstStage"
fake_alignratio_dir = r"C:\Users\wanghu\Documents\experiment\img\results\DT\000_Valid_SingleFrame_FirstStage\alignratio"


psnr = 0 
npsnr = 0
mpsnr = 0
idx = 0
nssim = 0
mse = 0 
for filename in sorted(os.listdir(ref_dir)):
    image_id = int(filename[:4])
    
    ref_hdr_image = io.imread_uint16_png(osp.join(ref_dir, "{:04d}_gt.png".format(image_id)), osp.join(ref_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    fake_hdr_image = io.imread_uint16_png(osp.join(fake_dir, "{:04d}.png".format(image_id)), osp.join(fake_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    temp =  utils.calculate_psnr(ref_hdr_image,fake_hdr_image)
    temp_n = utils.normalized_psnr(ref_hdr_image,fake_hdr_image,ref_hdr_image.max())
    temp_m = utils.calculate_tonemapped_psnr(fake_hdr_image,ref_hdr_image)
    temp_ssim = utils.normalized_ssim(ref_hdr_image,fake_hdr_image,ref_hdr_image.max())
    temp_mse = utils.mse_hdr(ref_hdr_image,fake_hdr_image)
    psnr+=temp
    npsnr +=temp_n
    mpsnr += temp_m
    nssim += temp_ssim
    mse += temp_mse
    idx+=1
    print(image_id,"npsnr:",temp_n,"mpsnr",temp_m)
print("psnr:",psnr/idx,"npsnr:",npsnr/idx,"mpsnr:",mpsnr/idx,"ssim:",nssim/idx,"mse:",mse/idx)