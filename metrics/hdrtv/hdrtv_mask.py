import os
import os.path as osp
import numpy as np
import utils 
from skimage.metrics import structural_similarity as ssim
ref_dir = r"E:\hdrtv\test_set\test_hdr_3"
#ref_alignratio_dir = r"E:\NTIRE_single\test\alignratio"

fake_dir = r"D:\sourceCode\openSource\KUNet\models\img_hdrtv_fine\000_Valid_SingleFrame_FirstStage"
#fake_alignratio_dir = r"C:\Users\wanghu\Documents\experiment\img\results\result_3skip\000_Valid_SingleFrame_FirstStage\alignratio"


psnr = 0 
npsnr = 0
mpsnr = 0
idx = 0
nssim = 0
mse = 0 
deltaE_ITP  = 0
for filename in sorted(os.listdir(ref_dir)):
    image_id = int(filename[:3])
    print(image_id)
    ref_hdr_image = utils.read_img(env =None,path=osp.join(ref_dir, "{:03d}.png".format(image_id)))
    fake_hdr_image = utils.read_img(env=None,path=osp.join(fake_dir, "{:03d}.png".format(image_id+1)))
    temp =  utils.calculate_psnr(ref_hdr_image,fake_hdr_image)
    
    
    temp_deltaE_ITP =utils.calculate_hdr_deltaITP(ref_hdr_image,fake_hdr_image)
   # temp_n = utils.normalized_psnr(ref_hdr_image,fake_hdr_image,ref_hdr_image.max())
    #temp_m = utils.calculate_tonemapped_psnr(fake_hdr_image,ref_hdr_image)
    temp_ssim = ssim(ref_hdr_image,fake_hdr_image,multichannel=True)
    #temp_mse = utils.mse_hdr(ref_hdr_image,fake_hdr_image)
    psnr+=temp
    deltaE_ITP += temp_deltaE_ITP
   # npsnr +=temp_n
    #mpsnr += temp_m
    nssim += temp_ssim
    #mse += temp_mse
    idx+=1
    print(image_id,"psnr:",temp)
print("psnr:",psnr/idx,"deltaE_ITP:",deltaE_ITP/idx,"mpsnr:",mpsnr/idx,"ssim:",nssim/idx,"mse:",mse/idx)