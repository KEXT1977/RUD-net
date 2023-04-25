##unet
CUDA_VISIBLE_DEVICES=5 python train.py --model_path=/home/extends/cxt/model/unet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet


#finetune
CUDA_VISIBLE_DEVICES=5 python train_finetune.py --model_path=/home/extends/cxt/model/unet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet_finetune --save_model_path=/home/extends/cxt/model/unet_fine/unet_finetune_model

#finetunefinalconv
CUDA_VISIBLE_DEVICES=5 python train_finetunefinalconv.py --model_path=/home/extends/cxt/model/unet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet_finetunefinalconv --save_model_path=/home/extends/cxt/model/unet_fine/unet_finetunefinalconv_model

#unet_resnet
训练：
CUDA_VISIBLE_DEVICES=1 python train.py --model_path=/home/extends/cxt/model/unet_resnet/model_new_8_withtest_512
验证：
CUDA_VISIBLE_DEVICES=5 python train.py --model_path=/home/extends/cxt/model/unet_resnet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet_resnet

#resnet_finetune
训练：
CUDA_VISIBLE_DEVICES=5 python train_finetune.py --model_path=/home/extends/cxt/model/unet_resnet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet_resnet_finetune --save_model_path=/home/extends/cxt/model/unet_resnet_fine/unet_resnet_finetune_model
验证：
同上，只需把CHECK_ACC=TRUE即可

#resnet_finetunefinalconv
训练：
CUDA_VISIBLE_DEVICES=5 python train_finetunefinalconv.py --model_path=/home/extends/cxt/model/unet_resnet/model_new_8_withtest_512 --output_img_path=/data2/cxt/CT/hospital_data_clean/png/pred_unet_resnet_finetunefinalconv --save_model_path=/home/extends/cxt/model/unet_resnet_fine/unet_resnet_finetunefinalconv_model
验证：
同上，只需把CHECK_ACC=TRUE即可