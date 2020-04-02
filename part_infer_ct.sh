rm -r pred_CT
mkdir pred_CT
CUDA_VISIBLE_DEVICES=2 python uuunet/inference/predict_simple.py -i nnUNet_test/CT_part -o pred_CT -t Task01_CHAOSCT -tr nnUNetTrainer -m 2d -z -M 1
