rm -r pred_MR
mkdir pred_MR
CUDA_VISIBLE_DEVICES=3 python uuunet/inference/predict_simple.py -i nnUNet_test/MR -o pred_MR -t Task00_CHAOSMR -tr nnUNetTrainer -m 2d -z -M 0
