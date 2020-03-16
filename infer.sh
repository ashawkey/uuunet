CUDA_VISIBLE_DEVICES=4 python nnunet/inference/predict_simple.py -i nnUNet_test/MR -o pred_MR -t Task00_CHAOSMR -tr nnUNetTrainer -m 2d -z
