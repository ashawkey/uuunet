import os
import shutil

base = "/home/tang/chaos/Processed_nii/Test_Sets/CT"
out = "/home/tang/nnUNet/nnUNet_test/CT"

cases = os.listdir(os.path.join(base, 'img'))

for c in cases:
    print(c)
    shutil.copy(os.path.join(base, 'img', c), os.path.join(out, c + '_0000.nii.gz'))
