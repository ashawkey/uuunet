# coding: utf-8
import os

base = "pred_MR"

files = os.listdir(base)
os.makedirs(os.path.join(base, "InPhase"), exist_ok=True)
os.makedirs(os.path.join(base, "OutPhase"), exist_ok=True)
        
for f in files:
    if "InPhase" in f:
        src = os.path.join(base, f)
        parts = f.split("_")
        parts2 = parts[2].split(".")
        dst = os.path.join(base, "InPhase", parts[0]+"_"+parts[1]+".nii."+parts2[-1])
        print(src, "----->", dst)
        os.rename(src, dst)
        
        
for f in files:
    if "OutPhase" in f:
        src = os.path.join(base, f)
        parts = f.split("_")
        parts2 = parts[2].split(".")
        dst = os.path.join(base, "OutPhase", parts[0]+"_"+parts[1]+".nii."+parts2[-1])
        print(src, "----->", dst)
        os.rename(src, dst)
