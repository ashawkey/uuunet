import SimpleITK as sitk
import numpy as np
import cv2
import os
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes

OUTDIR = "submit"

def largest_CC(mask):
    labels = label(mask)
    target = np.argmax(np.bincount(labels.flat)[1:])+1
    res = (labels == target).astype(np.uint8)
    return res

def post_process_ct(mask):
    mask = largest_CC(mask)
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask[mask>0] = 63
    return mask

def post_process_mr_liver(mask):
    mask[mask==2] = 0
    mask[mask==3] = 0
    mask[mask==4] = 0
    mask = largest_CC(mask)
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask[mask>0] = 63
    return mask

def post_process_mr_abdom(mask):
    res = np.zeros_like(mask)
    for i in range(1, 5):
        mask_organ = mask == i
        mask_organ = largest_CC(mask_organ)
        mask_organ = binary_fill_holes(mask_organ).astype(np.uint8)
        res += mask_organ.astype(np.uint8) * i
    res[res==1] = 63
    res[res==2] = 126
    res[res==3] = 189
    res[res==4] = 252
    return res

def view_batch(imgs, lbls, labels=['image', 'label'], stack=False):
    '''
    imgs: [D, H, W, C], the depth or batch dimension should be the first.
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title(labels[0])
    ax2.set_title(labels[1])
    """
    if init with zeros, the animation may not update? seems bug in animation.
    """
    if stack:
        lbls = np.stack((lbls, imgs, imgs), -1)
    img1 = ax1.imshow(np.random.rand(*imgs.shape[1:]))
    img2 = ax2.imshow(np.random.rand(*lbls.shape[1:]))
    def update(i):
        plt.suptitle(str(i))
        img1.set_data(imgs[i])
        img2.set_data(lbls[i])
        return img1, img2
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=10, blit=False, repeat_delay=0)
    plt.show()

def prepare_folders():
    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
    else:
        os.makedirs(OUTDIR)
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task1/CT/{i}/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task1/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task1/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task4/CT/{i}/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task4/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task4/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task2/CT/{i}/Results/", exist_ok=True)
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task3/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task3/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task5/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task5/MR/{i}/T2SPIR/Results/", exist_ok=True)


# CT+MR, liver
def task1():
    print("Task1")
    ct = glob.glob('pred_CT/*.nii.gz')
    t1 = glob.glob('pred_MR/*T1*gz')
    t2 = glob.glob('pred_MR/*T2*gz')
    ct.sort()
    t1.sort()
    t2.sort()
    # prepare results for CT
    for i in range(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        x = post_process_ct(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task1/CT/{setnumber}/Results/img{z:03}.png', s)

    # prepare results for MR
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        x = post_process_mr_liver(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task1/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x = post_process_mr_liver(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task1/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)

# CT+MR, abdominal
def task4():
    print("Task4")
    ct = glob.glob('pred_CT/*.nii.gz')
    t1 = glob.glob('pred_MR/*T1*gz')
    t2 = glob.glob('pred_MR/*T2*gz')
    ct.sort()
    t1.sort()
    t2.sort()
    # prepare results for CT
    for i in trange(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        x = post_process_ct(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/CT/{setnumber}/Results/img{z:03}.png', s)

    # prepare results for MR
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        x = post_process_mr_abdom(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x = post_process_mr_abdom(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)

# CT, liver
def task2():
    ct = glob.glob('pred_CT/*.nii.gz')
    ct.sort()
    # prepare results
    for i in trange(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        x = post_process_ct(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task2/CT/{setnumber}/Results/img{z:03}.png', s)


# MRI, abdominal
def task5():
    print("Task5")
    t1 = glob.glob('pred_MR/*T1*gz')
    t2 = glob.glob('pred_MR/*T2*gz')
    t1.sort()
    t2.sort()
    # prepare results
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        x = post_process_mr_abdom(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task5/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x = post_process_mr_abdom(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task5/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)

# MRI, liver
def task3():
    print("Task3")
    t1 = glob.glob('pred_MR/*T1*gz')
    t2 = glob.glob('pred_MR/*T2*gz')
    t1.sort()
    t2.sort()
    # prepare results
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        x = post_process_mr_liver(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task3/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x = post_process_mr_liver(x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task3/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)


if __name__ == "__main__":
    prepare_folders()
    #task1() # CT & MR liver
    #task2() # CT liver
    task3() # MR liver
    #task4() # CT & MR Abdom
    task5() # MR Abdom
