import SimpleITK as sitk
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange

OUTDIR = "submit"
os.makedirs(OUTDIR, exist_ok=True)

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

# CT+MR, liver
def task1():
    print("Task1")
    ct = glob.glob('pred_CT/*.nii.gz')
    t1 = glob.glob('pred_MR/*T1*gz')
    t2 = glob.glob('pred_MR/*T2*gz')
    ct.sort()
    t1.sort()
    t2.sort()
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task1/CT/{i}/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task1/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task1/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # prepare results for CT
    for i in range(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        #view_batch(x, x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task1/CT/{setnumber}/Results/img{z:03}.png', s)

    # prepare results for MR
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        #view_batch(x, x)

        x[x==1] = 63
        x[x==2] = 0
        x[x==3] = 0
        x[x==4] = 0

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task1/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        #view_batch(x, x)

        x[x==1] = 63
        x[x==2] = 0
        x[x==3] = 0
        x[x==4] = 0

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
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task4/CT/{i}/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task4/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task4/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # prepare results for CT
    for i in trange(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        #view_batch(x, x)

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/CT/{setnumber}/Results/img{z:03}.png', s)

    # prepare results for MR
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))

        x[x==1] = 63
        x[x==2] = 126
        x[x==3] = 189
        x[x==4] = 252


        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x[x==1] = 63
        x[x==2] = 126
        x[x==3] = 189
        x[x==4] = 252


        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task4/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)

# CT, liver
def task2():
    ct = glob.glob('pred_CT/*.nii.gz')
    ct.sort()
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task2/CT/{i}/Results/", exist_ok=True)
    # prepare results
    for i in trange(len(ct)):
        setnumber = ct[i].split('.')[0].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(ct[i]))
        #view_batch(x, x)

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
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task5/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task5/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # prepare results
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        x[x==1] = 63
        x[x==2] = 126
        x[x==3] = 189
        x[x==4] = 252

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task5/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x[x==1] = 63
        x[x==2] = 126
        x[x==3] = 189
        x[x==4] = 252

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
    # create empty templates
    for i in range(1,41):
        os.makedirs(f"{OUTDIR}/Task3/MR/{i}/T1DUAL/Results/", exist_ok=True)
        os.makedirs(f"{OUTDIR}/Task3/MR/{i}/T2SPIR/Results/", exist_ok=True)
    # prepare results
    for i in trange(len(t1)):
        setnumber = t1[i].split('_')[1].split('/')[1]

        x = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))

        x[x==1] = 63
        x[x==2] = 0
        x[x==3] = 0
        x[x==4] = 0

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task3/MR/{setnumber}/T1DUAL/Results/img{z:03}.png', s)

        x = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        x[x==1] = 63
        x[x==2] = 0
        x[x==3] = 0
        x[x==4] = 0

        for z in range(x.shape[0]):
            s = x[z,:,:]
            cv2.imwrite(f'{OUTDIR}/Task3/MR/{setnumber}/T2SPIR/Results/img{z:03}.png', s)


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()

