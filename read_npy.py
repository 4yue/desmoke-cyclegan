import os
import numpy as np
import cv2
import torch

npy_path = 'H:\\Surgical-video\\cycle-dataset'

pred_clean = np.load(os.path.join(npy_path, 'pred_clean.npy'))
pred_smoke = np.load(os.path.join(npy_path, 'pred_smoke.npy'))
real_clean = np.load(os.path.join(npy_path, 'real_clean.npy'))
real_smoke = np.load(os.path.join(npy_path, 'real_smoke.npy'))

for i in range(pred_clean.shape[0]):
    p_clean, p_smoke, r_clean, r_smoke = pred_clean[i], pred_smoke[i], real_clean[i], real_smoke[i]
    p_clean = torch.from_numpy(p_clean).float().permute(1, 2, 0).numpy()
    p_smoke = torch.from_numpy(p_smoke).float().permute(1, 2, 0).numpy()
    r_clean = torch.from_numpy(r_clean).float().permute(1, 2, 0).numpy()
    r_smoke = torch.from_numpy(r_smoke).float().permute(1, 2, 0).numpy()

    img1 = np.concatenate([p_clean, p_smoke], axis=1)
    img2 = np.concatenate([r_clean, r_smoke], axis=1)
    img = np.concatenate([img1, img2], axis=0)

    print(img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(0)



