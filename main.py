import os 
import sys
import shutil
from typing import Dict, Tuple, List, Optional 

import numpy as np 
import matplotlib 
# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.nn import functional as F 
import cv2 

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from homography import Homography, Trainer, mse, contrastive_sim, dv_loss, histogram_mutual_information

def init_tracker(frame: np.ndarray, trainer: Trainer): 
    
    fh, fw, c = frame.shape


    # plt.imshow(frame[:, :, ::-1])
    points = [] #plt.ginput(4)

    if len(points) == 0: 
        points = [[1236, 540], [1245, 1404], [2200, 558], [2184, 1275]] # window on screen
        points = [[14, 303], [218, 290], [40, 560], [237, 532]] # cereal box
        points = [[297, 296], [465, 287], [312, 522], [470, 511]] # book 1
        points = [[264, 273], [423, 272],  [282, 488], [432, 494]] # book 2
    # plt.show()
    points = np.array(points)
    x = points[:, 0] # x 
    y = points[:, 1] # y

    # sorting the points into tl, tr, bl, br
    min_x = x.min() 
    min_y = y.min() 
    max_x = x.max() 
    max_y = y.max() 

    dist_to_tl = np.sqrt((x - min_x)**2 + (y - min_y)**2) 
    idx_tl = np.argmin(dist_to_tl) 
    dist_to_br = np.sqrt((x-max_x)**2 + (y - max_y)**2) 
    idx_br = np.argmin(dist_to_br) 

    dist_to_tr = np.sqrt((x - max_x)**2 + (y - min_y)**2) 
    idx_tr = np.argmin(dist_to_tr) 
    dist_to_bl = np.sqrt((x-min_x)**2 + (y - max_y)**2) 
    idx_bl = np.argmin(dist_to_bl) 

    # raw points
    tl = points[idx_tl]
    tr = points[idx_tr]
    bl = points[idx_bl]
    br = points[idx_br]
    # design points
    tl_rec = np.array([-1, -1], dtype=np.float32)
    tr_rec = np.array([1, -1], dtype=np.float32)
    bl_rec = np.array([-1, 1], dtype=np.float32)
    br_rec = np.array([1, 1], dtype=np.float32)

    dst_points = np.stack([tl_rec, tr_rec, bl_rec, br_rec], axis=0)
    src_points = np.stack([tl, tr, bl, br], axis=0).astype(np.float64)

    # scaling source points between -1 and 1
    src_points[:, 0] = 2*src_points[:, 0] / fw - 1 
    src_points[:, 1] = 2*src_points[:, 1] / fh - 1 

    # raw --> design
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0) 
    H = np.linalg.inv(H) # design --> raw
    H /= H[2, 2]

    test_points = np.concatenate([dst_points, np.ones_like(dst_points[:, -1:])], axis=1) 

    output_points = H @ test_points.transpose(1, 0) 
    output_points /= output_points[2:, :] 
    output_points = output_points[:2, :].transpose(1, 0) 

    output_points[:, 0] = (output_points[:, 0] + 1) * fw / 2
    output_points[:, 1] = (output_points[:, 1] + 1) * fh / 2

    output_points = [list(output_points[i].astype(int)) for i in range(output_points.shape[0])]

    # cv2.line(frame, output_points[0], output_points[1], (0, 255, 0), 5)
    # cv2.line(frame, output_points[1], output_points[3], (0, 255, 0), 5)
    # cv2.line(frame, output_points[3], output_points[2], (0, 255, 0), 5)
    # cv2.line(frame, output_points[2], output_points[0], (0, 255, 0), 5)

    # print("OUTPUT POINTS", output_points)

    # cv2.imshow("test homography visualization", frame)
    # plt.show()

    w = int(max(np.abs(tl[0] - tr[0]), np.abs(bl[0] - br[0])))
    h = int(max(np.abs(tl[1] - bl[1]), np.abs(tr[1] - br[1]))) 


    with torch.no_grad():
        x = torch.linspace(-1, 1, w, device=DEVICE)
        y = torch.linspace(-1, 1, h, device=DEVICE)

        xx, yy = torch.meshgrid(x, y, indexing="xy")

        design_grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2) 
        design_grid = torch.cat([design_grid, torch.ones_like(design_grid)[:, -1:]], dim=1)

        H_ref = torch.from_numpy(H).to(DEVICE).float()

        raw_grid = H_ref @ design_grid.T

        raw_grid /= raw_grid[-1:, :].clone()
        raw_grid = raw_grid[:2, :].T.reshape(h, w, 2)[None, ...]

        frame_torch = torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1)[None, ...].float()

        template = F.grid_sample(frame_torch, raw_grid, "bilinear", align_corners=False, padding_mode="zeros")

    template_np = template.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    # cv2.imshow("template", template_np)
    # plt.show()

    Hnet = trainer.solve_initial(H_ref, template)


    return Hnet, template_np



def main(filename): 

    cap = cv2.VideoCapture(filename)
    count = 0

    if os.path.isdir("tracked_regions"):
        shutil.rmtree("tracked_regions/")
        shutil.rmtree("frames/")
    os.makedirs("tracked_regions/", exist_ok=True)
    os.makedirs("frames/", exist_ok=True)

    H = Homography(features=16)
    trainer = Trainer(H, 
                      lr=3e-4, 
                      levels=3, 
                      steps_per_epoch=100, 
                      loss_fn=dv_loss)
    
    f = open("metrics.csv", "w") 
    f.write("Mutual information\n")

    while cap.isOpened():
        ret,frame = cap.read()

        frame = frame.astype(np.float32) / 255.
        fh, fw, c = frame.shape


        if count == 0: 
            H, template = init_tracker(frame, trainer)


        h, w, c = template.shape


        with torch.no_grad():
            x = torch.linspace(-1, 1, w, device=DEVICE)
            y = torch.linspace(-1, 1, h, device=DEVICE)

            xx, yy = torch.meshgrid(x, y, indexing="xy")

            design_grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2) 
            design_grid = torch.cat([design_grid, torch.ones_like(design_grid)[:, -1:]], dim=1)

            tracked, _ = H(torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1)[None, ...])

        tracked = tracked.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        # cv2.imshow("tracked before reg", tracked)

        
        tracked_w, H_mat = trainer.register(template, frame)
        tracked_w = tracked_w.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        # cv2.imshow("tracked after reg", tracked_w)

        # cv2.imshow('window-name', frame)
        cv2.imwrite(f"tracked_regions/frame{count:05d}.jpg", (tracked_w * 255.).astype(np.uint8)) 


        ##### VISUALIZE BOX #######

        # design points
        tl_rec = np.array([-1, -1], dtype=np.float32)
        tr_rec = np.array([1, -1], dtype=np.float32)
        bl_rec = np.array([-1, 1], dtype=np.float32)
        br_rec = np.array([1, 1], dtype=np.float32)

        dst_points = np.stack([tl_rec, tr_rec, bl_rec, br_rec], axis=0)

        # raw --> design
        test_points = np.concatenate([dst_points, np.ones_like(dst_points[:, -1:])], axis=1) 

        output_points = H_mat.detach().cpu().numpy() @ test_points.transpose(1, 0) 
        output_points /= output_points[2:, :] 
        output_points = output_points[:2, :].transpose(1, 0) 

        output_points[:, 0] = (output_points[:, 0] + 1) * fw / 2
        output_points[:, 1] = (output_points[:, 1] + 1) * fh / 2

        output_points = [list(output_points[i].astype(int)) for i in range(output_points.shape[0])]

        cv2.line(frame, output_points[0], output_points[1], (0, 1, 0), 5)
        cv2.line(frame, output_points[1], output_points[3], (0, 1, 0), 5)
        cv2.line(frame, output_points[3], output_points[2], (0, 1, 0), 5)
        cv2.line(frame, output_points[2], output_points[0], (0, 1, 0), 5)

        print("OUTPUT POINTS", output_points)

        # cv2.imshow("test homography visualization", frame)
        cv2.imwrite(f"frames/frame{count:05d}.jpg", (frame * 255.).astype(np.uint8))


        ## Compute MI score 

        mi = histogram_mutual_information(template, tracked_w)

        f.write(f"{mi}\n")









        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows
    f.close()


if __name__ == "__main__": 

    if len(sys.argv) > 1: 
        main(sys.argv[1]) 
    else: 
        print("Usage: python3 main.py <video_file_name>")