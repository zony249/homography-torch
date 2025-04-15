import numpy as np 
import pandas as pd 
import os 
import sys 
from copy import deepcopy


if __name__ == "__main__": 

    if len(sys.argv) < 3: 
        print("Usage: python3 compute_rmse.py <target.tsv> <inputs.tsv>")
    
    target_filename = sys.argv[1] 
    input_filename = sys.argv[2] 

    targets = pd.read_csv(target_filename, delimiter="\t| ", engine="python")
    inputs = pd.read_csv(input_filename, delimiter="\t")


    targets = targets.iloc[:, 1:].to_numpy()
    inputs = inputs.iloc[:, 1:].to_numpy() 

    print(targets)
    print(inputs)

    N = inputs.shape[0] 

    per_frame_result = np.sqrt(((targets[:N] - inputs)**2).sum(axis=1) / 4)

    with open("per_frame_rmse.tsv", "w") as f:
        f.write(f"Frame\tRMSE\n")
        frame_no = 1
        for x in per_frame_result: 
            f.write(f"{frame_no}\t{x:.4f}\n")
            frame_no += 1 


    mask = per_frame_result <= 5
    pfr_less_thresh = per_frame_result[mask] 
    overall_success = mask.sum() / N 
    average_drift = pfr_less_thresh.mean()
    
    print("Mean RMSE:", per_frame_result.mean())
    print("Overall Success:", overall_success)
    print("Average Drift:", average_drift)



