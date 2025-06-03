import torch
from torchsurv.metrics.cindex import ConcordanceIndex

import pandas as pd
import numpy as np

ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_train.csv"

df = pd.read_csv(ANNOTATION_DIR)

N = 49
R = 36

events = torch.tensor([1 if i<R else 0 for i in range(N) ], dtype=torch.bool)

times = torch.tensor(range(N), dtype=torch.float32)

dist_rec = torch.distributions.Uniform(2.1, 5.0)
dist_nonrec = torch.distributions.Uniform(0.0, 2.0)
estimates_rec = dist_rec.sample((R,))
estimates_nonrec = dist_nonrec.sample((N-R,))
# Combine the estimates
estimates = torch.cat((estimates_nonrec, estimates_rec), dim=0)

cindex_computer = ConcordanceIndex()

cindex = cindex_computer(estimate=estimates, event=events, time=times)

print(f"C-index: {(cindex):.4f}")

#%%
