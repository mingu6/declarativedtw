# Declarative DTW (DecDTW)

This repo contains a reference implementation of our proposed declarative DTW layer named DecDTW (ICLR 2023). The repo contains a full implementation of our DecDTW layer as well as training/eval pipelines for the two experiments presented in the paper.

## Setup

Make to the base directory of this repo your current working directory. Then run

```
pip install -r requirements.txt
pip install -e .
```

## Structure

### DecDTW layer usage

Sample usage:

```
import torch
from decdtw.utils import BatchedSignal
from decdtw.decdtw import DecDTWLayer

reg_lmbda = 0.1
decdtw = DecDTWLayer(subseq_enabled=True)

x = torch.randn(1, 8, 1)  # BxNxD
y = torch.randn(1, 5, 1)  # BxMxD
tx = torch.linspace(0., 1., 8).unsqueeze(0)  # BxN
ty = torch.linspace(0., 1., 5).unsqueeze(0)  # BxM
sx = BatchedSignal(x, times=tx)
sy = BatchedSignal(y, times=ty)

warp_fn = decdtw.forward(sx, sy, reg_lmbda)  # output is a BatchedSignal
warp_fn.values  # BxM
warp_fn.times  # BxM
```

`warp_fn` is the predicted piecewise-linear warp function using GDTW. `warp_fn.values` is differentiable w.r.t. `x` and `y` and contains the warp function values at knot points specified by `warp_fn.times`.

### Experiments

#### Audio to score alignment experiments

See `experiments/audio_score/README.md` for detailed setup information of the underlying datasets and terminal commands to run training/eval. 

#### Visual place recognition experiments

See `experiments/vpr/README.md` for detailed setup information of the underlying datasets and terminal commands to run training/eval. 
