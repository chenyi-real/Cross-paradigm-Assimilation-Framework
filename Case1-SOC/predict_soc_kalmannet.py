"""Apply a trained SOC-only gain to frozen observations, without references.

The CSV interface expects one continuous trajectory with columns:
time_s, soc_observation, physical_increment, current_A.
physical_increment integrates ALL raw current samples between observation
endpoints and divides by 3600*capacity_Ah. It is not estimated from SOC labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from kt_target_data import gain_features
from kt_gain_model import TargetKalmanGain, scalar_filter_numpy


def correct_observations(observation, physical_increment, time_s, current_A,
                         checkpoint, device='cpu'):
    """Inference-only public interface; no SOC reference or voltage parameter."""
    observation,physical_increment,time_s,current_A = [np.asarray(a) for a in
        [observation,physical_increment,time_s,current_A]]
    if not len(observation) or not (len(observation)==len(physical_increment)==len(time_s)==len(current_A)):
        raise ValueError('Inputs must be nonempty and aligned')
    if not np.isfinite(np.concatenate([observation,physical_increment,time_s,current_A])).all():
        raise ValueError('Inputs must be finite')
    if (np.diff(time_s)<0).any():
        raise ValueError('One continuous trajectory required; reset at time/file/cycle boundaries')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    net = TargetKalmanGain().to(device).eval()
    net.load_state_dict(saved['state_dict'],strict=True)
    dt = np.r_[0.,np.diff(time_s)]
    raw = gain_features(observation,physical_increment,dt,current_A,time_s)
    x = np.clip((raw-saved['mean'])/saved['std'],-20,20)
    gains,hidden = [],None
    with torch.inference_mode():
        for start in range(0,len(x),1024):
            k,hidden = net(torch.as_tensor(x[None,start:start+1024],device=device),hidden)
            gains.append(k[0].cpu().numpy())
    gain = np.concatenate(gains)
    correction = scalar_filter_numpy(observation.astype(np.float32),
                                     physical_increment.astype(np.float32),gain)[0][0]
    return correction,gain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a = p.parse_args()
    f = pd.read_csv(a.input)
    corrected,gain = correct_observations(f.soc_observation,f.physical_increment,
                                         f.time_s,f.current_A,a.checkpoint,a.device)
    f['corrected_SOC'],f['K'] = corrected,gain
    a.output.parent.mkdir(parents=True,exist_ok=True)
    f.to_csv(a.output,index=False)


if __name__=='__main__':
    main()
