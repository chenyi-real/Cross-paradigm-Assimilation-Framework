"""Frozen SOC observations for target-domain gain training.

References and source checkpoints are read only. Every sequence is one physical
trajectory; normalization of the frozen observer follows its legacy data loader.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import model as soc_models
import dataloader as soc_data

ROOT = Path(__file__).resolve().parent
SHARED = Path(os.environ.get('CAF_DATASETS_ROOT', ROOT / 'data'))
FEATURES = ['observation_difference', 'classical_innovation', 'classical_update',
            'soc_observation', 'physical_soc_increment', 'log_dt', 'current_A',
            'elapsed_fraction']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def gain_features(obs, delta, dt, current, time):
    # Exactly the old F1/F2/F4 definitions, supplemented with causal context.
    pred = np.zeros(len(obs)); post = np.zeros(len(obs)); post[0] = pred[0] = obs[0]
    variance = 1.
    for i in range(1, len(obs)):
        pred[i] = post[i-1]
        variance += 1e-5
        k = variance / (variance + 1e-3)
        post[i] = pred[i] + k * (obs[i] - pred[i])
        variance *= 1-k
    # Elapsed time divided by a fixed hour: no future trajectory duration used.
    return np.column_stack([np.r_[0., np.diff(obs)], obs-pred, post-pred,
                            obs, delta, np.log1p(dt), current,
                            (time-time[0])/3600.]).astype(np.float32)


def checkpoint_for(family, unit, run):
    runs = Path(os.environ.get('CAF_SOC_RUNS_ROOT', ROOT / 'runs'))
    if family == 'UDDS':
        return runs / f'runs_transfer/MIT-Trans-UDDS-Ours/weights/run_{run}/{unit}/best_model_fc_finetuned.pth'
    prefix = {'NASA':'NASA', 'CS2':'CS2', 'MIT':'MIT'}[family]
    return runs / f'runs_baseline/{prefix}-Ours/weights/run_{run}/{unit}/best_model.pth'


def load_observer(path, output, device):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.nn.Module):
        net = obj
    else:
        output.mkdir(parents=True, exist_ok=True)
        net = soc_models.TwoBranchFramework(exp_dir=str(output))
        net.load_state_dict(obj.get('state_dict', obj), strict=True)
    return net.eval().requires_grad_(False).to(device)


def input_stats(frame, columns, factor=1.):
    x = frame[columns].copy()
    x[columns[0]] *= factor
    x = x.to_numpy(np.float32)
    return x.mean(0), x.std(0).clip(min=1e-6)


def split_indices(items, split):
    n = len(items)
    a, b = int(.7*n), int(.8*n)
    return {'train':items[:a], 'val':items[a:b], 'test':items[b:]}[split]


def even_select(items, limit):
    if not limit or len(items) <= limit:
        return items
    ids = np.linspace(0, len(items)-1, limit, dtype=int)
    return [items[i] for i in ids]


def sequence_predict(group, columns, factors, capacity, stats, net, device,
                     stride, batch_size, identity, predictions=None):
    time_col, voltage_col, current_col, ref_col = columns
    model_factor, seconds_factor, ref_factor = factors
    time = group[time_col].to_numpy(float) * seconds_factor
    current = group[current_col].to_numpy(float)
    # Sorting never joins trajectories. Duplicate timestamps are invalid for
    # physical integration and are rejected, not silently trimmed or repaired.
    if len(group) < 64:
        return None
    if not np.all(np.diff(time) >= 0):
        raise ValueError(f'Decreasing time in {identity}')
    ends = np.arange(63, len(group), stride)
    if predictions is None:
        x = group[[time_col, voltage_col, current_col]].copy()
        x[time_col] *= model_factor
        x = (x.to_numpy(np.float32)-stats[0])/stats[1]
        obs = []
        offsets = np.arange(-63, 1)
        with torch.inference_mode():
            for s in range(0, len(ends), batch_size):
                e = ends[s:s+batch_size]
                bx = np.ascontiguousarray(x[e[:, None]+offsets])
                logits = net(torch.from_numpy(bx).to(device))
                obs.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
        obs = np.concatenate(obs)
    else:
        obs = predictions
    t = time[ends]
    charge = np.cumsum(np.r_[0., .5*(current[1:]+current[:-1])*np.diff(time)])
    delta = np.r_[0., np.diff(charge[ends])]/(3600.*capacity)
    dt = np.r_[0., np.diff(t)]
    y = group[ref_col].to_numpy(np.float32)[ends]*ref_factor
    if not np.isfinite(np.r_[obs, y, delta, t]).all():
        raise ValueError(f'Nonfinite sequence: {identity}')
    return {'id':identity, 'obs':obs.astype(np.float32), 'y':y.astype(np.float32),
            'delta':delta.astype(np.float32), 'time':t, 'dt':dt,
            'current':current[ends].astype(np.float32), 'capacity':float(capacity),
            'features':gain_features(obs, delta, dt, current[ends], t)}


def batched_group_observations(frame,grouped,selected,columns,factor,stats,net,args,stride):
    """Batch short MIT/CS2 trajectories together without crossing boundaries."""
    group_indices = [grouped.indices[key] for key in selected]
    if any(not np.all(np.diff(i)==1) for i in group_indices):
        raise ValueError('Source windows require contiguous sorted trajectory groups')
    indices = [i[63::stride] for i in group_indices]
    ends = np.concatenate(indices)
    values = frame[columns[:3]].copy()
    values[columns[0]] *= factor
    x = (values.to_numpy(np.float32)-stats[0])/stats[1]
    predictions = []
    offsets = np.arange(-63,1)
    with torch.inference_mode():
        for start in range(0,len(ends),args.observer_batch):
            e = ends[start:start+args.observer_batch]
            bx = np.ascontiguousarray(x[e[:,None]+offsets])
            predictions.append(torch.sigmoid(net(torch.from_numpy(bx).to(args.device))).reshape(-1).cpu().numpy())
    flat = np.concatenate(predictions)
    cuts = np.cumsum([len(i) for i in indices])[:-1]
    return dict(zip(selected,np.split(flat,cuts)))


def ordinary_frame(family, unit, split):
    """Return a split and its physical grouping, matching the frozen observer."""
    if family == 'NASA':
        path = ROOT / f'data/NASA_csv/{unit}.csv'
        frame = pd.read_csv(path).sort_values(['Discharge_Cycle', 'Time(s)'])
        counts = frame.groupby('Discharge_Cycle').size()
        ids = split_indices(counts[counts >= 64].index.tolist(), split)
        frame = frame[frame.Discharge_Cycle.isin(ids)].copy()
        return frame, ['Discharge_Cycle'], ['Time(s)','Voltage(V)','Current(A)','SOC'], (1.,1.,1.), 'Capacity(Ah)', [str(path)]
    if family == 'CS2':
        path = SHARED / 'CS2_csv' / unit
        frame = soc_data.load_all_csv_in_folder(str(path))
        counts = frame.groupby('cycle').size()
        ids = split_indices(counts[counts >= 64].index.tolist(), split)
        frame = frame[frame.cycle.isin(ids)].copy().sort_values(['Date','cycle','time_s'])
        return frame, ['Date','cycle'], ['time_s','Voltage(V)','Current(A)','SOC(%)'], (1.,1.,.01), 'capacity_Ah', [str(path)]
    if family == 'MIT':
        path = SHARED / 'MIT/Processed_SOC' / unit
        files = sorted(path.glob('*.csv'), key=lambda p:int(re.search(r'battery[-_](\d+)',p.name)[1]))
        files = split_indices(files, split)
        tables = []
        for file in files:
            f = pd.read_csv(file, usecols=['cycle','t','V','I','SOC','cycle_capacity_used'])
            f['source_file'] = file.name
            tables.append(f)
        frame = pd.concat(tables, ignore_index=True).sort_values(['source_file','cycle','t'])
        return frame, ['source_file','cycle'], ['t','V','I','SOC'], (60.,3600.,1.), 'cycle_capacity_used', [str(p) for p in files]
    raise ValueError(family)


def prepare_unit(args, family, unit):
    directory = args.output / 'cache' / family / unit
    directory.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_for(family, unit, args.run_id)
    # Hash lookup prevents accidental reuse after a different source run.
    source_hash = sha256(ckpt)
    stride = getattr(args,'inference_stride',None) or (16 if family == 'MIT' else 8)
    if stride < 1:
        raise ValueError('Inference stride must be positive')
    net = None
    for split in getattr(args,'splits',['train','val','test']):
        destination = directory / f'{split}.pt'
        meta_path = directory / f'{split}.json'
        limit = args.train_trajectories if split == 'train' else args.val_trajectories if split == 'val' else 0
        if destination.exists() and meta_path.exists() and not args.recache:
            old = json.loads(meta_path.read_text())
            if old.get('cache_version') == 2 and old['source_sha256'] == source_hash and old['trajectory_limit'] == limit and old.get('stride') == stride:
                print(f'CACHE {family}/{unit}/{split}', flush=True)
                continue
        if net is None:
            net = load_observer(ckpt, directory/'observer_constructor', args.device)
        print(f'PREPARE {family}/{unit}/{split}', flush=True)
        if family != 'UDDS':
            frame, keys, columns, factors, cap_column, paths = ordinary_frame(family, unit, split)
            stats = input_stats(frame, columns[:3], factors[0])
            grouping = keys[0] if len(keys) == 1 else keys
            sizes = frame.groupby(grouping, sort=True).size()
            eligible = [key for key, n in sizes.items() if n >= 64]
            selected = even_select(eligible, limit)
            grouped = frame.groupby(grouping, sort=True)
            observations = batched_group_observations(frame,grouped,selected,columns,factors[0],stats,net,args,stride)
            sequences = []
            for i, key in enumerate(selected):
                # A contiguous iloc slice avoids PyArrow string-column take,
                # which otherwise scans huge chunks for every MIT cycle.
                positions = grouped.indices[key]
                group = frame.iloc[positions[0]:positions[-1]+1]
                cap = float(group[cap_column].median())
                s = sequence_predict(group, columns, factors, cap, stats, net,
                                     args.device, stride, args.observer_batch, f'{family}/{unit}/{key}',observations[key])
                if s is not None:
                    sequences.append(s)
                if (i+1)%100 == 0:
                    print(f'  {split}: {i+1}/{len(selected)} trajectories', flush=True)
            total_available = len(eligible)
            del frame, grouped, observations
        else:
            path = ROOT / f'data/UDDS_SOC/combined_cache_{unit}_{split}.parquet'
            columns = ['Step_Time(s)','Voltage(V)','Current(A)','SOC']
            # The old loader groups Order_ID alone and can join different
            # files/cells. Physical recursion must reset at actual boundaries.
            frame = pd.read_parquet(path, columns=columns+['Order_ID','Cell_ID','Cycle_Number','Source_File'])
            frame = frame.sort_values(['Order_ID','Cell_ID','Cycle_Number','Step_Time(s)'])
            stats = input_stats(frame, columns[:3])
            grouping = ['Order_ID','Cell_ID','Cycle_Number','Source_File']
            grouped = frame.groupby(grouping,sort=True)
            keys = [key for key,n in grouped.size().items() if n>=64]
            selected = even_select(keys, limit)
            sequences = []
            for i, key in enumerate(selected):
                group = grouped.get_group(key)
                group = group.sort_values('Step_Time(s)')
                identity = f'UDDS/{unit}/{key}'
                s = sequence_predict(group, columns, (1.,1.,1.), 4.93, stats,
                                     net, args.device, stride, args.observer_batch, identity)
                if s is not None:
                    sequences.append(s)
                print(f'  {split}: {i+1}/{len(selected)} trajectories', flush=True)
            paths = [str(path)]
            total_available = len(keys)
            del frame, grouped
        if not sequences:
            raise ValueError(f'Empty split: {family}/{unit}/{split}')
        metadata = {'cache_version':2,'family':family, 'unit':unit, 'split':split,
                    'source_checkpoint':str(ckpt), 'source_sha256':source_hash,
                    'source_class':type(net).__name__, 'source_files':paths,
                    'source_normalization':'Legacy frozen observer: mean/std separately on original split; no labels',
                    'mean':stats[0].tolist(), 'std':stats[1].tolist(),
                    'window':64, 'stride':stride, 'trajectory_limit':limit,
                    'available_trajectories':total_available,
                    'selected_trajectories':len(sequences), 'points':sum(len(s['obs']) for s in sequences),
                    'trajectory_selection':'Evenly spaced sorted trajectory IDs, independently of labels/errors; test uncapped',
                    'reference':'Original SOC column unchanged; CS2 percent divided by 100',
                    'integration':'Signed trapezoidal raw-current integration, per-cycle capacity; UDDS original 4.93Ah',
                    'boundaries':'Never cross file/cell/cycle; UDDS old Order_ID-only grouping is refined. Duplicate timestamps retained, zero charge increment over zero elapsed time.',
                    'features':FEATURES}
        torch.save(sequences, destination)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f'SAVED {family}/{unit}/{split}: {metadata["points"]} points', flush=True)
        del sequences
        gc.collect()
    del net
    torch.cuda.empty_cache()
    return directory
