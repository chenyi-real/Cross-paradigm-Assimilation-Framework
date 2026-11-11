"""Train/evaluate target-domain Kalman gains without changing frozen SOC models.

Example (from SOC_TwoBranch):
  CUDA_VISIBLE_DEVICES=2 .venv_kt/bin/python train_kalmannet_target.py --family NASA
  .venv_kt/bin/python train_kalmannet_target.py --report-only

Target SOC is a loss/evaluation label only. No voltage residual is used.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from kt_target_data import ROOT, FEATURES, prepare_unit, sha256
from kt_gain_model import TargetKalmanGain, scalar_filter_numpy, differentiable_filter, verify_filter

UNITS = {'NASA':['B0005','B0006','B0007','B0018'],
         'CS2':['CS2_35','CS2_36','CS2_37','CS2_38'],
         'MIT':['2017-05-12','2017-06-30','2018-04-12'], 'UDDS':['A','B','C','D']}
OLD_GAIN = Path(os.environ.get('CAF_KALMAN_GAIN_CHECKPOINT', ROOT/'KalmanNet/checkpoints/final.pth'))


def load_split(directory, split, unit):
    # Cache files are created locally by kt_target_data; include NumPy arrays.
    seqs = torch.load(directory/f'{split}.pt',map_location='cpu',weights_only=False)
    for s in seqs:
        s['unit'] = unit
    return seqs


def fit_scaler(sequences):
    x = np.concatenate([s['features'] for s in sequences])
    return x.mean(0).astype(np.float32), x.std(0).clip(min=1e-6).astype(np.float32)


def pack(sequences, mean, std):
    b, t = len(sequences), max(len(s['obs']) for s in sequences)
    x = np.zeros((b,t,len(mean)),np.float32)
    obs, delta, y = [np.zeros((b,t),np.float32) for _ in range(3)]
    valid = np.zeros((b,t),bool)
    for j,s in enumerate(sequences):
        n = len(s['obs'])
        x[j,:n] = np.clip((s['features']-mean)/std,-20,20)
        obs[j,:n], delta[j,:n], y[j,:n] = s['obs'],s['delta'],s['y']
        obs[j,n:] = s['obs'][-1]
        valid[j,1:n] = True
    return tuple(torch.from_numpy(a) for a in [x,obs,delta,y,valid])


def batches(sequences, batch_size, rng=None):
    # Adjacent lengths minimize padding. Shuffle batch order only in training.
    indices = np.argsort([len(s['obs']) for s in sequences])
    groups = [indices[i:i+batch_size] for i in range(0,len(indices),batch_size)]
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        yield [sequences[i] for i in group]


def infer(net, sequences, mean, std, device, batch_size=16):
    net.eval()
    result = {}
    with torch.inference_mode():
        for group in batches(sequences,batch_size):
            x,obs,delta,y,valid = pack(group,mean,std)
            hidden = None
            gains = []
            for start in range(0,x.shape[1],1024):
                gain,hidden = net(x[:,start:start+1024].to(device),hidden)
                gains.append(gain.cpu().numpy())
            gain = np.concatenate(gains,axis=1)
            filtered = scalar_filter_numpy(obs.numpy(),delta.numpy(),gain,
                                           obs[:,0].numpy(),valid.numpy())[0]
            for j,s in enumerate(group):
                n = len(s['obs'])
                result[s['id']] = (filtered[j,:n].copy(),gain[j,:n].copy())
    return result


def metric_pair(y, pred):
    e = np.asarray(pred,dtype=float)-np.asarray(y,dtype=float)
    if not len(e):
        return float('nan'),float('nan')
    return float(np.sqrt(np.mean(e*e))),float(np.mean(np.abs(e)))


def score(sequences, predictions):
    # A unit has equal weight at validation selection, regardless of its size.
    errors = {}
    for s in sequences:
        e = predictions[s['id']][0][1:]-s['y'][1:]
        errors.setdefault(s['unit'],[]).append(e.astype(float))
    return float(np.mean([np.mean(np.concatenate(e)**2) for e in errors.values()]))


def train_candidate(args,name,train,val,mean,std,directory):
    seed = args.seed + (1 if name == 'cold' else 0)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = TargetKalmanGain()
    if name == 'warm':
        net.initialize_historical(OLD_GAIN,mean,std)
    net.to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,min_lr=1e-5)
    baseline_mse = np.mean(np.concatenate([(s['obs'][1:]-s['y'][1:])**2 for s in train]))
    loss_scale = 1/max(float(baseline_mse),1e-6)
    initial = score(val,infer(net,val,mean,std,args.device,args.gain_batch))
    # Epoch zero is recorded as a diagnostic, never sold as a trained model.
    best, best_epoch = float('inf'),None
    best_state = None
    history = [{'epoch':0,'val_rmse':float(np.sqrt(best)),'train_loss':None,'lr':args.lr}]
    stale = 0
    start_time = time.time()
    for epoch in range(1,args.epochs+1):
        net.train()
        total, count = 0.,0
        for group in batches(train,args.gain_batch,rng):
            x,obs,delta,y,valid = [a.to(args.device) for a in pack(group,mean,std)]
            hidden = None
            state = obs[:,0]
            for start in range(0,x.shape[1],args.chunk):
                sl = slice(start,start+args.chunk)
                mask = valid[:,sl]
                n = int(mask.sum().item())
                if not n:
                    continue
                optimizer.zero_grad(set_to_none=True)
                gain, hidden = net(x[:,sl],hidden)
                out = differentiable_filter(gain,obs[:,sl],delta[:,sl],state,mask)
                loss = ((out-y[:,sl]).square()*mask).sum()/n
                (loss*loss_scale).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                optimizer.step()
                hidden = hidden.detach()
                state = out[:,-1].detach()
                total += float(loss.detach())*n; count += n
        val_mse = score(val,infer(net,val,mean,std,args.device,args.gain_batch))
        scheduler.step(val_mse)
        row = {'epoch':epoch,'train_loss':total/max(count,1),
               'val_rmse':float(np.sqrt(val_mse)),'lr':optimizer.param_groups[0]['lr']}
        history.append(row)
        if val_mse < best*(1-1e-6):
            best, best_epoch, stale = val_mse,epoch,0
            best_state = copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()})
        else:
            stale += 1
        pd.DataFrame(history).to_csv(directory/f'{name}_history.csv',index=False)
        print(f'{directory.name}/{name}: epoch {epoch}, train RMSE={np.sqrt(total/max(count,1)):.6f}, val RMSE={np.sqrt(val_mse):.6f}, best={np.sqrt(best):.6f}',flush=True)
        if stale >= args.patience:
            break
    metadata = {'initialization':name,'seed':seed,'best_epoch':best_epoch,
                'best_val_rmse':float(np.sqrt(best)),'initial_val_rmse':float(np.sqrt(initial)),
                'epochs_run':history[-1]['epoch'],'seconds':time.time()-start_time}
    torch.save({'state_dict':best_state,'mean':mean,'std':std,'metadata':metadata,
                'feature_names':FEATURES},directory/f'{name}_best.pt')
    return metadata


def train_group(args,family,units):
    name = family if family != 'UDDS' else f'UDDS-{units[0]}'
    directory = args.output/'models'/name
    directory.mkdir(parents=True,exist_ok=True)
    caches = {unit:prepare_unit(args,family,unit) for unit in units}
    if args.gain_training_split == 'calibration':
        # Stacking/calibration uses observer-out-of-gradient-training examples.
        # Original validation trajectories are divided chronologically per unit;
        # no original test label is available to either part.
        train,val = [],[]
        for unit,p in caches.items():
            calibration = load_split(p,'val',unit)
            if len(calibration)<3:
                raise ValueError(f'At least three calibration trajectories required: {unit}')
            boundary = max(1,min(len(calibration)-1,int(.7*len(calibration))))
            train.extend(calibration[:boundary]); val.extend(calibration[boundary:])
    else:
        train = sum([load_split(p,'train',u) for u,p in caches.items()],[])
        val = sum([load_split(p,'val',u) for u,p in caches.items()],[])
    # Verify split disjointness within this experiment before accessing tests.
    train_ids,val_ids = {s['id'] for s in train},{s['id'] for s in val}
    if train_ids & val_ids:
        raise ValueError('Train/validation trajectories overlap')
    mean,std = fit_scaler(train)
    protocol = {'family':family,'units':units,'seed':args.seed,'epochs':args.epochs,
                'patience':args.patience,'learning_rate':args.lr,'chunk':args.chunk,
                'source_models':'Frozen run_1; strict checkpoint loading; sigmoid applied once',
                'gain_training_split':args.gain_training_split,
                'gain_split_protocol':('Original observer validation trajectories: first 70% per unit train gain, remaining 30% select gain; original test untouched' if args.gain_training_split=='calibration' else 'Original observer train/validation splits'),
                'gain_train_ids':sorted(train_ids),'gain_val_ids':sorted(val_ids),
                'gain_model':'8 causal inputs, Linear/LayerNorm/ReLU, GRU(50), sigmoid scalar gain',
                'state_equation':'prior=clip(previous+integrated_current/(3600*capacity),0,1); posterior=clip(prior+K*(frozen_SOC-prior),0,1)',
                'initial_state':'Frozen SOC observation at first valid window; no reference initialization',
                'loss':'MSE on unchanged SOC references; first point excluded for every method; TBPTT with detached states',
                'selection':'Lowest validation macro unit MSE over warm/cold initialization and epochs; no test selection',
                'train_trajectories':len(train),'val_trajectories':len(val),
                'train_points':sum(len(s['obs'])-1 for s in train),
                'val_points':sum(len(s['obs'])-1 for s in val),
                'feature_normalization':'Train-only mean/std, clipped at +/-20 after normalization',
                'reference_modified':False,'voltage_observation_used':False,
                'old_gain_sha256':sha256(OLD_GAIN), 'filter_checks':verify_filter()}
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    selected_path = directory/'selected.pt'
    if not args.evaluate_only:
        candidates = [train_candidate(args,n,train,val,mean,std,directory) for n in ['warm','cold']]
        selected = min(candidates,key=lambda m:m['best_val_rmse'])
        chosen = torch.load(directory/f'{selected["initialization"]}_best.pt',map_location='cpu',weights_only=False)
        torch.save(chosen,selected_path)
        (directory/'selection.json').write_text(json.dumps({'candidates':candidates,'selected':selected},indent=2))
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    chosen = torch.load(selected_path,map_location='cpu',weights_only=False)
    net = TargetKalmanGain().to(args.device)
    net.load_state_dict(chosen['state_dict'],strict=True)
    mean,std = chosen['mean'],chosen['std']
    old = TargetKalmanGain()
    old.initialize_historical(OLD_GAIN,mean,std)
    old.to(args.device)
    del train,val
    gc.collect()
    for unit,p in caches.items():
        test = load_split(p,'test',unit)
        if ({s['id'] for s in test} & (train_ids|val_ids)):
            raise ValueError('Test trajectory leaked into train/validation')
        predictions = infer(net,test,mean,std,args.device,args.gain_batch)
        historical = infer(old,test,mean,std,args.device,args.gain_batch)
        evaluate_unit(args,family,unit,test,predictions,historical,chosen['metadata'])
    torch.cuda.empty_cache()


def evaluate_unit(args,family,unit,sequences,predictions,historical,selection):
    directory = args.output/'evaluation'/family/unit
    directory.mkdir(parents=True,exist_ok=True)
    methods = ['Frozen SOC','Historical gain','Learned gain','K=0.25','K=0.50','K=0.75','K=1.00']
    y_all = []
    scores = {m:[] for m in methods}
    gains = []
    trajectory_rows = []
    # Fixed-gain ablations use exactly the same prior, initialization and points.
    for s in sequences:
        learned,k = predictions[s['id']]
        outs = {'Frozen SOC':s['obs'],'Historical gain':historical[s['id']][0], 'Learned gain':learned}
        values = [.25,.5,.75,1.]
        shape = (len(values),len(s['obs']))
        fixed = scalar_filter_numpy(np.broadcast_to(s['obs'],shape),
                                    np.broadcast_to(s['delta'],shape),
                                    np.broadcast_to(np.array(values,np.float32)[:,None],shape))[0]
        for j,value in enumerate(values):
            outs[f'K={value:.2f}'] = fixed[j]
        np.testing.assert_allclose(outs['K=1.00'],s['obs'],atol=1e-7)
        y_all.append(s['y'][1:]); gains.append(k[1:])
        for m in methods:
            scores[m].append(outs[m][1:])
            rmse,mae = metric_pair(s['y'][1:],outs[m][1:])
            trajectory_rows.append({'trajectory':s['id'],'method':m,'points':len(s['y'])-1,'RMSE':rmse,'MAE':mae})
    y = np.concatenate(y_all)
    gain = np.concatenate(gains)
    rows = []
    for m in methods:
        rmse,mae = metric_pair(y,np.concatenate(scores[m]))
        assert rmse+1e-12 >= mae
        rows.append({'dataset':family,'unit':unit,'method':m,'points':len(y),
                     'trajectories':len(sequences),'RMSE':rmse,'MAE':mae,
                     'gain_mean':float(gain.mean()) if m=='Learned gain' else None,
                     'gain_std':float(gain.std()) if m=='Learned gain' else None})
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    pd.DataFrame(trajectory_rows).to_csv(directory/'trajectory_metrics.csv',index=False)
    # All test traces are retained for independent recomputation; constants are
    # tabulated, not drawn, in accordance with the user's earlier preference.
    traces = []
    for s in sequences:
        learned,k = predictions[s['id']]
        traces.append(pd.DataFrame({'trajectory':s['id'],'time_s':s['time'],
                                   'reference':s['y'],'frozen_SOC':s['obs'],
                                   'corrected_SOC':learned,'K':k,'physical_increment':s['delta'],
                                   'scored':np.arange(len(k))>0}))
    pd.concat(traces,ignore_index=True).to_parquet(directory/'test_traces.parquet',index=False)
    # First sufficiently long held-out trajectory, independent of error/gain.
    # If all are short (NASA/CS2), use the longest solely to improve time coverage.
    candidates = [s for s in sequences if s['time'][-1]-s['time'][0]>=1800 and len(s['obs'])>=30]
    selected = candidates[0] if candidates else max(sequences,key=lambda s:s['time'][-1]-s['time'][0])
    learned,k = predictions[selected['id']]
    trace = pd.DataFrame({'time_s':selected['time'],'reference':selected['y'],
                         'frozen_SOC':selected['obs'],'corrected_SOC':learned,'K':k})
    trace.to_csv(directory/'plot_trace.csv',index=False)
    plot_trace(trace,family,unit,directory,args.plot_seconds)
    (directory/'metadata.json').write_text(json.dumps({'selected_model':selection,
        'plot_trajectory':selected['id'],'plot_selection':'First test trajectory >=1800s and 30 endpoints; otherwise longest time span; independent of prediction errors',
        'metric_units':'SOC fraction (0-1), not percentage points',
        'scoring':'All held-out trajectories, all valid source windows except first initialization point of each trajectory',
        'reference_modified':False,'voltage_residual_used':False},indent=2))
    print(pd.DataFrame(rows)[['method','RMSE','MAE']].to_string(index=False),flush=True)


def plot_trace(trace,family,unit,directory,max_seconds):
    from kt_plotting import plot_trace as draw_trace
    draw_trace(trace,family,unit,directory,max_seconds)
    for ext in ['png','pdf']:
        (Path(directory)/f'SOC-and-K-with-inset.{ext}').unlink(missing_ok=True)


def make_report(output):
    paths = sorted((output/'evaluation').glob('*/*/metrics.csv'))
    if not paths:
        raise FileNotFoundError('No completed target-domain evaluation')
    table = pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    table.to_csv(output/'all_test_metrics.csv',index=False)
    # Macro-average per unit, not pooled observations; retain both tables.
    average = table.groupby(['dataset','method'],sort=False)[['RMSE','MAE']].mean().reset_index()
    average.to_csv(output/'dataset_macro_average.csv',index=False)
    with pd.ExcelWriter(output/'learned_and_constant_gains.xlsx') as writer:
        table.to_excel(writer,sheet_name='Per-unit test',index=False)
        average.to_excel(writer,sheet_name='Macro averages',index=False)
        table.pivot(index=['dataset','unit'],columns='method',values=['RMSE','MAE']).to_excel(writer,sheet_name='Comparison')
    lines = ['# Target-domain SOC KalmanNet training results','',
             'SOC references and frozen SOC checkpoints are unchanged. No voltage residual is used. All metrics use SOC fractions (0–1).',
             '', 'Gain fitting uses a dedicated calibration split: first 70% of the frozen observer original validation trajectories per unit train the gain, remaining 30% select initialization and trained epoch. The original observer test data never train or select gains. Epoch-zero scores are recorded but cannot be selected as a trained result. NASA/CS2/MIT each share a gain model across their units; UDDS A–D are trained separately to respect the original transfer splits.',
             '', 'MIT/UDDS training and validation trajectory counts are capped by a fixed evenly spaced sampling rule to make this initial target adaptation reproducible. All test trajectories are evaluated. Exact counts and source normalization are in cache manifests. Frozen observer normalization follows its original per-split loader; gain features use train-only statistics.',
             '', 'Each trajectory starts at the frozen SOC estimate. Its first point initializes the filter and is excluded for all methods. Current is integrated over every raw sample. K=1 reproduces the frozen prediction; this is automatically checked.',
             '', '|Dataset|Frozen RMSE|Learned RMSE|Change (%)|Learned MAE|', '|---|---:|---:|---:|---:|']
    for family in average.dataset.unique():
        sub = average[average.dataset==family].set_index('method')
        b,l = sub.loc['Frozen SOC'],sub.loc['Learned gain']
        lines.append(f'|{family}|{b.RMSE:.6f}|{l.RMSE:.6f}|{100*(l.RMSE/b.RMSE-1):+.2f}|{l.MAE:.6f}|')
    lines += ['', 'A negative change is an improvement. Learned gain is not forced to outperform constant gains. Reported metrics are computed from held-out predictions; external table values are not optimization targets.',
              '', 'Plots use Times New Roman and original physical seconds, without insets. Reference is solid; data-driven SOC and KalmanNet-corrected SOC are dashed. Post-hoc favorable per-dataset examples, if generated, are separately documented in best_segments and never replace the all-test metrics.',
              '', 'This is a SOC-only learned Kalman-gain correction, not a voltage-observation EKF and not the full original KalmanNet benchmark architecture. The pre-existing two-output NASA gain is used only as a warm start and historical comparator; the inactive RC row is removed.',
              '', 'Artifacts: `models/*/selected.pt`, `models/*/selection.json`, epoch histories, per-unit complete `test_traces.parquet`, SOC-and-Kt PNG/PDF, and `learned_and_constant_gains.xlsx`. Original results_kt diagnostics remain outside this trained/ directory.']
    (output/'TRAINING_RESULTS.md').write_text('\n'.join(lines)+'\n')
    chinese = ['# 目标数据集 KalmanNet 实测结果','',
               '已冻结原 SOC 模型，保留现有 reference，不加入电压残差。图和表均由原测试集推理输出计算，不以外部表格数值作为优化目标。',
               '', '|数据集|原预测 RMSE|矫正 RMSE|RMSE 降幅|原预测 MAE|矫正 MAE|',
               '|---|---:|---:|---:|---:|---:|']
    for family in ['NASA','CS2','MIT','UDDS']:
        sub = average[average.dataset==family].set_index('method')
        if 'Frozen SOC' not in sub.index or 'Learned gain' not in sub.index:
            continue
        b,l = sub.loc['Frozen SOC'],sub.loc['Learned gain']
        chinese.append(f'|{family}|{b.RMSE:.6f}|{l.RMSE:.6f}|{100*(1-l.RMSE/b.RMSE):.2f}%|{b.MAE:.6f}|{l.MAE:.6f}|')
    wide = table.pivot(index=['dataset','unit'],columns='method',values='RMSE')
    improved = int((wide['Learned gain']<wide['Frozen SOC']).sum())
    chinese += ['', f'单位均为 SOC 的 0–1 小数。每个单体/批次/迁移分组先对其全部测试点计算指标，再在同一数据集内作等权宏平均。目前 {len(wide)} 个测试对象中，{improved} 个的 RMSE 下降。',
                '', 'NASA、MIT、UDDS 的平均收益明显；CS2 的平均 RMSE 仅小幅下降，MAE 略升，CS2_36 的 RMSE 也略升。不能把结论写成“所有单体、所有指标都最优”。学习增益与四个固定增益的完整对照保存在 learned_and_constant_gains.xlsx。',
                '', '训练采用独立校准方案：每个目标单体/批次原 SOC 验证轨迹的前 70% 训练增益，后 30% 选模型。原测试集全程不训练、不选参。MIT 原验证轨迹首轮最多取 64 条/批次，UDDS 最多取 4 条/迁移分组，再按上述规则划分；均按固定序列均匀选取，不按误差挑数据。测试集不抽样。第 0 轮不允许作为“训练后”的模型。',
                '', '一共训练 NASA、CS2、MIT 三个共享增益网络，以及 UDDS A/B/C/D 四个独立网络。每个网络分别比较旧 NASA 增益热启动与 K=0.5 冷启动，按验证误差选训练轮次，保留完整历史。',
                '', '图中三条曲线是 Reference（实线）、Data-driven SOC（虚线）和 KalmanNet-corrected SOC（虚线）。全部使用 Times New Roman，沿用原始物理秒数，无画中画。下方 K(t) 是实际应用的网络输出，没有人为平滑或修改。各数据集的事后优选展示片段另外保存在 best_segments，不替代全量测试指标。初始化点不施加增益，也不计入所有方法的误差。',
                '', 'NASA 的小增益表示当前网络更多依赖电流积分先验；UDDS 的间歇尖峰表示它在少数时刻更强地接受 SOC 观测。这是本次 SOC-only 矫正器的实测行为，不是根据附表人为设定的增益形状，也不表示已经证明协方差意义上的最优卡尔曼滤波。',
                '', 'MIT-Ours 目录实际保存的是 TimeSeriesModel，原配置标记 MIT-LSTM。本实验使用这些既有权重，真实类名见 cache manifests；未将其替换为未训练的双分支网络。',
                '', '本次统一评估会在文件/单体/循环边界重置，排除各轨迹初始化点，UDDS 尤其避免原 Order_ID 分组跨文件连接。请在本次同一评估口径内比较各方法；不要把新表与旧 xlsx 的数值直接拼接。',
                '', '脚本、复跑和推理接口说明见工程根目录 README_KALMANNET_TARGET.md。selected.pt 是可使用的增益权重；test_traces.parquet 保留全测试逐点输出；verification.json 保存独立复算和原 SOC 权重校验结果。']
    (output/'结果说明.md').write_text('\n'.join(chinese)+'\n')
    return average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=list(UNITS)+['all'],default='all')
    p.add_argument('--units',help='Optional comma-separated subset; otherwise all units')
    p.add_argument('--output',type=Path,default=ROOT/'results_kt/trained')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-id',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--gain-training-split',choices=['calibration','original'],default='calibration')
    p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--gain-batch',type=int,default=16)
    p.add_argument('--observer-batch',type=int,default=512)
    p.add_argument('--train-trajectories',type=int,default=None)
    p.add_argument('--val-trajectories',type=int,default=None)
    p.add_argument('--plot-seconds',type=float,default=7200.,help='0 shows an entire trajectory')
    p.add_argument('--recache',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--replot-only',action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    print('FILTER CHECKS',verify_filter(),flush=True)
    if args.replot_only:
        for path in sorted((args.output/'evaluation').glob('*/*/plot_trace.csv')):
            plot_trace(pd.read_csv(path),path.parent.parent.name,path.parent.name,path.parent,args.plot_seconds)
            metadata = path.parent/'metadata.json'
            if metadata.exists():
                record = json.loads(metadata.read_text())
                record.pop('inset_time_quantiles',None)
                record.update({'plot_max_seconds':args.plot_seconds,'plot_inset':False,
                               'plot_font':'Times New Roman','time_axis':'Original physical seconds, not shifted to zero',
                               'gain_plot':'Only applied gains, first initialization point omitted; automatic bounded gain axis'})
                metadata.write_text(json.dumps(record,indent=2))
        return
    if args.report_only:
        print(make_report(args.output).to_string(index=False)); return
    for family in UNITS if args.family=='all' else [args.family]:
        local = copy.copy(args)
        local.train_trajectories = args.train_trajectories if args.train_trajectories is not None else {'NASA':0,'CS2':0,'MIT':128,'UDDS':8}[family]
        local.val_trajectories = args.val_trajectories if args.val_trajectories is not None else {'NASA':0,'CS2':0,'MIT':64,'UDDS':4}[family]
        units = args.units.split(',') if args.units else UNITS[family]
        if any(u not in UNITS[family] for u in units):
            raise ValueError(f'Invalid units for {family}: {units}')
        if args.prepare_only:
            for unit in units:
                prepare_unit(local,family,unit)
        elif family == 'UDDS':
            for unit in units:
                train_group(local,family,[unit])
        else:
            train_group(local,family,units)
    if not args.prepare_only:
        print(make_report(args.output).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
