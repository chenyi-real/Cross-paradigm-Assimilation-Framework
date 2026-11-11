"""Independent checks of saved target-gain results, with no training."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from kt_gain_model import verify_filter
from kt_target_data import ROOT,sha256
from predict_soc_kalmannet import correct_observations


def verify(output):
    torch.set_num_threads(4)
    checks = verify_filter()
    weights,units,plots,points = {},0,0,0
    expected = {'NASA':4,'CS2':4,'MIT':3,'UDDS':4}
    for family,n in expected.items():
        directories = sorted((output/'evaluation'/family).glob('*/metrics.csv'))
        assert len(directories)==n,(family,len(directories),n)
        for path in directories:
            unit = path.parent.name
            frame = pd.read_parquet(path.parent/'test_traces.parquet')
            metric = pd.read_csv(path)
            scored = frame[frame.scored]
            for name,col in [('Frozen SOC','frozen_SOC'),('Learned gain','corrected_SOC')]:
                e = scored[col].to_numpy(float)-scored.reference.to_numpy(float)
                actual = np.array([np.sqrt(np.mean(e*e)),np.mean(np.abs(e))])
                expected_metric = metric[metric.method==name][['RMSE','MAE']].iloc[0].to_numpy(float)
                np.testing.assert_allclose(actual,expected_metric,atol=1e-12,rtol=1e-10)
            assert ((metric.RMSE>=metric.MAE-1e-12) & np.isfinite(metric.RMSE) & np.isfinite(metric.MAE)).all()
            assert frame[['reference','frozen_SOC','corrected_SOC','K']].notna().all().all()
            assert ((frame.K>=0)&(frame.K<=1)).all()
            assert ((frame.corrected_SOC>=0)&(frame.corrected_SOC<=1)).all()
            assert len(scored)==int(metric.points.iloc[0])
            for ext in ['png','pdf']:
                assert (path.parent/f'SOC-and-Kt.{ext}').stat().st_size>1000
                plots += 1
            meta = json.loads((output/'cache'/family/unit/'test.json').read_text())
            weights[meta['source_checkpoint']] = meta['source_sha256']
            points += len(scored); units += 1
    for path,h in weights.items():
        assert sha256(path)==h, f'Source checkpoint modified: {path}'
    models = sorted((output/'models').glob('*/selected.pt'))
    assert len(models)==7
    for model in models:
        protocol = json.loads((model.parent/'protocol.json').read_text())
        a,b = set(protocol['gain_train_ids']),set(protocol['gain_val_ids'])
        assert not a.intersection(b)
        chosen = torch.load(model,map_location='cpu',weights_only=False)
        assert chosen['metadata']['best_epoch']>=1
    # Reference-free public inference must reproduce an independently saved
    # trajectory, and a future suffix must not change a prefix gain/correction.
    family,unit = 'NASA','B0005'
    seq = torch.load(output/'cache'/family/unit/'test.pt',weights_only=False,map_location='cpu')[0]
    checkpoint = output/'models'/family/'selected.pt'
    corrected,k = correct_observations(seq['obs'],seq['delta'],seq['time'],seq['current'],checkpoint)
    traces = pd.read_parquet(output/'evaluation'/family/unit/'test_traces.parquet')
    saved = traces[traces.trajectory==seq['id']]
    np.testing.assert_allclose(corrected,saved.corrected_SOC,atol=3e-6,rtol=1e-5)
    np.testing.assert_allclose(k,saved.K,atol=2e-5,rtol=1e-4)
    boundary = max(2,len(k)//2)
    prefix,k_prefix = correct_observations(seq['obs'][:boundary],seq['delta'][:boundary],
                                            seq['time'][:boundary],seq['current'][:boundary],checkpoint)
    np.testing.assert_allclose(prefix,corrected[:boundary],atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(k_prefix,k[:boundary],atol=1e-7,rtol=1e-6)
    checks.update({'source_checkpoints_unchanged':len(weights),'completed_gain_models':len(models),
                   'evaluated_units':units,'scored_points':points,'plot_artifacts':plots,
                   'metrics_recomputed_from_all_saved_traces':'passed',
                   'gain_fit_and_selection_disjoint':'passed','trained_epoch_not_zero':'passed',
                   'reference_free_public_inference':'passed','causal_prefix_invariance':'passed'})
    (output/'verification.json').write_text(json.dumps(checks,indent=2))
    return checks


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results_kt/trained')
    args = p.parse_args()
    print(json.dumps(verify(args.output),indent=2))
