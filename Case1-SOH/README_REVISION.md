# SOH revision: implementation and dependencies

The existing implementation is in `Model/Model.py` and
`Model/Compare_Models.py`. This update retains class names and parameter keys
for checkpoint compatibility; documentation and output paths are normalized.
`Solution_u_debug` contains the CNN branch, `TSSNNGRU` branch, feature projection
and prediction head. `PINN_debug` adds the learned dynamical function and
physics-related training losses. The historical `debug` name is not a synthetic
data mode.

The S-GRU dependency is provided, with its original MIT license, under
`third_party/SeqSNN` at the repository root. From the repository root:

```bash
python -m pip install -e third_party/SeqSNN
```

Install a PyTorch build appropriate for the hardware. Existing root requirements
also apply. The vendored package has its own dependencies in `setup.py`.
No weights, health-indicator tables or experimental measurements are generated
by installing this dependency.

Run the MIT entry from `Case1-SOH` after preparing the processed data:

```bash
python main_MIT.py
```

The normalized MIT entry expects `data/MIT/HIs` and writes to `runs/MIT`.
Device visibility is controlled by the caller. The NASA entry is not part of
this update: its independent-test protocol needs author confirmation before
publishing a revised entry point.

## Items requiring author confirmation

The existing `main_CS2.py` selects `PINN_v2`, not `PINN_debug`. Both
`main_NASA.py` and `main_CS2.py` supply the same `valid_2` loader for validation
and testing. This release review does not
silently change the architecture or split, because that would change the
experiment. Confirm and publish the actual rerun entry used for the revised
NASA and CALCE results. A path-normalized NASA candidate was reviewed locally
but withheld from this upload; it is not a verified independent-test protocol.

The current review did not locate the trained parameter-matched Dual-ANN
control, all coordination replacement implementations, or a SOH
CNN-Transformer baseline matching the revised results. These must be supplied
by their actual experiment implementations, not reconstructed from table means.

The local `predict_soh_distill.py` is older than the public counterpart; it is
not included as an overwrite. Existing upstream source acknowledgements,
citations and licenses must remain in the repository.
