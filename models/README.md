# models/

Each training run writes `models/<run_id>/`:

| File | Contents |
|---|---|
| `metadata.json` | algorithm, seed, reward weights, action space, library versions, status, best validation score |
| `progress.json` | live progress: reward curve, validation history, baseline scores on the validation set |
| `train.log` | human-readable training log |
| `best.zip` | checkpoint with the best validation objective (used by default) |
| `final.zip` | checkpoint at the end of training |
| `evaluation.json` | written by `python -m neuralos.rl.evaluate` |

Checkpoints are generated artifacts and are git-ignored. Recreate the reference model with:

```
python -m neuralos.rl.train --run-id ppo-main --timesteps 400000 --seed 0
```

Set `NEURALOS_MODELS_DIR` to keep models elsewhere.
