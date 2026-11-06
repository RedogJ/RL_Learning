# Critic terminal mask: isolated validation

The production change keeps steps whose *current* reference frame is below its episode limit. Thus the transition from limit-1 to limit remains; imagined states at limit and later are excluded from critic fitting. The training lambda-return, soft failure weighting, Actor, world model, optimizer hyperparameters and existing checkpoints are unchanged. `reference_critic_mask` boundary check passed for both one-step-before-end and end cases. Python compilation passed.

Validation starts from v19 cycle 3280, saved Actor/WM/termination/Critic and Adam state. Frozen Actor/WM/termination models generate 256 real on-policy reference RSI reset episodes (observation noise on, no action noise), with 768 sampled alive states. The imagined 16-step windows and lambda targets are generated once, shared by both conditions. Train/holdout split is by env id 0–191 / 192–255, yielding 544 / 224 initial states. Independent Critic and Adam copies each run 5 epochs x 4 minibatches; unmasked uses 8704 imagined states, masked uses 7070. Both targets are kept fixed. Held-out real return is one actual stochastic-observation trajectory, not the exact conditional expected value. 224 states include temporal overlaps and are not independent episodes. No production logs, checkpoints, actor, or WM were modified.

| Metric, held-out starts | Before | Unmasked | Masked |
|---|---:|---:|---:|
| Critic vs actual discounted return MAE | 1.1425 | 1.1526 | 0.9767 |
| Critic vs actual discounted return RMSE | 1.4861 | 1.5378 | 1.3196 |
| Critic vs frozen imagined λ target MSE | 1.6600 | 1.5578 | 1.1815 |

One run supports improvement in this sample but does not establish on-policy long-term policy improvement or exact expected-value accuracy; training samples and minibatch composition differ inherently. There was no full training-cycle or continued-policy rollout in this validation. Post-hoc uncertainty, other seeds/checkpoints, true replay sampling and soft failure survival were not isolated. Raw data: results.json. Script writes its output to /tmp.
