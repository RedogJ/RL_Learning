# Critic audit, 2026-11-05

No model was trained or changed. Model v19 cycle 3280 was evaluated using MuJoCo Warp, with frozen Actor, dynamics, termination head, and Critic. 256 episodes were initialized from the saved adaptive RSI distribution (20% reference-start override for non-looping task), with actual observation noise, no action noise. 17 of 256 failed. All 9,310 pre-termination real transitions were eligible; 768 state/history pairs were randomly sampled for a 16-step imagined rollout. Because samples may share episodes, they are not 768 independent rollouts. This is a freshly collected on-policy approximation, not the lost historical replay buffer. Checkpoint context was fixed to 0.5. Real and imagined subsequent observation noise were sampled separately; Monte Carlo returns are one realized future, not exact expected values.

At the same sampled states, compute V, 16-step imagined λ-return with soft terminations, 16-step real λ-return with hard termination using the SAME frozen critic for bootstrapping, and realized full real discounted return. γ=.99, λ=.95. The last two λ returns share a critic, so their agreement cannot prove model accuracy or exclude common bootstrap bias.

| Comparison | Mean difference | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Critic minus imagined λ | -0.0883 | 0.9112 | 1.3096 |
| Critic minus actual realized MC | -0.0833 | 1.3210 | 1.8902 |
| Imagined λ minus real λ | -0.0407 | 0.1991 | 0.4684 |
| Real λ minus actual realized MC | +0.0457 | 0.7419 | 1.2282 |

Critic mean 6.906, real realized return mean 6.990: near-equal means do not imply accurate individual predictions. The three quantities have different noise/bootstrapping, and single-sample errors do not isolate irreducible observation-noise variability. The 3280 training log reported fixed-batch critic loss 1.0071 -> 0.9386: it improved within the batch but remained nonzero and is not a generalization or MC score.

For imagined states whose reference frame has already reached frame_limit BEFORE a step: 18.262% of 768×16 flattened training slots, and 36.458% of sampled windows contain at least one. In this particular reconstructed on-policy sample, these post-reference slots would still be passed to fit_critic by current state_train.py; their target mean was +0.4424 while critic mean was +1.2069. This is not a measurement of the historical replay distribution or proof of causal policy degradation. The correct transition entering terminal must remain; only later states are certainly post-terminal. Soft failure survival is a separate case requiring probability weighting, not naive hard thresholding.

Code evidence: state_train.py:618-621 zeros actor survival on reference end, but 622-624 keeps advancing imagined state; :643-658 flattens every state and trains critic unweighted, even after the reference end. An episode has already terminated in the real environment; computing a reward for subsequent imaginary steps creates targets for states outside that episode. A targeted correction should mask by deterministic reference-alive status before fitting, and can separately investigate survival-weighted soft failure; include entering-terminal transition, avoid changing real rewards/terminal head, and validate with held-out on-policy samples. Such an edit was not performed here.
