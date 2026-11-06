# Actor direction diagnostic, 2026-11-05

Checkpoints: v19 model_3280.pt as actor/WM/BCE/Critic base, model_3270.pt for actor direction only. Actor weights are `actor_3280 + alpha * (actor_3280 - actor_3270)`, alpha in {-0.25, 0, +0.25}. This direction is the difference of two checkpoints 10 cycles apart, not the gradient of one training update. WM, head and critic remain fixed; no training or file mutation in the project logs. 96 environments per seed, seeds 123 and 456, random reference-phase reset, observation and action noise off, identical random reset phases for each alpha. Context is fixed to 0.5. H=16, gamma=0.99. It is a reference-reset test, not a replay-distribution or arbitrary on-policy state test; repeated reset phases are not independent task variations.

Two experiments: closed_loop.py continues with the perturbed policy through the real episode; fixed_continuation.py changes back to base actor at real step 16, so the true tail uses the same continuation policy as the base critic intends to represent. All means below are 192 paired samples, unless specified. Soft imaginary reward includes survival weights; imaginary failure penalty is reported as a positive deduction. Real critic tail uses real state at step 16, with real survival mask and discount. The full imaginary objective is reward minus failure penalty plus critic tail. True short/full returns use hard terminations.

| Quantity | alpha -0.25 | alpha 0 | alpha +0.25 |
| --- | ---: | ---: | ---: |
| Imagined reward (first 16) | 5.97286 | 5.96911 | 5.96598 |
| Imagined failure deduction | 0.00101 | 0.00091 | 0.00087 |
| Imagined critic tail | 3.46539 | 3.48556 | 3.50498 |
| Imagined total | 9.43723 | 9.45376 | 9.47009 |
| Real 16-step reward (fixed continuation experiment) | 5.99256 | 5.98761 | 5.98169 |
| Real full return, fixed continuation | 9.98915 | 9.99005 | 10.09218 |
| Failure count, fixed continuation | 5 | 5 | 0 |

For -0.25 -> 0, imagined objective +0.01653, of which imagined short reward -0.00375, failure deduction reduction +0.000097, and critic tail +0.02018. Real full return +0.00089 (two seeds individually -0.00558 and +0.00736), thus essentially no consistent observed improvement. Real critic tail increases +0.02090 on real states, while realized real tail increases +0.00585 on all samples. For the 187 paired samples where neither policy failed, imagined objective +0.01633 versus real return -0.01325; this is conditional post-selection, not overall policy performance.

For 0 -> +0.25, imagined objective +0.01632, real full return +0.10214. Five initial states fail under alpha 0 and not under +0.25, so this mean includes a meaningful failure event change. On the 187 pairs where neither fails, real return -0.00349; this subset is descriptive and selected on outcomes. Differences across matched reference resets are not evidence of a monotonic overall decline.

Closed-loop evaluation produced full real returns 10.00454, 9.99013, 10.08270. Because each version continues with a different policy beyond step 16, differences combine changed state and changed continuation; use the fixed-continuation experiment to isolate this effect. Even fixed-continuation real actor history is policy-dependent, whereas StateValue does not receive actor history, so it does not establish critic bias as a function of state alone.

Limitations: only one direction and one base checkpoint, two reset seeds, no observation noise, finite precision and early-failure discontinuity, and reset states do not represent all replay states. Paired local finite differences do not measure the actual backpropagated actor gradient or prove critic is solely responsible. Test outcomes indicate an actionable short-term vs tail ranking discrepancy, not a general causal proof. Reproducible via the project's image with `/workspace` PYTHONPATH, project mounted read-only at `/workspace/go2_mujoco_mimic`, assets at `/assets`, and `/tmp` writable. Each script writes its raw output to `/tmp`.
