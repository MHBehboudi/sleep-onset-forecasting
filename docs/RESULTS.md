# Results and ablations

All development numbers are W-bMAE in seconds on the 128-night development panel (40 nights from
people seen in training, 88 from unseen people), reported as the mean of the seen and unseen group
means, as on the blind test. Models are scored at their final epoch with EMA weights.

Views used throughout:

| View | Opening withheld | Faults |
|---|---|---|
| clean0 | none | none |
| clean300 | 300 s | none |
| mix | random 120–600 s | 35% of nights, one random fault |
| short | random 60–240 s | 35% of nights |

## 1. Blind-test evaluations

| # | Date | Package | Blind-test W-bMAE |
|---|---|---|---:|
| 1 | Nov 12 | fine-tuned REVE foundation model + GRU (41.66 under the original protocol) | 50.24 |
| 2 | Nov 17 | best clean-data model, fixed 4-channel layout | 41.94 |
| 3 | Nov 17–18 | PSG-pretrained transitions model | 42.33 |
| 4 | Nov 18 | spectral-shape model | 43.74 |
| 5 | Nov 18 | SENTINEL, 6 seeds | 35.49 |
| 6 | Nov 18 | + frozen MANAS-2 features, two selective-timescale members | 34.05 |
| 7 | Nov 19 | + refit on all 540 nights, half-strength output map | 33.89 |
| 8 | Nov 20 | + lookahead-teacher distillation | 33.74 |
| 9 | Nov 21 | attention-hierarchy line-up | 34.30 |
| 10 | Nov 21 | + near-onset-weighted members, 75% output map (final) | **33.52** |

Evaluation 9 was worse on the blind test despite looking better on long nights in development. The new
members had lost accuracy in the last 40 s, which is the bin with ten times the weight.

## 2. SENTINEL against the previous model (372 training nights)

| View | SENTINEL (3 seeds) | previous model |
|---|---:|---:|
| clean0 | 38.29 | 40.03 |
| clean300 | 35.79 | 37.50 |
| mix | 35.83 | 55.60 |

The clean gain is about 1.7 s. Under faults it is about 20 s.

## 3. What worked, in the order it was added

| Change | Evidence |
|---|---|
| Fault and opening-cut simulation during training | mix 49.85 → 35.73 for the same network |
| Reliability gating and quality probe | mix 37.30 → 35.73 |
| Frozen MANAS-2 packet features, zero-initialised | −0.35 / −0.55 (clean300 / mix), 3 seeds |
| Train-time global gain augmentation (p = 0.2) | about −0.3 on clean300 and the blind-test proxy |
| PSG pretraining mapped onto Muse slots | zero-shot mix 39.3; small gain after fine-tuning |
| Lookahead-teacher distillation (30 s teacher, cross-fitted) | −0.54 / −0.67 / −0.50 / −0.62 on all four views, 3 seeds |
| Monotone output map (participant-CV fitted) | −0.5 on the proxy; applied at 50–75% strength |
| Two selective-timescale members in the ensemble | roughly neutral on the development proxy (±0.1); kept for model diversity |

The 30 s lookahead teacher scores 30.45 on clean300 because it sees the future. A 2-minute teacher
did not help: the gap between teacher and student was too large to distil.

## 4. What did not work (tested end to end, 2–3 seeds unless noted)

| Idea | Result |
|---|---|
| Sharpness-aware minimisation (ρ = 0.05) | +0.4 to +0.5 worse |
| Phase-of-night mixture of experts (3 experts, learned gate) | tie; the gate could not tell the phase of the night |
| Hierarchical attention memory replacing the GRU | better on mixed views, worse on clean ones; worse on the blind test |
| In-model closed-loop correction from its own past forecasts | +0.2 to +0.36 worse |
| Long-range (2/5/10 min) self-supervised forecasting head | +0.1 to +0.2 worse |
| Per-person loss reweighting / group-robust tilt | +0.1 to +0.3 worse, unseen people not helped |
| Fine-tuning the last foundation-model blocks | tie or worse; unseen people got worse |
| A second frozen foundation model (REVE) | tie |
| Gradient-boosted trees on 949 hand features | 40.4 alone; residuals correlated 0.95 with the network; blend weight 0 |
| Post-hoc smoothing, latent-onset HMM filter, person priors | all worse; errors are persistent per-night offsets |
| Slow-eye-movement branch, multi-scale change-detector bank | no gain in within-night ranking |
| Shorter training (10 epochs) for better near-onset error | shifted predictions, did not improve ranking |

## 5. Two findings that decided the final model

**Shift versus skill.** Several changes lowered near-onset error on development by moving all
predictions down (near-weighted members, stronger output maps). A rule fitted on seven blind-test scores
said the blind-test score tracks development near-onset error. For these shift-type changes it projected a
gain of about 2 s, and the blind test gave 0.2 s. Changes that improved ranking skill transferred;
changes that only moved the output level did not. The measure that separates them is within-night
AUROC for "N2 within 40 s" against "40–90 s": every model sat at 0.944–0.949.

**What the model looks at.** Removing 11–16 Hz power (sleep spindles) from the input moves the
final-40-s predictions by +152 s; removing slow eye movements moves them by 0.5–2 s. The model finds
N2 mainly through spindles. Stage-supervised co-training on PSG data reduced that dependence to +60 s.
