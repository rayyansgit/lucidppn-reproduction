# Material for the report and slides: Jumps 5-7 and multi-seed Stage 4

Drop-in text for REPORT.md and the presentation, organised by report section. Rewrite it in your own words before
submitting (the brief assesses this). All numbers come from files in `experiments/`; the full write-up is at the end
of `EXPERIMENTS.md`.

**Headline in one sentence:** with 13 training runs and 12 evaluated models, the colour branch adds a small but real
+1.4 points (12/12 models, p = 0.0005, as in the paper), its hue sensitivity is concentrated in colourful species
(ρ = 0.62), and neither the paper's training recipe nor a 4× better-trained part detector improves accuracy on our
25-class subset.

---

## Section 4, Implementation details: add after the comparison table
> After Stage 4 we ran a 2 × 2 study on a local RTX 2080 laptop GPU, 3 seeds per configuration. One axis is the
> training recipe: the official code's default arguments (our Jump 4) versus the authors' own launch script
> `MetiNet/run_training.sh`, which matches the paper's supplement (40 epochs, 15 frozen, lr 0.002, α_D = 1.4,
> batch 64). The other axis is the part detector: PDiscoNet trained for 5 epochs as in Stage 3, or for the paper's
> 28 epochs. Backbone (ResNet18), K = 8 parts, classes and evaluation are unchanged. The official defaults differ from
> the published configuration; the published one is only in the launch script.

## Section 6, Reproduction results: add a multi-seed table
| Test top-1, mean ± std | 5-epoch part detector | 28-epoch part detector |
|---|---|---|
| Code defaults, 82 epochs (Jump 4) | **83.1 ± 1.0** (4 runs) | 83.0 ± 0.7 (3 runs) |
| Paper recipe, 40 epochs | 81.0 ± 1.7 (3 runs) | 81.4 ± 2.0 (3 runs) |
| Paper, Table 1 (200 classes, ConvNeXt-Tiny) | 81.5 ± 0.4 | |

Figure: `experiments/summary.png` (every run's test accuracy per epoch).

> Our Jump 4 result is stable across seeds (83.1 ± 1.0 over four runs, including the original 81.9% and the rerun's
> 84.0%). The seed spread of 0.7-2.0 points is larger than the paper's ±0.4, as expected with 652 test images, where
> one image is worth 0.15 points.

Part-detector quality (authors' `evaluate_cub.py` metrics on the 25-class test set):

| Part detector | NMI ↑ | ARI ↑ | Keypoint regression error ↓ |
|---|---|---|---|
| 5 epochs (Stage 3) | 24.1 | 11.8 | 14.3% |
| 28 epochs (paper's length) | 31.0 | 14.9 | 12.7% |

## Section 7, Additional experiment: replace the single-model results with these
| Over 12 models | Ours | Paper |
|---|---|---|
| Colour gain on clean images (LucidPPN − ShapeTexNet) | **+1.4** (positive in 12/12, p = 0.0005) | +1.4 |
| LucidPPN change at hue shift 0.5 | **−4.1** (negative in 11/12) | −10.2 |
| ShapeTexNet change at hue shift 0.5 | **0** (every model) | 0 |

> **Answer to our question:** the colour branch adds about 1.4 points of accuracy, the same as the paper reports, and
> the effect is consistent: it is positive in all 12 models. The cost is robustness: under strong hue changes
> LucidPPN loses about 4 points on average, while the shape/texture branch is completely unaffected.

Per-species figure: `experiments/colour_by_species.png`.

## Section 8, Analysis and discussion: new points
1. **One model was not enough.** On the single Stage 4 model the colour gain was not significant (12 images fixed vs
   5 broken, p ≈ 0.14). Across 12 models it is (12/12 positive, p = 0.0005). Colour reliance also varies widely
   between seeds with identical settings: the hue-shift change ranges from +0.2 to −6.6 points. A single-model
   ablation can mislead.
2. **Hue sensitivity is concentrated in colourful species.** Per species, the accuracy drop under hue shift
   correlates with how colourful its images are (Spearman ρ = 0.62, p = 0.001). The 12 most colourful species lose
   7.2 points, the 12 least colourful 0.8. Cardinal (−29) and Indigo Bunting (−28) dominate, while albatrosses,
   cormorants and auklets barely change. This explains why our average drop (−4.1) is well below the paper's (−10.2):
   our 25 species are mostly grey, brown and black; the full CUB has many more colour-defined species.
3. **Better parts did not mean better accuracy.** The 28-epoch part detector is clearly better on all three part
   metrics, yet accuracy did not change (83.1 vs 83.0; 81.0 vs 81.4). Classification only needs roughly correct
   parts: the part loss acts mainly as a regulariser. Any
   benefit would show in explanation quality (prototype-part IoU), which we did not measure.
4. **The paper's recipe doesn't transfer one-to-one to a small subset.** It learns faster early (higher learning
   rate) but ends ~2 points lower: with 750 images and batch 64, 40 epochs make only ~440 weight updates vs ~1,900 for
   Jump 4. The schedule was designed for 8× more training images.
5. **Correction to our Stage 3 warm-up explanation.** With the paper's higher learning rate the model learns during
   the frozen phase (~30% by epoch 15); with the default rate it stays at chance. The "dead" warm-up was mainly the
   low learning rate, not only the missing `layer4.2` block.

## Section 10, Limitations: update
- Remove or soften "one seed" and "PDiscoNet quality unchecked": both are now addressed (3 seeds per configuration;
  part quality measured with NMI/ARI/keypoint error).
- Keep: 25 classes, ResNet18 backbone (weaker than ConvNeXt-Tiny; same 28×28 map resolution), K = 8, no validation split, no prototype-part IoU, hue-only perturbation.
- Add: the colour finding depends on which species are in the subset (see point 2 above).
- Add: the paper recipe was not re-tuned for our smaller dataset (e.g. more epochs to match its number of updates).

## Section 9, Deployment: one line to add
> The app shows a top-5 "share of evidence" chart (each species' score divided by the sum over all species). The
> model scores species independently (per-species sigmoid, trained with binary cross-entropy), so these are
> normalised scores, not calibrated probabilities.

## Suggested slides
1. **"Is the colour gain real?"** The 12/12 models table: our +1.4 vs the paper's +1.4.
2. **"Who is affected by colour changes?"** `experiments/colour_by_species.png`: Cardinal and Indigo Bunting vs the
   grey cluster.
3. **"Did following the paper more closely help?"** The 2 × 2 accuracy table and the part-quality table. Message:
   better parts, same accuracy; the paper recipe is tuned for 8× more data.
4. **"Why seeds matter"** (optional): hue-shift change ranging from +0.2 to −6.6 with identical settings.
