# Presentation outline (about 13-15 minutes + demo)

The order follows the brief: problem → key idea → what we implemented → results vs paper → our experiment and its
finding → deployment demo. Suggested split between three speakers: Rayyan (slides 1-5), Laksh (6-10),
Ahmad (11-15 + demo). Every number below comes from `REPORT.md` / `EXPERIMENTS.md` / `experiments/`.

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | LucidPPN reproduction, team, ICLR 2025 paper | App screenshot (`assets/demo/app_albatross_full_page.png`) |
| 2 | The problem | Prototype networks explain with "this looks like that", but *why* did it match: shape, texture, or colour? And prototypes of different classes point at different body parts. | A red-wing prototype patch with "shape? colour?" |
| 3 | Key idea | Two branches: ShapeTexNet sees grayscale only, ColorNet sees one pixel's colour at a time, fused by multiplication with a stop-gradient. PDiscoNet part maps tie prototype k to part k in every class. | Redrawn architecture diagram (draw it yourselves, don't paste Fig. 3) |
| 4 | What the user gets | For each part: how well the shape matched vs how well the colour matched | App's per-part bar chart |
| 5 | What we implemented | Data prep → PDiscoNet → part maps → MetiNet → evaluation → app. Authors' model code reused unchanged; drivers, experiments and deployment are ours (`PROVENANCE.md`). Colab for Stages 3-5, laptop GPU for the follow-up study. | Pipeline flowchart |
| 6 | Scaling down | Paper vs ours: 200 → 25 classes, ConvNeXt-Tiny → ResNet18, 28×28 → 7×7 maps, K 12 → 8, PDiscoNet 28 → 5 epochs | Table |
| 7 | Reproduction results | Jumps 2 → 4: 33.9 → 41.6 → 81.9%; over 4 runs **83.1 ± 1.0%** vs paper 81.5 ± 0.4%. Same-looking number, easier task (25 vs 200 classes): not a like-for-like match. | `analysis/figures/training_curves.png` |
| 8 | What the curves taught us | (1) Part maps halve the train-test gap (18 → 10 points). (2) Accuracy stays at chance during the frozen warm-up, mainly because the default learning rate is 4× too low. (3) Plateau from ~epoch 40; train 99% vs test 83%. (4) We report final epochs, never the test-picked best. | Curves with unfreeze lines |
| 9 | Following the paper more closely | We found the paper's real settings in the authors' `run_training.sh` (the defaults differ in 6 settings: epochs, frozen epochs, 3 learning rates, part weight) and ran a 2 × 2 study, 3 seeds each. Paper recipe: **81.0-81.4%** vs code defaults **83.0-83.1%**, because 40 epochs on 750 images = 4× fewer weight updates. | 2 × 2 accuracy table + `experiments/summary.png` |
| 10 | Better parts, same accuracy | 28-epoch part detector: NMI 24.1 → 31.0, ARI 11.8 → 14.9, keypoint error 14.3 → 12.7%, yet accuracy unchanged (83.1 vs 83.0). Why: 7×7 maps throw away fine part detail; part loss acts as a regulariser. | Part-quality table |
| 11 | Our experiment: the question | "How much of the accuracy comes from the colour branch, and what does it cost in robustness to colour changes?" No retraining needed: the stop-gradient means ShapeTexNet never learns from colour. | One sentence, big |
| 12 | Is the colour gain real? | First run, one model: +1.1, not significant (p ≈ 0.14). Over 12 models: **+1.4, positive in 12/12 (p = 0.0005)**, same as the paper's +1.4. Hue shift 0.5: LucidPPN **−4.1**, ShapeTexNet **0** (paper −10.2 / 0). | Ours vs paper table |
| 13 | Who is affected by colour changes? | Hue sensitivity tracks colourfulness (Spearman ρ = 0.62, p = 0.001): Cardinal −29, Indigo Bunting −28, grey birds ≈ 0. Our 25 species are mostly grey, which explains −4.1 vs −10.2. | `experiments/colour_by_species.png` |
| 14 | What we learnt | Measure noise before comparing (seed spread ±1-2 points); read the launch scripts, not just the defaults; hyperparameters depend on dataset size; better components ≠ better results; the test data shapes the conclusion. | 5 bullets |
| 15 | Limitations | 25 mostly grey species, ResNet18 7×7, no validation split, no prototype-part IoU, hue-only perturbation, paper recipe not re-tuned for small data | Bullets |
| 16 | Demo | Live Streamlit app (below), then one API call | Live app (recorded backup ready) |

## Demo script (about 2 minutes)
1. Start the app on the laptop **before** presenting:
   `D:\lvenv\Scripts\python.exe -m streamlit run deploy/app.py` (or double-click `run_demo.bat`). It runs offline on
   the CPU; no Colab needed.
2. "Example from test set" → **Black-footed Albatross**: prediction ✓ correct, share of evidence 68% vs Sooty Albatross
   14%. Point at the per-part chart: shape and shape+colour are nearly equal (a grey bird).
3. Move the **hue slider** to 0.4: the albatross keeps its prediction and loses only ~15% of its score (grey has little colour to shift).
4. Switch to a **Cardinal**: 71% Cardinal. Move the hue slider to 0.5: the bird turns cyan, the Cardinal share drops
   to 46%, four parts lose their colour evidence, and the shape-only score doesn't move. "This is our experiment on a
   single image."
5. Optional: one API call from `deploy/README.md` to show the JSON.

## Demo checklist
- [ ] App started and tested on the presentation laptop 15 minutes before.
- [ ] Albatross and Cardinal test images found in the picker in advance.
- [ ] A 60-90 s screen recording of steps 2-4 as a backup.
- [ ] `deploy/README.md` open for the API call.

## Likely questions
- *Why is your accuracy equal to the paper's with a smaller model?* 25 vs 200 classes, so the task is easier; it is
  not a like-for-like match.
- *Why not report the 83.6% best epoch?* It was chosen by looking at test accuracy, which is selection on the test set.
- *Why can you ablate ColorNet without retraining?* The stop-gradient: ColorNet's loss never updates ShapeTexNet, so
  the shape branch inside the trained model is the model without colour.
- *Is +1.4 points worth a second branch?* Accuracy isn't the main point; the branch makes colour reliance visible per
  part. And the gain is consistent (12/12 models).
- *Why is your hue drop smaller than the paper's?* Our species are mostly grey; per species, colourful birds lose up
  to 29 points (ρ = 0.62).
- *Why did the paper's recipe do worse?* It's designed for 8× more images; on our subset it makes 4× fewer weight
  updates.
- *If the part detector got better, why didn't accuracy?* Our 7×7 feature maps discard fine part detail, and the part
  loss mainly regularises; the gain would show in explanation quality (IoU), which we didn't measure.
- *Are the app's percentages probabilities?* No: normalised scores ("share of evidence"), because the model scores each
  species independently with its own sigmoid.
- *What would you do with more compute?* ConvNeXt-Tiny with finer maps, a validation split, prototype-part IoU, and the
  colour experiment on a colourful subset or all 200 classes.
