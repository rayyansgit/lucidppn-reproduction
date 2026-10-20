# Presentation (about 12-15 minutes including the demo)

Slides: 14 slides with speaker notes (download from the slide deck as PowerPoint or PDF). Order follows the brief:
problem, key idea, implementation, results against the paper, our experiment, demo, limitations.

| # | Slide | Speaker |
|---|---|---|
| 1 | Reproducing LucidPPN | Rayyan |
| 2 | The problem: shape, texture or colour? | Rayyan |
| 3 | LucidPPN: shape and colour judged separately | Rayyan |
| 4 | What we implemented | Laksh |
| 5 | Scaled to free GPU compute | Laksh |
| 6 | Reproduction: 83.1 ± 1.0% vs paper 81.5 ± 0.4% (not like-for-like) | Laksh |
| 7 | Following the paper more closely | Laksh |
| 8 | Experiment question | Laksh |
| 9 | How we tested it | Laksh |
| 10 | Colour helps a little; branches truly separate | Ahmad |
| 11 | The cost of colour falls on colourful birds | Ahmad |
| 12 | Live demo | Ahmad |
| 13 | Limitations | Ahmad |
| 14 | What we found | Ahmad |

## Demo script (about 2 minutes)
Start the app before presenting (`.\run_demo.bat`), on the laptop's CPU; keep the recorded video ready as backup.
1. Sidebar, guided example **Cardinal**: tab 1 (prediction, 71%), tab 2 (part by part: where it looked, matching
   training part in grayscale, colour check). Point out the note on parts sharing one spot.
2. Guided example **Indigo Bunting**, tab 3, slider to **+0.40**: the full model switches to Painted Bunting, the
   shape-only model stays correct.
3. Scroll down: test-set charts (accuracy against recolouring; accuracy lost by species group).
4. Guided example **Least Auklet**: an incorrect, low-confidence prediction.

## Likely questions
- *Why does your accuracy match the paper's with a weaker model?* 25 species is a much easier task; not a like-for-like
  match.
- *Why no retraining for the ablation?* The stop-gradient: the shape branch never learns from colour.
- *Why is your recolouring loss smaller than the paper's?* Our species are mostly grey; on the 12 most colourful the
  loss is 7.5 points (paper 10.2 on all 200).
- *Why not report the best epoch?* It would be chosen on the test set.
- *Are the app's percentages probabilities?* No, relative scores; each species is scored independently.
- *What is keypoint NMI?* PDiscoNet's part metric applied to our prototypes; not in the LucidPPN paper, only used to
  compare our own models.
- *With more compute?* ConvNeXt-Tiny, a validation split, a colourful subset or all 200 species, a user study.
