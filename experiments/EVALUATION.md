# Full evaluation of all 12 models

Test set: 652 images of the 25-class subset. Classification metrics as in the paper (top-1, top-5); part alignment with the paper's IoU (prototype map vs the PDiscoNet mask the model was trained with, definition from the authors' `metinet/test.py`) and with keypoint NMI/ARI (which part's prototype is strongest at each of CUB's 15 human-annotated keypoints; 7,571 visible keypoints; comparable across models). Mean ± std over models.

## Classification and part alignment

| Configuration | Models | Top-1 | Top-5 | ShapeTexNet top-1 | ShapeTexNet top-5 | IoU with own masks | IoP | IoT | Keypoint NMI ↑ | Keypoint ARI ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Code defaults, 5-epoch parts | 3 | 83.5 ± 0.6 | 97.9 ± 0.2 | 82.1 ± 0.9 | 97.1 ± 0.4 | 0.332 ± 0.008 | 0.602 ± 0.009 | 0.582 ± 0.003 | 17.6 ± 0.6 | 8.6 ± 0.3 |
| Code defaults, 28-epoch parts | 3 | 83.0 ± 0.7 | 97.8 ± 0.2 | 81.5 ± 0.6 | 97.2 ± 0.4 | 0.268 ± 0.007 | 0.636 ± 0.016 | 0.424 ± 0.003 | 20.8 ± 0.8 | 10.4 ± 0.2 |
| Paper recipe, 5-epoch parts | 3 | 81.0 ± 1.7 | 97.2 ± 0.5 | 79.6 ± 0.9 | 96.8 ± 0.1 | 0.439 ± 0.027 | 0.744 ± 0.030 | 0.581 ± 0.001 | 18.1 ± 0.3 | 8.6 ± 0.2 |
| Paper recipe, 28-epoch parts | 3 | 81.4 ± 2.0 | 97.2 ± 0.5 | 80.4 ± 1.9 | 96.9 ± 0.5 | 0.313 ± 0.008 | 0.718 ± 0.031 | 0.429 ± 0.006 | 22.6 ± 0.6 | 11.1 ± 0.3 |

IoU is measured against each model's own training masks, which differ between the 5-epoch and the 28-epoch detector (different, unsupervised parts), so IoU values are comparable within a column of part detectors only. Keypoint NMI/ARI uses the same human annotations for every model.

## Per-species accuracy (all 12 models pooled)

| Species | Top-1 | Most often confused with |
|---|---|---|
| Brandt Cormorant | 47.1% | Pelagic Cormorant (17% of its images) |
| Pelagic Cormorant | 51.4% | Brandt Cormorant (30% of its images) |
| Rusty Blackbird | 57.8% | Brewer Blackbird (26% of its images) |
| Black footed Albatross | 67.2% | Sooty Albatross (19% of its images) |
| Red faced Cormorant | 69.7% | Brandt Cormorant (9% of its images) |
| Rhinoceros Auklet | 76.9% | Brandt Cormorant (4% of its images) |
| Brewer Blackbird | 77.9% | Rusty Blackbird (12% of its images) |
| Indigo Bunting | 81.7% | Lazuli Bunting (6% of its images) |
| Groove billed Ani | 82.2% | Pelagic Cormorant (3% of its images) |
| Red winged Blackbird | 82.8% | Bobolink (8% of its images) |
| Lazuli Bunting | 86.3% | Indigo Bunting (4% of its images) |
| Cardinal | 87.7% | Painted Bunting (4% of its images) |
| Laysan Albatross | 88.1% | Sooty Albatross (6% of its images) |
| Painted Bunting | 88.7% | Indigo Bunting (5% of its images) |
| Crested Auklet | 89.9% | Black footed Albatross (4% of its images) |
| Yellow breasted Chat | 90.5% | Painted Bunting (4% of its images) |
| Least Auklet | 90.9% | Eastern Towhee (2% of its images) |
| Gray Catbird | 91.4% | Painted Bunting (6% of its images) |
| Yellow headed Blackbird | 92.3% | Laysan Albatross (3% of its images) |
| Bobolink | 92.5% | Yellow breasted Chat (3% of its images) |
| Eastern Towhee | 95.0% | Rhinoceros Auklet (1% of its images) |
| Sooty Albatross | 96.1% | Laysan Albatross (1% of its images) |
| Parakeet Auklet | 96.7% | Crested Auklet (1% of its images) |
| Chuck will Widow | 96.8% | Pelagic Cormorant (1% of its images) |
| Spotted Catbird | 100.0% | — |

## Most confused pairs

| True species | Predicted as | Share of its test images |
|---|---|---|
| Pelagic Cormorant | Brandt Cormorant | 29.7% |
| Rusty Blackbird | Brewer Blackbird | 25.6% |
| Black footed Albatross | Sooty Albatross | 19.2% |
| Brandt Cormorant | Pelagic Cormorant | 16.7% |
| Brewer Blackbird | Rusty Blackbird | 12.4% |
| Red faced Cormorant | Brandt Cormorant | 9.5% |
| Pelagic Cormorant | Red faced Cormorant | 8.9% |
| Brandt Cormorant | Red faced Cormorant | 8.0% |
| Red winged Blackbird | Bobolink | 7.5% |
| Gray Catbird | Painted Bunting | 6.3% |

![Confusion matrix](confusion_matrix.png)

![Part alignment](part_alignment.png)
