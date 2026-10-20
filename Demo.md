# Demo

## Demo video
Recorded walkthrough of the deployed application (Google Drive):
https://drive.google.com/drive/folders/1KHj0iQDrvZElw08nvjhO0okOaFM_KFEi?usp=sharing

The video is kept on Drive rather than in this repository because of its size.

## Run the demo yourself
Needs Python 3.10-3.13. No GPU, dataset or Colab is needed: the trained model and example images are in the repository.
- **Windows:** double-click `run_demo.bat` (in PowerShell: `.\run_demo.bat`)
- **macOS / Linux:** `bash run_demo.sh`

The first run installs the dependencies into `.venv` (5-10 minutes); the app then opens at http://localhost:8501.

## What the app shows
1. **Prediction:** the predicted species, whether it is correct for test images, and the top relative match scores.
2. **Explanation:** for each of the 8 parts, where its prototype fires on the bird, the matching patch of a training
   image, and the colour check (the colour on the bird and the colours the colour branch accepts for that part).
3. **Colour ablation:** the full model and the shape-only model side by side, with a slider that recolours the bird
   while keeping its brightness.

The REST API, input and output formats and a worked example are in [`deploy/README.md`](deploy/README.md).
