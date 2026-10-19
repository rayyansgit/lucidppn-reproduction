"""
Streamlit demo for our LucidPPN reproduction (25 CUB bird species).

Run from the repository root:
    streamlit run deploy/app.py
Set LUCID_CHECKPOINT to point at a checkpoint other than MetiNet/checkpoint_quick_train.pt.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from lucid_inference import CLASSES, DEFAULT_CHECKPOINT, NUM_PARTS, ROOT, load_model, predict, pretty

st.set_page_config(page_title='LucidPPN bird classifier', layout='wide')


@st.cache_resource
def get_model(path):
    return load_model(path, 'cuda' if torch.cuda.is_available() else 'cpu')


st.title('LucidPPN: interpretable bird classification')
st.caption('Reproduction of Pach et al., ICLR 2025, trained on 25 CUB-200-2011 species (ResNet18 backbone). '
           'The model looks at shape/texture (grayscale branch) and colour (ColorNet) separately, '
           'so you can see which kind of evidence each bird part contributed.')

checkpoint = st.sidebar.text_input('Checkpoint', DEFAULT_CHECKPOINT)
if not os.path.exists(checkpoint):
    st.error(f'Checkpoint not found: {checkpoint}. Train with MetiNet/train_quick.py or download it '
             '(see deploy/README.md).')
    st.stop()
net = get_model(checkpoint)

st.sidebar.markdown('**Species the model knows**')
st.sidebar.write(', '.join(pretty(c) for c in CLASSES))

# Full 25-class test set if prepare_cub_subset.py has been run, otherwise the one-image-per-species samples in the repo
examples = (sorted(glob.glob(os.path.join(ROOT, 'CUB_subset', 'test', '*', '*.jpg')))
            or sorted(glob.glob(os.path.join(ROOT, 'deploy', 'examples', '*', '*.jpg'))))
source = st.radio('Image source', ['Upload', 'Example from test set'] if examples else ['Upload'], horizontal=True)
image = None
if source == 'Upload':
    f = st.file_uploader('Bird photo (JPG/PNG). Crop it tightly around the bird: the model was trained on '
                         'bounding-box crops.', type=['jpg', 'jpeg', 'png'])
    if f:
        image = Image.open(f).convert('RGB')
else:
    path = st.selectbox('Test image', examples, format_func=lambda p: f'{pretty(os.path.basename(os.path.dirname(p)))}'
                                                                     f' / {os.path.basename(p)}')
    image = Image.open(path).convert('RGB')

hue = st.slider('Hue shift (recolours the bird, keeps brightness; tests the colour-branch ablation live)',
                -0.5, 0.5, 0.0, 0.05)

if image is not None:
    result, maps, shown = predict(net, image, top_k=5, hue_shift=hue)
    left, right = st.columns([1, 2])
    with left:
        st.image(shown, caption='Model input (224×224)' + (f', hue shifted by {hue:+.2f}' if hue else ''))
    with right:
        st.subheader(f"Prediction: {result['prediction']}")
        st.write(f"Shape/texture branch alone would say: **{result['shapetexnet_prediction']}**")
        st.table([{'Species': t['class'], 'LucidPPN score': f"{t['lucidppn_score']:.3f}",
                   'Shape/texture only': f"{t['shapetexnet_score']:.3f}"} for t in result['top_k']])

    st.subheader(f"Evidence per part for '{result['prediction']}'")
    st.caption('Parts are discovered without labels by PDiscoNet, so they are numbered rather than named. '
               '"Shape + colour" is the shape/texture match multiplied by the colour match: when it is much lower '
               'than "shape/texture", the colour of that part did not fit the species.')
    fig, ax = plt.subplots(figsize=(8, 2.6))
    idx = np.arange(NUM_PARTS)
    ax.bar(idx - 0.2, [p['shape_texture'] for p in result['parts']], 0.4, label='Shape/texture')
    ax.bar(idx + 0.2, [p['shape_plus_colour'] for p in result['parts']], 0.4, label='Shape + colour')
    ax.set_xticks(idx, [f"Part {p['part']}" for p in result['parts']]); ax.set_ylim(0, 1)
    ax.set_ylabel('Resemblance'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

    st.caption('Where each part prototype of the predicted species fires (ShapeTexNet activation, 7×7 upsampled).')
    cols = st.columns(NUM_PARTS // 2)
    base = np.asarray(shown).astype(np.float32) / 255
    for k in range(NUM_PARTS):
        heat = maps[k].numpy()
        overlay = 0.5 * base + 0.5 * plt.cm.jet(heat)[..., :3]
        cols[k % len(cols)].image(np.clip(overlay, 0, 1), caption=f'Part {k + 1}: {heat.max():.2f}')

    with st.expander('Raw JSON response (same as the API)'):
        st.json(result)
