"""
Streamlit demo for our LucidPPN reproduction (25 CUB bird species).

Run from the repository root (so that .streamlit/config.toml, the light theme, is picked up):
    streamlit run deploy/app.py
Set LUCID_CHECKPOINT to use a checkpoint other than MetiNet/checkpoint_quick_train.pt.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from lucid_inference import (CLASSES, DEFAULT_CHECKPOINT, NUM_PARTS, PROTOTYPE_DIR, ROOT, accepted_colours,
                             crop_patch, load_model, load_prototypes, patch_box, predict, pretty)

st.set_page_config(page_title='LucidPPN demo', layout='wide')

# Guided examples: (species folder, label, what to try). The hue values were checked against the trained model.
GUIDED = [
    ('Cardinal', 'colour supports the answer',
     'Set the slider to +0.50: the red parts are blocked by the colour check, but the answer holds.'),
    ('Indigo_Bunting', 'colour changes the answer',
     'Set the slider to +0.40: the full model switches to Painted Bunting, the shape-only model stays correct.'),
    ('Lazuli_Bunting', 'colour lowers the score, the answer holds',
     'Set the slider to +0.40: the score drops, the shape-only score does not move.'),
    ('Black_footed_Albatross', 'colour barely matters',
     'Recolouring a grey bird changes almost nothing.'),
    ('Least_Auklet', 'an incorrect prediction',
     'An incorrect, low-confidence prediction: no species matches well.'),
]
BLUE, LIGHT_BLUE, ORANGE, GREY = '#2f6fdf', '#b3c9f2', '#e8613d', '#9aa3b2'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c5cbd6', 'axes.labelcolor': '#1d2533',
                     'xtick.color': '#1d2533', 'ytick.color': '#1d2533'})


@st.cache_resource
def get_model(path):
    return load_model(path, 'cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def get_prototypes():
    return load_prototypes()


def ablation_files():
    return glob.glob(os.path.join(ROOT, 'experiments', '*', 'ablation', 'ablation.json')) + \
        glob.glob(os.path.join(ROOT, 'experiments', '*', '*', 'ablation', 'ablation.json'))


@st.cache_data
def deployed_results():
    """Test-set accuracy of the model in this demo, full and shape-only, per recolouring strength."""
    path = os.path.join(ROOT, 'experiments', 'jump4_rerun', 'ablation', 'ablation.json')
    if not os.path.exists(path):
        return None
    r = json.load(open(path))['results']
    return {float(h): (100 * v['lucidppn_top1'][0], 100 * v['shapetexnet_top1'][0]) for h, v in r.items()}


@st.cache_data
def results_by_colourfulness():
    """Pooled over every trained model: accuracy change under strong recolouring (random hue shift up to 0.5),
    for the 12 most colourful species, the 12 least colourful and all 25 (colourfulness from
    experiments/colour_by_species.json), plus the colour branch's gain on original colours."""
    colour_file, files = os.path.join(ROOT, 'experiments', 'colour_by_species.json'), ablation_files()
    if not os.path.exists(colour_file) or not files:
        return None
    colourfulness = {s: v['colourfulness'] for s, v in json.load(open(colour_file))['species'].items()}
    order = sorted(colourfulness, key=colourfulness.get, reverse=True)
    runs = [json.load(open(f))['results'] for f in files]

    def accuracy(species, hue, key):
        out = []
        for r in runs:
            pc = r[hue]['per_class']
            out.append(100 * sum(pc[s][key] * pc[s]['n'] for s in species) / sum(pc[s]['n'] for s in species))
        return np.array(out)

    groups = {}
    for label, species in [('12 most colourful\nspecies', order[:12]), ('12 least colourful\nspecies', order[-12:]),
                           ('All 25 species', order)]:
        full0, shape0 = accuracy(species, '0.0', 'lucidppn_top1'), accuracy(species, '0.0', 'shapetexnet_top1')
        full5, shape5 = accuracy(species, '0.5', 'lucidppn_top1'), accuracy(species, '0.5', 'shapetexnet_top1')
        groups[label] = {'full': float(np.mean(full5 - full0)), 'shape': float(np.mean(shape5 - shape0)),
                         'gain': float(np.mean(full0 - shape0)), 'positive': int(np.sum(full0 > shape0))}
    return {'groups': groups, 'models': len(runs)}


def example_path(species):
    paths = sorted(glob.glob(os.path.join(ROOT, 'deploy', 'examples', species, '*.jpg')))
    return paths[0] if paths else None


def colour_panel(pixel, swatch, size=144):
    """Top: the colour seen on the bird. Bottom: colours the model accepts for this part of this species."""
    panel = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, size - 1, 52], fill=tuple(pixel), outline='#1d2533')
    w, h = size // 8, (size - 58) // 5
    for i, rgb in enumerate(swatch[:40]):
        x, y = (i % 8) * w, 58 + (i // 8) * h
        draw.rectangle([x, y, x + w - 2, y + h - 2], fill=rgb)
    return panel


def colour_verdict(part):
    if part['colour_space_accepted'] >= 0.99:
        return ':gray[Colour not used]'
    if part['colour_match'] >= 0.9:
        return ':green[Colour fits]'
    if part['colour_match'] <= 0.1:
        return ':red[Colour does not fit]'
    return ':orange[Colour partly fits]'


VERDICT_COLOURS = {'green': '#1e8e3e', 'red': '#d93025', 'gray': '#6b7280', 'orange': '#c26401'}


def parts_figure(rows, species):
    """One row per part: a text column, then where the model looked, your bird, the training example and colour."""
    titles = ['Where it looked', 'Your bird', f'Training {species}', 'Colour: seen / accepted']
    fig, axes = plt.subplots(len(rows), 5, figsize=(9, 1.95 * len(rows)),
                             gridspec_kw={'width_ratios': [1.25, 1, 1, 1, 1], 'wspace': 0.08, 'hspace': 0.12})
    for r, (part, images) in enumerate(rows):
        verdict = colour_verdict(part)
        colour_name, text = verdict[1:verdict.index('[')], verdict[verdict.index('[') + 1:-1]
        ax = axes[r, 0]
        ax.axis('off')
        ax.text(0, 0.78, f"Part {part['part']}", fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0, 0.56, f"Shape match {part['shape_texture']:.2f}", fontsize=9, transform=ax.transAxes)
        ax.text(0, 0.36, text, fontsize=9, color=VERDICT_COLOURS[colour_name], transform=ax.transAxes)
        ax.text(0, 0.16, f"Evidence {part['shape_plus_colour']:.2f}", fontsize=9, transform=ax.transAxes)
        for c, img in enumerate(images):
            ax = axes[r, c + 1]
            ax.imshow(img.resize((144, 144)))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#d5dae3')
            if r == 0:
                ax.set_title(titles[c], fontsize=8.5, color='#1d2533')
    return fig


def ranking(result, key):
    rows = sorted(result['top_k'], key=lambda t: t[key], reverse=True)
    total = sum(t[key] for t in rows)
    return [(t['class'], t[key] / total) for t in rows]


def top3_chart(ranked, highlight):
    names, values = [n for n, _ in ranked[:3]][::-1], [100 * v for _, v in ranked[:3]][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 1.5))
    bars = ax.barh(names, values, color=[BLUE if n == highlight else LIGHT_BLUE for n in names], height=0.6)
    ax.bar_label(bars, labels=[f'{v:.0f}%' for v in values], padding=3)
    ax.set_xlim(0, 100); ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    fig.tight_layout()
    return fig


def part_chart(a, b, label_a, label_b, colour_a, colour_b):
    idx = np.arange(NUM_PARTS)
    fig, ax = plt.subplots(figsize=(7, 2.1))
    ax.bar(idx - 0.2, a, 0.4, color=colour_a, label=label_a)
    ax.bar(idx + 0.2, b, 0.4, color=colour_b, label=label_b)
    ax.set_xticks(idx, [f'Part {k + 1}' for k in range(NUM_PARTS)]); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.3))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def correctness(answer, truth):
    if truth is None:
        return ''
    return ':green[Correct]' if answer == truth else f':red[Incorrect] (true species: {truth})'


net = get_model(os.environ.get('LUCID_CHECKPOINT', DEFAULT_CHECKPOINT))

# ------------------------------------------------------------------ sidebar: choose a bird
with st.sidebar:
    st.header('Choose a bird')
    mode = st.radio('Source', ['Guided examples', 'Test set', 'Upload'], label_visibility='collapsed')
    image, true_species, hint = None, None, None
    if mode == 'Guided examples':
        available = [g for g in GUIDED if example_path(g[0])]
        choice = st.radio('Example', available, format_func=lambda g: f'{pretty(g[0])}: {g[1]}')
        image = Image.open(example_path(choice[0])).convert('RGB')
        true_species, hint = pretty(choice[0]), choice[2]
    elif mode == 'Test set':
        full_set = os.path.isdir(os.path.join(ROOT, 'CUB_subset', 'test'))
        folder = os.path.join(ROOT, 'CUB_subset', 'test') if full_set else os.path.join(ROOT, 'deploy', 'examples')
        species = st.selectbox('Species', sorted(d for d in os.listdir(folder)
                                                 if os.path.isdir(os.path.join(folder, d))), format_func=pretty)
        paths = sorted(glob.glob(os.path.join(folder, species, '*.jpg')))
        path = st.selectbox(f'Image ({len(paths)} available)', paths,
                            format_func=lambda p: f'{paths.index(p) + 1}. {os.path.basename(p)}')
        if not full_set:
            st.caption('One image per species is bundled. Run `prepare_cub_subset.py` for all 652 test images.')
        image, true_species = Image.open(path).convert('RGB'), pretty(species)
    else:
        f = st.file_uploader('Photo of one of the 25 species, cropped to the bird', type=['jpg', 'jpeg', 'png'])
        if f:
            image = Image.open(f).convert('RGB')
    st.divider()
    st.caption('Model: our LucidPPN reproduction (Pach et al., ICLR 2025), ResNet18 backbone, trained on 25 '
               'CUB-200-2011 species. Test images were not used for training.')
    with st.expander('The 25 species'):
        st.caption(', '.join(pretty(c) for c in CLASSES))

st.title('LucidPPN: explainable bird classification')
st.caption('Each prediction is explained part by part, with shape and colour assessed separately.')
if image is None:
    st.info('Choose or upload a bird in the sidebar.')
    st.stop()

result, maps, shown = predict(net, image, top_k=len(CLASSES))
pred_index = result['top_k'][0]['class_index']
tab_pred, tab_expl, tab_abl = st.tabs(['1  Prediction', '2  Explanation', '3  Colour ablation'])

# ------------------------------------------------------------------ 1. prediction
with tab_pred:
    left, right = st.columns([1, 1.3], gap='large')
    left.image(shown, caption='Model input (224 x 224)', width=320)
    with right:
        st.caption('Predicted species')
        st.header(result['prediction'])
        st.markdown(correctness(result['prediction'], true_species))
        full_rank = ranking(result, 'lucidppn_score')
        st.pyplot(top3_chart(full_rank, result['prediction']))
        st.caption('Relative match scores over the 25 species (not calibrated probabilities).')
        if full_rank[0][1] < 0.3:
            st.warning('Low confidence: no species matches well.')

# ------------------------------------------------------------------ 2. explanation
with tab_expl:
    st.markdown(f"Each part of the bird is compared with the same part of training **{result['prediction']}** "
                'images. Shape is judged in black and white; colour is checked separately.')
    if result['parts_at_same_spot']:
        groups = '; '.join(', '.join(map(str, g)) for g in result['parts_at_same_spot'])
        st.caption(f'Parts looking at the same spot: {groups}. Each group counts as one piece of evidence.')
    protos = get_prototypes().get(CLASSES[pred_index], {})
    gray = shown.convert('L')
    base = np.asarray(gray.convert('RGB')).astype(np.float32) / 255
    rows = []
    for k, part in enumerate(result['parts']):
        row, col = part['location']
        overlay = Image.fromarray((255 * np.clip(0.5 * base + 0.5 * plt.cm.jet(maps[k].numpy())[..., :3], 0, 1))
                                  .astype(np.uint8))
        ImageDraw.Draw(overlay).rectangle(patch_box(row, col, part['grid']), outline='white', width=4)
        proto = protos.get(str(k + 1))
        rows.append((part, [overlay, crop_patch(gray, row, col, part['grid']).convert('RGB'),
                            Image.open(os.path.join(PROTOTYPE_DIR, proto['file'])).convert('L').convert('RGB')
                            if proto else Image.new('RGB', (64, 64), 'white'),
                            colour_panel(part['pixel_colour'],
                                         accepted_colours(net._colour_acceptance, k, pred_index)[0])]))
    halves = st.columns(2, gap='large')
    for half, chunk in zip(halves, [rows[:4], rows[4:]]):
        half.pyplot(parts_figure(chunk, result['prediction']))
    st.caption('Evidence = shape match x colour check. The class score is the mean evidence over the 8 parts. '
               'Parts are found without labels during training, so they are numbered rather than named.')

    sources = {k: os.path.join(ROOT, protos[str(k + 1)]['source']) for k in range(NUM_PARTS)
               if str(k + 1) in protos and os.path.exists(os.path.join(ROOT, protos[str(k + 1)]['source']))}
    if sources:
        with st.expander('Full training images behind each training example'):
            grid = st.columns(4)
            for k, path in sources.items():
                full = Image.open(path).convert('RGB').resize((224, 224))
                ImageDraw.Draw(full).rectangle(patch_box(*protos[str(k + 1)]['location'], protos[str(k + 1)]['grid']),
                                               outline='#2f6fdf', width=4)
                grid[k % 4].image(full, caption=f'Part {k + 1}: {os.path.basename(path)}')

    st.subheader('Compare with another species')
    runner_up = result['comparison']['class_index']
    others = [i for i in range(len(CLASSES)) if i != pred_index]
    other = st.selectbox('Species', others, index=others.index(runner_up),
                         format_func=lambda i: pretty(CLASSES[i]) + (' (second best)' if i == runner_up else ''))
    cmp = result['comparison'] if other == runner_up else \
        predict(net, image, top_k=len(CLASSES), compare=other)[0]['comparison']
    st.pyplot(part_chart([p['shape_plus_colour'] for p in result['parts']], [p['shape_plus_colour'] for p in cmp['parts']],
                         result['prediction'], cmp['class'], BLUE, ORANGE))
    st.caption(f"Evidence per part. Mean: {result['prediction']} {result['top_k'][0]['lucidppn_score']:.2f}, "
               f"{cmp['class']} {cmp['lucidppn_score']:.2f}.")

# ------------------------------------------------------------------ 3. colour ablation
with tab_abl:
    st.markdown('Turning the colour branch off leaves the **shape-only model**. The shape branch never learns from '
                'colour, so no retraining is needed.')
    if hint:
        st.info(hint)
    hue = st.slider('Recolour the bird (hue shift; brightness unchanged)', -0.5, 0.5, 0.0, 0.05)
    res_h, _, shown_h = predict(net, image, top_k=len(CLASSES), hue_shift=hue)
    c0, c1, c2 = st.columns([1, 1.2, 1.2], gap='large')
    c0.image(shown_h, caption='Original colours' if hue == 0 else f'Hue shifted by {hue:+.2f}', width=230)
    for column, title, answer, key in [(c1, 'Full model (shape + colour)', res_h['prediction'], 'lucidppn_score'),
                                       (c2, 'Colour branch off (shape only)', res_h['shapetexnet_prediction'],
                                        'shapetexnet_score')]:
        with column:
            st.caption(title)
            st.subheader(answer)
            st.markdown(correctness(answer, true_species))
            st.pyplot(top3_chart(ranking(res_h, key), answer))
    if res_h['prediction'] == res_h['shapetexnet_prediction']:
        st.success('Both models give the same answer.')
    else:
        st.warning('The colour branch changes the answer.')
    st.pyplot(part_chart([p['shape_texture'] for p in res_h['parts']], [p['shape_plus_colour'] for p in res_h['parts']],
                         'Shape match', 'Evidence after colour check', GREY, BLUE))
    st.caption(f"Parts of {res_h['prediction']}. A blue bar far below the grey one means the colour did not fit.")

    st.subheader('Results on the test set')
    left, right = st.columns(2, gap='large')
    deployed, split = deployed_results(), results_by_colourfulness()
    if deployed:
        hues = sorted(deployed)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(hues, [deployed[h][0] for h in hues], 'o-', color=BLUE, label='Full model')
        ax.plot(hues, [deployed[h][1] for h in hues], 's--', color=GREY, label='Shape only')
        ax.set_xticks(hues); ax.set_xlabel('Maximum random hue shift'); ax.set_ylabel('Top-1 accuracy (%)')
        ax.set_ylim(75, 90); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        left.pyplot(fig)
        left.caption('This model, 652 test images.')
    if split:
        labels = [g.replace('\n', ' ') for g in split['groups']] + ['Paper (all 200 species)']
        lost = [-g['full'] for g in split['groups'].values()] + [10.2]
        shape_lost = [-g['shape'] for g in split['groups'].values()] + [0.0]
        y = np.arange(len(labels))[::-1]
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(y, lost, color=[BLUE] * (len(labels) - 1) + [LIGHT_BLUE], height=0.55)
        ax.bar_label(bars, labels=[f'-{v:.1f} points' for v in lost], padding=3, fontsize=8)
        ax.set_yticks(y, labels); ax.set_xlim(0, max(lost) * 1.35); ax.set_xticks([])
        fig.suptitle('Full model: accuracy lost when colours change', fontsize=9, x=0.02, ha='left')
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        fig.tight_layout()
        right.pyplot(fig)
        right.caption(f"Shape-only model: {max(abs(v) for v in shape_lost):.1f} points lost in every case. Hue shifted at "
                      f"random by up to 0.5; mean of the {split['models']} models we trained. Paper: Pach et al. (2025), "
                      'Table 3.')
        allg = split['groups']['All 25 species']
        st.markdown(f"**Summary.** With original colours the colour branch adds **{allg['gain']:+.1f} points** "
                    f"(higher in {allg['positive']} of {split['models']} models; paper +1.4). When colours are "
                    'misleading it costs accuracy, mainly on colourful species. The shape-only model is unaffected.')
