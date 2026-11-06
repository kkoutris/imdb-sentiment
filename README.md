# IMDb sentiment classification

NLP1 project 1: a comparison of text representations and classifiers for binary sentiment on
[`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb). It uses cross-validation and a Bayesian analysis against a
region of practical equivalence (ROPE).

The whole pipeline is in **`imdb_sentiment.ipynb`**: data, tokeniser, representations, models, tuning,
cross-validation, Bayesian comparison, test evaluation and error analysis.

## Running

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab imdb_sentiment.ipynb
```

Slow stages save to `results/` and skip work already done, so a run from the saved results takes about a minute
plus the dataset download. To redo everything, set `RECOMPUTE = True` in the setup cell. That writes to
`results_recomputed/` and takes about 1.5 hours on a laptop CPU.

## Layout

| Path | Contents |
|---|---|
| `imdb_sentiment.ipynb` | The full pipeline, with all project code inline |
| `nlp1_tutorial/` | Tutorial modules used by the notebook, copied unchanged (see Credits) |
| `results/vocab_sweep.csv`, `results/tokenizers/` | Tokeniser candidates and their dev scores |
| `results/tuning.csv`, `results/cv_scores.csv` | Dev tuning and 5-fold × 3-seed CV scores for the nine systems (three architectures × three representations) |
| `results/cv_predictions/` | Validation predictions for every fold, for error analysis |
| `results/test_results.json`, `results/test_predictions.npz` | The single test evaluation |
| `results/figures/` | Figures written by the notebook |
| `results/logs/` | Logs from the original runs |
| `AI_USAGE.md` | Record of AI assistance |

## Results

Mean CV accuracy over 15 folds:

| | Token indicator | Bag-of-words | TF-IDF |
|---|---|---|---|
| Log-linear | 0.8714 | 0.8723 | 0.8813 |
| MLP, 1 hidden layer | 0.8846 | **0.8885** | 0.8913 |
| MLP, 2 hidden layers | 0.8839 | 0.8874 | 0.8911 |

- **Selected system:** at ROPE 0.005, the one-hidden-layer MLP with bag-of-words. The selected representation changes
  with the ROPE: TF-IDF at 0.0025, token indicator at 0.01.
- **Architectures:** the MLP beats log-linear on every representation (by 0.010 to 0.016); a second hidden layer
  ties with one.
- **Test:** single models average 0.8803, and a majority vote of the 15 fold models reaches 0.8853.

## Credits

`nlp1_tutorial/` contains `analysis.py`, `clslib.py`, `experiment.py`, `loglin.py` and `smlp.py`, copied without
changes from the NLP1 2026 tutorial repository (`part1/`). They are under the Apache License 2.0 (`nlp1_tutorial/LICENSE`).
Data: Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, through the `mteb/imdb` copy on Hugging Face.
