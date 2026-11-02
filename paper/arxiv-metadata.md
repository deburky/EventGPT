# arXiv submission metadata

Paste each field into the arXiv "Enter Metadata" form. Upload `tarball/eventgpt-arxiv.tar.gz`
(built by `make zip`; arXiv compiles it with pdfLaTeX, 42 pages).

## Title
```
EventGPT: Next-Event Pretraining on Banking Customer Histories
```

## Author(s)
```
Denis Burakov
```

## Abstract
(plain text, 1,257 characters; arXiv's limit is 1,920)
```
A model pretrained to predict a customer's next event can estimate the probability of future events without training on their labels and provide a reusable representation for supervised downstream banking tasks. We pretrain EventGPT, a hierarchical causal transformer, by next-token maximum likelihood estimation on the public Multimodal Banking Dataset (MBD-mini: 100,224 clients, 38.8 million transactions and 66.3 million geolocation pings). Keeping one state per event, EventGPT trains in about two hours on one L4 GPU at windows up to 8,192 tokens. From its next-event distribution, with no labels, EventGPT scores five targets of the form "the client makes a transaction of a given kind for the first time next month" at ROC-AUC 0.79 to 0.92. Gradient boosting on 114 engineered features needs 10,000 to 300,000 labeled client-months to match it on four targets and stays below it on the fifth. For product prediction, frozen EventGPT embeddings score below the engineered features (mean ROC-AUC 0.787 against 0.808 over five test folds), and a rank-averaged ensemble of EventGPT, profile fields and token counts with the engineered features adds 0.014 to 0.017, of which 0.007 to 0.016 is EventGPT's own. The code and pretrained weights are released.
```

## Comments
```
42 pages, 12 figures, 15 tables. Code: https://github.com/deburky/EventGPT. Weights: https://huggingface.co/deburky/EventGPT
```

## Leave blank
- Report number
- Journal reference
- External DOI
- ACM class (optional: `I.2.6` learning)
- MSC class

## Categories (next screen)
- Primary: **cs.LG** (the endorsed category)
- Optional cross-lists: **stat.ML**, **q-fin.ST**

## License
- **CC BY 4.0** (matches the weights and MBD-mini; the code is Apache-2.0). Choose the arXiv.org perpetual,
  non-exclusive license instead to keep journal options open, as for arXiv:2509.09855.

## After the ID is assigned
- Add the arXiv link and ID to `README.md`, the Hugging Face model card and both BibTeX entries.
