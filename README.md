# Emotion Recognition from Text — CNN vs LSTM
## Note : the code file is for local use
**NLP + Deep Learning | Pattern Recognition Project**

This project trains and compares two neural networks—a **1D Convolutional Neural Network (CNN)** and a **Long Short-Term Memory network (LSTM)**—to recognize the emotion expressed in an English sentence. The project uses a labeled Kaggle dataset with six classes: **anger, fear, joy, love, sadness, and surprise**.

> **Final result:** On the cleaned test set of **1,986 sentences**, CNN achieved **92.45% accuracy** and **0.8760 macro F1**, while LSTM achieved **91.24% accuracy** and **0.8666 macro F1**. CNN was better overall in this particular experiment; LSTM did better at recognizing *surprise* (recall).

## Project objective

The question is simple: **Can deep learning recognize emotional patterns in text, and which architecture works better on this dataset?**

For Pattern Recognition, the process is:

```text
Sentence → Text cleaning → Token IDs → Learned word embeddings
         → CNN or LSTM feature extraction → Softmax classifier → Emotion
```

- **Input pattern:** a sentence, such as `i feel happy today`.
- **Pattern representation:** a sequence of 50 token IDs; an embedding layer learns a 100-dimensional vector for each token.
- **Feature extraction:** CNN detects short local word combinations; LSTM learns relationships along the sentence sequence.
- **Decision/classification:** a six-output Softmax layer predicts one emotion.
- **Evaluation:** compare predicted classes with the ground-truth labels using accuracy, precision, recall, F1 and confusion matrices.

This is **supervised multiclass classification**. The models learn from labeled examples rather than relying on a list of hand-written emotional keywords.

## Dataset

**Source:** [Emotions Dataset for NLP — Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

The notebook loads the three semicolon-separated text files below. Each row contains a sentence and an emotion, e.g. `i didnt feel humiliated;sadness`.

| Split | Original sentences | After cleaning / overlap removal | Purpose |
|---|---:|---:|---|
| `train.txt` | 16,000 | **15,969** | Learn model parameters |
| `val.txt` | 2,000 | **1,993** | Monitor training and choose checkpoints |
| `test.txt` | 2,000 | **1,986** | Final assessment on held-out examples |

The notebook checks missing values, removes duplicate sentences, and prevents the same cleaned text from appearing in more than one split. All remaining cross-split overlaps are **zero**. This matters because seeing the same sentence during training and testing could make evaluation results appear better than they really are.

### Emotion class distribution

![Training emotion class distribution](assets/emotion_class_distribution.png)

The *original* 16,000-example training split contains:

| Emotion | Number of examples |
|---|---:|
| Joy | 5,362 |
| Sadness | 4,666 |
| Anger | 2,159 |
| Fear | 1,937 |
| Love | 1,304 |
| Surprise | 572 |

**Why does the graph look like this?** The dataset is **imbalanced**: joy and sadness are more common, while surprise is rare. As a result, the model receives many more training examples of certain emotions. This is one plausible contributor to poorer performance on surprise, but it does not prove that imbalance is the only cause.

### Sentence length distribution

![Sentence length distribution](assets/sentence_length_distribution.png)

Training sentences average **19.17 words**, and **95% contain 41 words or fewer**. The maximum observed length is **66 words**. Most sentences are short, with a smaller number extending to much longer lengths.

**Why use a length of 50?** A maximum sequence length of **50 tokens** covers most of the training examples while keeping the model input fixed in size. Shorter sequences are padded with zeros; longer ones are truncated. The plot measures whitespace-separated words, whereas the model uses Keras tokenization, so these lengths are close but not necessarily identical.

## Tools and libraries

- **Python** for the notebook and model code.
- **Pandas and NumPy** for loading data and processing arrays.
- **Matplotlib and Seaborn** for data and training plots.
- **TensorFlow / Keras** for text vectorization, CNN/LSTM architectures, training and saving.
- **scikit-learn** for label encoding, metrics and confusion matrices.
- **Jupyter Notebook or Kaggle Notebooks** to run the project interactively.

### How to run

1. Install the dependencies (or use a Kaggle environment that already provides them):

   ```bash
   pip install -r requirements.txt
   ```

2. Open [`code.ipynb`](code.ipynb) and run cells **in order**. The notebook's first cell reads `train.txt`, `val.txt` and `test.txt` from the **current working directory**; the included text files are placed alongside the notebook for that reason.
3. **On Kaggle**, attach the Emotions dataset through **Add Input** and update the three paths in Cell 0 to the actual `/kaggle/input/.../train.txt`, `val.txt` and `test.txt` locations. Dataset/input paths depend on your Kaggle notebook setup.
4. The notebook trains both networks, prints results, saves `.keras` models in `models/` and writes test results in `results/`. Model weights are generated when you train and are **not** included in this package.

> The results below are from the **provided executed notebook and CSV**. A new run can differ because of library versions, hardware and randomness. Your notebook uses seed 42, but perfect determinism is not guaranteed.

## Notebook explained, cell by cell

Cell numbers below are **zero-based notebook indices**, so *Cell 0* is the first cell in `code.ipynb`. Even comment-only cells are listed so you can follow the file from start to finish.

### Part 1 — Loading, cleaning and exploring the data (Cells 0–7)

| Cell | What the code does | Why it matters |
|---|---|---|
| **0** | Imports Pandas, NumPy, Matplotlib, Seaborn and `os`; loads three text files; prints example rows, dataset shapes, labels, counts and missing values. | Confirms what we are classifying and how many samples are available. |
| **1** | Checks missing values and exact duplicate rows in all splits. | Empty or repeated records can affect data quality and results. |
| **2** | Removes null/empty text, lowercases text, trims spaces, normalizes whitespace and cleans emotion labels. | Makes similar text consistent without removing emotional words or negations. |
| **3** | Plots the emotion class counts. | Reveals class imbalance (joy/sadness are common; surprise is rare). |
| **4** | A comment-only cell asking whether the dataset is balanced. | Records the observation that rare classes may be harder to recognize. |
| **5** | Counts words in each training sentence, plots a histogram and prints length statistics. | Helps choose a practical fixed input length: **50 tokens**. |
| **6** | Finds identical texts shared across training, validation and test splits. | Detects data leakage that could inflate evaluation results. |
| **7** | Deduplicates by text, removes cross-split overlaps and resets row indices. | Leaves **15,969 / 1,993 / 1,986** examples and **zero** shared texts between splits. |

### Part 2 — NLP preprocessing (Cells 8–14)

| Cell | What the code does | Why it matters |
|---|---|---|
| **8** | A comment marking the NLP section. | Separates data preparation from numerical text representation. |
| **9** | Imports TensorFlow's `TextVectorization` and scikit-learn's `LabelEncoder`. | Prepares text and labels for neural-network training. |
| **10** | A comment explaining why emotion names must be converted to numbers. | Models predict numeric class IDs, not Python strings. |
| **11** | Fits `LabelEncoder` on training labels and encodes all three splits. | Creates a consistent mapping: **0 anger, 1 fear, 2 joy, 3 love, 4 sadness, 5 surprise**. |
| **12** | A comment introducing tokenization and integer sequences. | Explains the transition from NLP text to numeric patterns. |
| **13** | Creates `TextVectorization` with **10,000 tokens** and **50-token sequences**; adapts it only to training text and transforms all splits. | Avoids learning vocabulary from validation/test data; pads inputs to the same shape. |
| **14** | Prints the vocabulary and the training array shapes. | Confirms a vocabulary size of **10,000** and training input shape of **(15969, 50)**. |

**Example:** `i didnt feel humiliated` becomes `[2, 139, 3, 685, 0, 0, ...]` in the recorded run. The integers are vocabulary IDs. Zero is padding, and `[UNK]` represents an out-of-vocabulary token. The embedding layer later learns useful features from these IDs.

### Part 3 — CNN model (Cells 15–23)

| Cell | What the code does | Why it matters |
|---|---|---|
| **15** | A comment introducing CNN construction and training. | Begins the first deep-learning experiment. |
| **16** | A comment noting that this CNN works on word-embedding sequences, not images. | Explains why a 1D convolution is used for NLP. |
| **17** | A comment describing automatic pattern learning. | Shows that features are learned rather than manually coded. |
| **18** | Imports Keras layers, callbacks and sets random seed **42**. | Prepares a repeatable CNN training setup. |
| **19** | Builds and compiles the CNN, then prints its model summary. | Uses **Embedding(100) → Conv1D(64, kernel 3) → GlobalMaxPooling1D → Dropout(0.3) → Dense(6, Softmax)**. |
| **20** | Trains for at most **15 epochs**, with batch size **32**, Adam learning rate **0.001**, and early stopping on validation loss (patience **3**). | Learns emotional patterns while restoring the checkpoint with the lowest observed validation loss. |
| **21** | Plots CNN training/validation accuracy and loss. | Helps identify learning progress and overfitting. |
| **22** | Evaluates CNN on validation sentences and prints precision, recall and F1 for each emotion. | Measures generalization without using the final test split. |
| **23** | Saves the network to `models/cnn_model.keras`. | Preserves the trained CNN for later loading and inference. |

**Why CNN?** A 1D convolution with a kernel size of three scans local combinations of neighboring token embeddings. It can learn signals from phrases that often occur with a particular emotion. Global max pooling keeps the strongest learned signals, and Softmax outputs the six class probabilities.

### Part 4 — LSTM model (Cells 24–31)

| Cell | What the code does | Why it matters |
|---|---|---|
| **24** | A comment introducing LSTM model training. | Begins the second architecture experiment. |
| **25** | Imports `LSTM`, `Embedding`, `Dropout`, `Dense` and training callbacks. | Prepares the recurrent network. |
| **26** | Builds and compiles **Embedding(100, mask_zero=True) → LSTM(64) → Dropout(0.3) → Dense(6, Softmax)**. | LSTM learns sequential dependencies and ignores zero-padding through masking. |
| **27** | Trains LSTM with the same batch size, maximum epochs, learning rate and early-stopping patience as CNN. | Makes the experiment more comparable, although the models have different architectures and padding handling. |
| **28** | Plots LSTM training/validation accuracy and loss. | Allows comparison of its learning behavior with CNN. |
| **29** | Prints LSTM validation accuracy, loss and per-class classification report. | Shows how well it classifies unseen validation sentences. |
| **30** | Creates a validation table comparing CNN and LSTM accuracy, loss and macro F1. | Uses validation performance to compare trained checkpoints before final testing. |
| **31** | Saves the model as `models/lstm_model.keras`. | Keeps the trained recurrent model for later use. |

**Why LSTM?** LSTM updates its internal state while processing tokens in sequence, allowing it to learn relationships that extend beyond a short local phrase. This could be useful for word order or context, but does not guarantee better performance on this dataset.

### Part 5 — Final evaluation (Cells 32–37)

| Cell | What the code does | Why it matters |
|---|---|---|
| **32** | A comment introducing final model evaluation. | Marks the transition to held-out testing. |
| **33** | Evaluates both models on `X_test`, predicts labels and calculates accuracy, loss, macro precision, macro recall and macro F1. | Produces the final head-to-head test comparison. |
| **34** | Prints a full classification report for each model. | Reveals which emotions work well and which remain difficult. |
| **35** | Generates and saves CNN and LSTM confusion matrices at **300 DPI**. | Shows exact correct counts and types of mistakes. |
| **36** | Creates tables of incorrectly classified test sentences for both models. | Helps inspect ambiguous, difficult or potentially noisy examples. |
| **37** | Saves the comparison CSV, vocabulary and emotion-label mappings. | Preserves results and preprocessing information for later use. |

### Part 6 — Test your own sentences (Cells 38–44)

| Cell | What the code does | Why it matters |
|---|---|---|
| **38** | A comment introducing custom sentence predictions. | Begins the interactive demonstration. |
| **39** | Defines `predict_emotion(text, model)` to clean, vectorize, predict and decode an emotion label. | Reuses the training preprocessing for inference. |
| **40** | Tests CNN on `I am so happy that I passed my exam`. | Produces **joy (86.66% top Softmax output)**. |
| **41** | Tests CNN on `I am so sad that I did not pass my exam`. | Produces **sadness (92.61% top Softmax output)**. |
| **42** | Runs six example sentences through both CNN and LSTM and displays a comparison table. | Shows agreement on clear statements and disagreement on more ambiguous ones. |
| **43** | Plots all six CNN outputs for `I feel so blue today` and selects the largest. | Predicts **sadness**, but the small margin between classes shows uncertainty. |
| **44** | An empty trailing notebook cell. | No code or result to interpret. |

## Training performance

### CNN learning curves

![CNN accuracy and loss curves](assets/cnn_training_curves.png)

- Training accuracy increases steadily, but validation accuracy levels off near **92%**.
- Training loss keeps falling, while validation loss is lowest at **epoch 2** (**0.1784**) and then rises.
- **Why?** This pattern suggests the model starts overfitting: it becomes better at the training examples without improving on unseen validation sentences. `EarlyStopping(restore_best_weights=True)` retains the epoch-2 weights. Although epoch 3 had slightly higher validation accuracy, the checkpoint was selected using **validation loss**, not accuracy.

**CNN validation metrics:** accuracy **92.27%**, loss **0.1784**, macro F1 **0.8982**.

### LSTM learning curves

![LSTM accuracy and loss curves](assets/lstm_training_curves.png)

- Training accuracy rises toward **98%**, but validation accuracy stays near **91%**.
- Validation loss reaches its best level at **epoch 3** (**0.2790**) and then worsens.
- **Why?** The growing training/validation gap also suggests overfitting. LSTM has more opportunities to memorize word-sequence details without improving performance on new sentences. This is a possible interpretation of this run, not proof of an inherent weakness in all LSTM models.

**LSTM validation metrics:** accuracy **91.22%**, loss **0.2790**, macro F1 **0.8825**.

> Both models use a trainable 100-dimensional embedding layer, batch size 32, the Adam optimizer and early stopping. However, **the LSTM masks zero-padding and the CNN does not**, so this is a comparison of the *implemented pipelines*, not a perfectly controlled isolation of architecture alone.

## Final test comparison

The uploaded [`model_comparison.csv`](results/model_comparison.csv) contains these results on the **1,986-sentence test set**:

| Metric | CNN | LSTM | Better result |
|---|---:|---:|---|
| Accuracy | **0.9245 (92.45%)** | 0.9124 (91.24%) | CNN |
| Cross-entropy loss | **0.1622** | 0.2626 | CNN (lower) |
| Macro precision | **0.8695** | 0.8560 | CNN |
| Macro recall | 0.8844 | **0.8851** | LSTM (very small lead) |
| Macro F1-score | **0.8760** | 0.8666 | CNN |
| Correct predictions | **1,836 / 1,986** | 1,812 / 1,986 | CNN |
| Incorrect predictions | **150** | 174 | CNN (fewer) |

**Why does CNN perform better overall here?** A plausible explanation is that these texts tend to be short (average ~19 words) and often contain useful local emotional cues. CNN is designed to learn short phrase patterns efficiently. LSTM models sequence order, but that added capacity did not improve the overall scores in this experiment. We cannot conclude that CNN is universally superior based on one dataset and one training run.

**What do the metrics mean?** Accuracy counts all correct answers. Precision measures how often a class prediction is right. Recall measures how many real examples of a class are identified. F1 balances precision and recall. **Macro** scores average equally across the six emotions, which matters because the class distribution is uneven.

## Confusion matrices: which emotions get mixed up?

Rows are **true emotions** and columns are **predicted emotions**. A diagonal cell is a correct prediction; an off-diagonal cell is a mistake. Darker cells here correspond to larger counts (not necessarily higher error rates).

### CNN confusion matrix

![CNN confusion matrix](assets/cnn_confusion_matrix.png)

- The CNN recognizes **643 of 688 joy** sentences and **558 of 579 sadness** sentences.
- It labels **36 joy sentences as love**, suggesting overlap between affectionate and happy wording.
- It recognizes **46 of 65 surprise** sentences; **13** surprise sentences are labeled fear.
- Why might surprise be harder? The training split has only **572 original surprise examples**, and excitement, shock and fear can share expressions. Both class imbalance and linguistic ambiguity may contribute.

### LSTM confusion matrix

![LSTM confusion matrix](assets/lstm_confusion_matrix.png)

- The LSTM correctly identifies **629 of 688 joy** and **557 of 579 sadness** sentences.
- It labels **43 joy** sentences as love, more than the CNN's 36.
- It identifies **52 of 65 surprise** sentences, better than CNN's 46; however, it also predicts surprise for **20 fear** sentences.
- This shows a trade-off: LSTM has **higher surprise recall** (80% vs ~71%) but does not achieve higher overall accuracy or macro F1.

### Per-class F1-score (test set)

| Emotion | CNN | LSTM | Observation |
|---|---:|---:|---|
| Anger | 0.92 | 0.91 | Both strong |
| Fear | **0.89** | 0.87 | CNN better |
| Joy | **0.95** | 0.94 | Both strong |
| Love | **0.84** | 0.82 | Often confused with joy |
| Sadness | **0.96** | 0.95 | Highest scoring class for both |
| Surprise | 0.69 | **0.71** | Hardest F1 class; LSTM slightly better |

## Trying new sentences

The `predict_emotion` function processes a sentence with the same cleaning and vectorization steps used for training, runs the requested network and returns the highest-probability class.

| Sentence | CNN result | LSTM result |
|---|---|---|
| `I am so happy today` | joy (0.928) | joy (0.774) |
| `I feel lonely and depressed` | sadness (0.994) | sadness (0.999) |
| `I am extremely angry about this` | anger (0.963) | anger (0.956) |
| `I am scared of the darkness` | fear (0.981) | fear (0.949) |
| `I love spending time with you` | joy (0.246) | love (0.233) |
| `I never expected this to happen` | joy (0.239) | anger (0.376) |

**Why the disagreement?** The first four examples contain relatively direct emotional cues. The final two are less certain for these models: `love` and `joy` overlap in language, and `I never expected this to happen` has no clear positive or negative context. A top-class value of **0.239** or **0.233** is not convincing evidence of a reliable prediction. Softmax values are model scores interpreted as probabilities, not necessarily calibrated confidence.

### Custom expression: “I feel so blue today”

![CNN prediction scores for I feel so blue today](assets/cnn_custom_sentence.png)

The CNN predicts **sadness**, which matches the usual meaning of the idiom *feeling blue*. However, the model's highest output is only around **0.25**, with joy close behind at about **0.24**. The model is **uncertain** despite selecting a plausible label.

**Why might this happen?** Idioms, limited training examples for a particular phrase and the absence of wider context can all make classification harder. The figure alone cannot tell us which of those causes is responsible. The notebook does not measure word-level explanations or calibrated confidence.

## What did we learn for Pattern Recognition?

1. **Representing patterns:** Natural-language sentences cannot go straight into a conventional numeric neural-network input. Tokenization and embeddings create useful numerical representations.
2. **Learning features:** CNN learns short sequential patterns; LSTM learns representations across token order. Neither needs a manually written emotion keyword list.
3. **Classifying patterns:** A six-unit Softmax layer maps learned features to one emotion class.
4. **Generalization:** Separate validation and test sets, leakage checks, and early stopping help assess performance on unseen sentences.
5. **Recognizing limitations:** Emotion classes overlap, the dataset is imbalanced, and uncertain examples are difficult even when overall accuracy is high.

## Limitations and possible improvements

- **Class imbalance:** Consider class weights or collecting more rare-class data. Do not alter the test distribution when comparing with these results.
- **Single-label classification:** A sentence can express several emotions, but this dataset assigns one label per sentence.
- **Ambiguity and idioms:** Statements without context (or with figurative language) may receive unstable outputs.
- **Overfitting:** Both models' validation loss worsens after a few epochs. Additional regularization or simpler models could be tested using validation data.
- **Padding difference:** CNN did not use the same masking setup as LSTM. A future controlled experiment could address padding and repeat training across multiple seeds.
- **Dataset-label noise:** Some examples may admit more than one reasonable emotion; inspect questionable mistakes without using the test set to repeatedly tune models.
- **English only:** These models were trained on the supplied English-language dataset; performance on Arabic or other languages has not been measured.

## Conclusion

Both deep-learning models learned useful features for recognizing emotions in short English sentences. The **CNN achieved 92.45% test accuracy and 0.8760 macro F1**, outperforming the **LSTM's 91.24% and 0.8666 macro F1**. CNN also made **24 fewer errors** across the 1,986 test sentences. LSTM recognized more surprise examples, illustrating that overall accuracy does not tell the full story.

For this dataset and training setup, **CNN is the better overall choice**. More broadly, the project demonstrates the Pattern Recognition pipeline: **clean the inputs, represent text numerically, learn features, classify patterns, and evaluate generalization**.

## Repository files

```text
emotion-recognition-github/
├── README.md                      # This project explanation
├── code.ipynb                     # Original notebook; all 45 cells retained
├── train.txt                      # Dataset files provided for this project
├── val.txt
├── test.txt
├── requirements.txt
├── assets/
│   ├── emotion_class_distribution.png
│   ├── sentence_length_distribution.png
│   ├── cnn_training_curves.png
│   ├── lstm_training_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   └── cnn_custom_sentence.png
└── results/
    └── model_comparison.csv
```

The notebook additionally creates a `models/` directory for trained weights and may write confusion-matrix images and vocabulary metadata to `results/` and `models/` when run.

**Data attribution:** Kaggle, [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). Before publishing dataset files in a public GitHub repository, check the dataset's redistribution terms. You can remove `train.txt`, `val.txt`, and `test.txt` from the public repository and ask readers to download them from Kaggle instead.
