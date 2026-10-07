# NeuroFocus

### Subject-Independent EEG Affective State Classification

NeuroFocus is an EEG machine learning project investigating whether **valence** and **arousal** can be predicted from brain activity recorded from people the model has never seen before.

Using the **DEAP EEG dataset**, the project compares classical spectral features, differential entropy, and an EEGNet-style neural network under a strict **participant-independent evaluation protocol**.

The project intentionally prioritizes cross-subject generalization and leakage-resistant evaluation over maximizing headline accuracy.

---

## Live Demo

An interactive **Hugging Face Spaces** demo allows users to explore EEG-derived features and model predictions.

**Hugging Face Demo:** https://huggingface.co/spaces/Samer17/NeuroFocus

The demo provides an interactive view of:

- Theta, Alpha, Beta, and Gamma EEG activity
- EEG-derived feature representations
- Valence predictions
- Arousal predictions
- Ground-truth labels
- Prediction confidence
- Correct and incorrect cross-subject predictions

Rather than presenting the system as a reliable "emotion detector," the demo acts as an **EEG experiment explorer**, showing both what the models learn and where they fail to generalize.

---

## Results at a Glance

### Model Selection — Validation Participants

| Representation | Valence Balanced Accuracy | Arousal Balanced Accuracy |
|---|---:|---:|
| **Band Power** | **58.50%** | **59.22%** |
| Differential Entropy | 55.50% | 56.34% |
| EEGNet-style Network | 50.00% | 53.75% |

The strongest validation pipelines were:

- **Valence:** Band Power → StandardScaler → Logistic Regression
- **Arousal:** Band Power → StandardScaler → RBF SVM

These models were selected before evaluating the held-out test participants.

### Final Held-Out Test

| Target | Model | Accuracy | Balanced Accuracy |
|---|---|---:|---:|
| Valence | Logistic Regression | 54.50% | **52.70%** |
| Arousal | RBF SVM | 49.50% | **52.50%** |

The final test consists of EEG from **five completely unseen participants**.

The validation-to-test drop is an important finding of the project: EEG patterns learned from one group of people transferred only weakly to new individuals.

---

## Tech Stack

**Python · NumPy · Pandas · SciPy · scikit-learn · PyTorch · EEG Signal Processing · Welch PSD · Logistic Regression · SVM · Random Forest · EEGNet**

---

## Repository Structure

~~~text
NeuroFocus/
│
├── models/
│   ├── valence_model.joblib
│   ├── arousal_model.joblib
│   ├── eegnet_valence.pt
│   └── eegnet_arousal.pt
│
├── notebooks/
│   └── original_cogs189project.ipynb
│
├── src/neurofocus/
│   ├── data.py
│   ├── labels.py
│   ├── features.py
│   ├── extract.py
│   ├── train.py
│   ├── compare_models.py
│   ├── eegnet_data.py
│   ├── eegnet_model.py
│   ├── train_eegnet.py
│   └── final_evaluate.py
│
├── requirements.txt
├── .gitignore
└── README.md
~~~

---

# Running NeuroFocus

## 1. Clone the Repository

~~~bash
git clone <repository-url>
cd NeuroFocus
~~~

## 2. Install Dependencies

~~~bash
pip install -r requirements.txt
~~~

## 3. Download DEAP

NeuroFocus uses the preprocessed Python version of the **DEAP Dataset for Emotion Analysis using Physiological Signals**.

Download the dataset separately and place the participant `.dat` files inside:

~~~text
data_preprocessed_python/
├── s01.dat
├── s02.dat
├── ...
└── s32.dat
~~~

The raw DEAP dataset is intentionally excluded from this repository.

## 4. Extract Band-Power Features

~~~bash
PYTHONPATH=src python -m neurofocus.extract --feature-type bandpower
~~~

## 5. Extract Differential-Entropy Features

~~~bash
PYTHONPATH=src python -m neurofocus.extract --feature-type de
~~~

## 6. Compare Classical Models

For band-power features:

~~~bash
PYTHONPATH=src python -m neurofocus.compare_models \
    --features data/processed/eeg_features.csv
~~~

For differential-entropy features:

~~~bash
PYTHONPATH=src python -m neurofocus.compare_models \
    --features data/processed/eeg_de_features.csv
~~~

## 7. Train EEGNet

Valence:

~~~bash
PYTHONPATH=src python -m neurofocus.train_eegnet \
    --target valence \
    --epochs 30 \
    --patience 5
~~~

Arousal:

~~~bash
PYTHONPATH=src python -m neurofocus.train_eegnet \
    --target arousal \
    --epochs 30 \
    --patience 5
~~~

## 8. Run Final Evaluation

~~~bash
PYTHONPATH=src python -m neurofocus.final_evaluate
~~~

---

# The Experiment

## What Are We Trying to Predict?

EEG measures electrical activity recorded from the scalp.

NeuroFocus investigates whether patterns in this activity contain enough information to distinguish two dimensions of affect:

- **Valence:** negative ↔ positive
- **Arousal:** low activation ↔ high activation

DEAP provides self-reported ratings from 1–9.

For this project, a threshold of **5** converts these ratings into binary targets:

| Target | Class 0 | Class 1 |
|---|---|---|
| Valence | Negative | Positive |
| Arousal | Low | High |

---

# Dataset

The DEAP dataset contains:

- **32 participants**
- **40 trials per participant**
- **1,280 total trials**
- **32 EEG channels**
- **128 Hz sampling rate**
- **60 seconds of stimulus EEG per trial**
- Valence, arousal, dominance, and liking ratings

Each preprocessed recording also contains a three-second pre-stimulus baseline.

NeuroFocus removes this baseline before feature extraction.

---

# Why Participant-Independent Evaluation?

This became one of the most important decisions in the project.

EEG signals are highly person-specific.

If recordings from the same person appear in both training and testing, a model can potentially learn characteristics of that individual's EEG rather than patterns that generalize between people.

For example:

~~~text
EASIER SETTING

Participant A ──→ Training
Participant A ──→ Testing

Participant B ──→ Training
Participant B ──→ Testing
~~~

The model has already encountered EEG from the people it is later asked to classify.

NeuroFocus instead evaluates:

~~~text
HARDER SETTING

22 Participants ──→ Training

5 New Participants ──→ Validation

5 New Participants ──→ Testing
                        ↑
                  Never seen before
~~~

The actual fixed split contains:

| Split | Participants | Trials |
|---|---:|---:|
| Training | 22 | 880 |
| Validation | 5 | 200 |
| Test | 5 | 200 |

No test participant appears during training or model selection.

---

# Why Window-Level Splitting Can Be Misleading

EEG recordings are often divided into smaller windows to create additional training examples.

NeuroFocus also uses windows during feature extraction.

However, adjacent windows from the same EEG recording are highly related.

If windows from the same recording are randomly distributed between training and testing:

~~~text
60-second EEG trial
        ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Window 1│ Window 2│ Window 3│ Window 4│
└─────────┴─────────┴─────────┴─────────┘
     ↓          ↓          ↓          ↓
   Train      Train       Test       Train
~~~

the test data may be extremely similar to data the model already encountered.

NeuroFocus avoids this problem by performing the split at the **participant level** and representing each trial independently.

---

# Experiment 1 — Spectral Band Power

The first approach uses an interpretable EEG representation: **frequency-band power**.

After removing the three-second baseline, each 60-second trial is divided into overlapping:

- **2-second windows**
- **1-second stride**

Welch's power spectral density estimate is calculated for every EEG channel.

Power is extracted from:

| Band | Frequency |
|---|---:|
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

With 32 channels:

~~~text
32 channels × 4 frequency bands = 128 features
~~~

Features are averaged across the windows in each trial.

### Why try band power?

Raw EEG contains thousands of measurements per channel.

Band-power extraction reduces this into a compact representation describing how signal energy is distributed across physiologically meaningful frequency ranges.

Instead of forcing a model to discover everything from raw EEG, we provide it with **128 structured EEG features**.

Four classical classifiers were compared:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest

### Result

The strongest models reached:

~~~text
Valence: 58.50% balanced validation accuracy
Arousal: 59.22% balanced validation accuracy
~~~

This became our baseline.

---

# Experiment 2 — Differential Entropy

The next question was:

> What if simple band power is throwing away useful information?

We tested **Differential Entropy (DE)** as an alternative EEG representation.

Each channel is filtered into the same four frequency bands, and differential entropy is estimated from the variance of the band-limited signal.

This again produces:

~~~text
32 channels × 4 bands = 128 features
~~~

### Why try it?

Differential entropy is commonly used in EEG affective-computing research.

It provides a different representation of the distribution of activity within each frequency band.

By keeping the participant split and classifiers unchanged, we could directly test:

> Does changing the EEG representation improve generalization?

### Result

| Target | Band Power | Differential Entropy |
|---|---:|---:|
| Valence | **58.50%** | 55.50% |
| Arousal | **59.22%** | 56.34% |

It did not.

Changing from band power to differential entropy alone was not enough to solve the cross-participant problem.

---

# Experiment 3 — EEGNet

That led to another hypothesis:

> Maybe manually designed EEG features are the bottleneck.

Band power and differential entropy decide in advance what information the classifier receives.

A neural network could potentially learn useful representations directly from the EEG.

We therefore implemented a compact **EEGNet-style convolutional architecture** containing:

- Temporal convolution
- Depthwise spatial convolution across EEG channels
- Separable convolution
- Batch normalization
- ELU activation
- Average pooling
- Dropout
- Final classifier

The network contains approximately **1.6K trainable parameters**.

### Why EEGNet?

A very large neural network would be difficult to justify with only 1,280 EEG trials.

EEGNet was specifically designed as a compact architecture for EEG-based brain-computer interfaces.

Its temporal and spatial convolutions provide an EEG-specific inductive bias while requiring far fewer parameters than a conventional deep CNN.

### Result

| Representation | Valence | Arousal |
|---|---:|---:|
| Band Power | **58.50%** | **59.22%** |
| Differential Entropy | 55.50% | 56.34% |
| EEGNet | 50.00% | 53.75% |

The neural model did not outperform the simpler spectral baseline.

---

# Why Didn't Deep Learning Win?

Deep learning is powerful when enough data exists to learn robust representations.

NeuroFocus has:

~~~text
1,280 trials
32 people
~~~

The model must learn patterns that are:

1. related to valence or arousal,
2. consistent between different people,
3. robust to EEG noise,
4. robust to subjective labels,
5. and learnable from a relatively small dataset.

The handcrafted spectral representation provides the model with a strong prior about what information to examine.

In this setting, **more model complexity did not automatically produce better generalization**.

---

# Final Model Selection

The experiments selected:

### Valence

~~~text
EEG
 ↓
Band-Power Extraction
 ↓
128 Features
 ↓
StandardScaler
 ↓
Logistic Regression
~~~

### Arousal

~~~text
EEG
 ↓
Band-Power Extraction
 ↓
128 Features
 ↓
StandardScaler
 ↓
RBF SVM
~~~

After selecting these models using validation participants, model selection was frozen.

Training and validation were then combined:

~~~text
880 training trials
       +
200 validation trials
       =
1,080 development trials
~~~

The models were retrained and evaluated once on the remaining 200 trials.

---

# Why Is Final Accuracy Only Around 52%?

This deserves more explanation than simply showing the number.

Imagine trying to recognize handwriting.

If you study 100 writing samples from Sam and are then asked to classify another sample from Sam, you already know what Sam's handwriting looks like.

Now imagine learning from 27 people and suddenly receiving handwriting from five people you have **never encountered before**.

That is a much harder problem.

EEG has a similar issue.

NeuroFocus deliberately evaluates the second scenario.

## 1. EEG differs between people

Two people experiencing similar affective states do not necessarily produce identical EEG patterns.

Anatomy, electrode measurements, physiology, individual baselines, noise, and responses to stimuli can all influence the signal.

## 2. There are only 32 participants

DEAP contains 1,280 trials, but those trials come from only **32 different people**.

For cross-subject learning, the diversity of people matters enormously.

## 3. The labels are subjective

Valence and arousal are self-reported.

A rating of 6 from one person does not necessarily represent exactly the same internal state as a 6 from another.

## 4. Thresholding creates an artificial boundary

A continuous psychological rating is converted into a binary class.

Two ratings close to the threshold can therefore become different classes despite being very similar.

## 5. The test participants are genuinely unseen

The model receives no EEG from those participants during model selection.

That removes an important source of information that can make EEG classification considerably easier.

---

# Why Do Some EEG Papers Report Much Higher Accuracy?

EEG emotion-recognition papers sometimes report results far above those observed here.

However:

> **Two accuracy numbers are only comparable when the experiments behind them are comparable.**

One major distinction is between **subject-dependent** and **subject-independent** evaluation.

## Subject-Dependent

~~~text
Participant A ──→ Train
Participant A ──→ Test
~~~

The model has previously observed EEG from the test person.

This is generally an easier problem.

## Subject-Independent

~~~text
Participants A–V ──→ Train

Participant Z ──────→ Test
                      ↑
                   unseen
~~~

The model must transfer what it learned between different people's brains.

This is the setting evaluated by NeuroFocus.

Published EEG reviews identify **inter-subject variability and distribution shift** as major challenges in subject-independent emotion recognition.

---

# What About High-Performing Cross-Subject Research?

Some research genuinely achieves stronger cross-subject results.

Modern approaches specifically attack the differences between participants using techniques such as:

- Domain adaptation
- Adversarial learning
- Distribution alignment
- Graph neural networks
- EEG electrode topology
- Transfer learning
- Self-supervised learning
- Subject-invariant representation learning

Conceptually:

~~~text
Participant A ─┐
Participant B ─┤
Participant C ─┤
Participant D ─┘
       ↓
Learn shared EEG structure
       ↓
Reduce participant-specific information
       ↓
Unseen Participant E
~~~

These approaches address a different problem than simply replacing Logistic Regression with a larger neural network.

Published results also use different:

- participant splits
- cross-validation strategies
- label thresholds
- preprocessing pipelines
- window sizes
- evaluation metrics
- domain-adaptation procedures
- target-participant information
- hyperparameter-selection strategies

Therefore, headline accuracy values across EEG papers should not be compared without examining the underlying protocol.

---

# Why Balanced Accuracy Matters

NeuroFocus uses **balanced accuracy** as its primary evaluation metric.

This matters because the DEAP-derived classes are not perfectly balanced.

For example, a classifier could achieve a seemingly reasonable ordinary accuracy simply by becoming very good at predicting the majority class while performing poorly on the minority class.

Balanced accuracy instead calculates performance across both classes so that each class contributes equally to the final score.

For binary classification:

~~~text
Balanced Accuracy =
(Recall of Class 0 + Recall of Class 1) / 2
~~~

This makes it more informative than raw accuracy when evaluating an imbalanced classification problem.

Our final held-out results were:

~~~text
Valence: 52.70% balanced accuracy
Arousal: 52.50% balanced accuracy
~~~

These results do not demonstrate that EEG affect recognition is solved.

They demonstrate almost the opposite:

> **Spectral EEG patterns that appear promising during development do not necessarily transfer strongly to completely unseen individuals.**

---

# Why We Didn't Keep Tuning After the Test

Once the final test results were observed, we could have done this:

~~~text
See test score
     ↓
Change features
     ↓
Test again
     ↓
Change model
     ↓
Test again
     ↓
Tune parameters
     ↓
Test again
~~~

Eventually, the number might improve.

But we would gradually be optimizing our decisions around the supposedly unseen test participants.

The test set would stop being a true test.

Instead, NeuroFocus treats the frozen evaluation as final.

The lower result remains because preserving the integrity of the experiment is more important than producing a larger number.

---

# What We Learned

The project progressed through a sequence of hypotheses:

~~~text
Can EEG predict affect for unseen people?
                ↓
       Band-Power Features
                ↓
        ~59% validation
                ↓
Would another EEG representation help?
                ↓
      Differential Entropy
                ↓
          No improvement
                ↓
Can deep learning discover better features?
                ↓
             EEGNet
                ↓
          No improvement
                ↓
Freeze strongest classical models
                ↓
Evaluate completely unseen participants
                ↓
      ~52–53% balanced accuracy
                ↓
Cross-subject EEG generalization remains
the dominant challenge.
~~~

The central result of NeuroFocus is therefore not simply a classification score.

It is the observation that:

> **Changing the feature representation or increasing model complexity did not solve the participant distribution shift.**

That suggests future improvements should target **subject invariance directly**.

---

# What I Would Try Next

If the objective shifted from establishing a controlled baseline to maximizing cross-subject performance, the next experiments would focus specifically on reducing differences between participants.

Potential directions include:

- Domain-Adversarial Neural Networks
- CORAL and other distribution-alignment techniques
- Riemannian covariance features
- Graph neural networks using EEG electrode topology
- Self-supervised EEG pretraining
- Leave-One-Subject-Out evaluation
- Subject-invariant representation learning
- Larger multi-subject EEG datasets

The experiments in NeuroFocus suggest that the primary bottleneck is not simply classifier size.

It is learning an EEG representation that transfers between people.

---

# Limitations

NeuroFocus is an experimental machine-learning project, **not a clinical emotion-detection system**.

Important limitations include:

- DEAP contains only 32 participants.
- EEG differs substantially between individuals.
- Valence and arousal are subjective self-reported labels.
- Continuous ratings are converted into binary classes.
- The current models do not explicitly remove participant-specific variation.
- No target-subject domain adaptation is performed.
- EEGNet is trained on a relatively small dataset.
- Final cross-subject performance remains only slightly above chance.

---

# References

1. Koelstra, S. et al. (2012). **DEAP: A Database for Emotion Analysis Using Physiological Signals.** *IEEE Transactions on Affective Computing.*

2. Lawhern, V. J. et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.** *Journal of Neural Engineering.*

3. **A Survey on Physiological Signal-Based Emotion Recognition.** Discussion of subject-dependent and subject-independent affective-computing evaluation.

4. **Challenges in EEG Emotion Recognition.** Discussion of inter-subject variability, EEG segmentation, and generalization.

5. **EEG Affective Recognition with a Neuroscience Perspective.** Review of subject-independent learning, domain adaptation, and EEG representations.

6. **Cross-Subject Generalization for EEG Emotion Recognition.** Review of domain adaptation, distribution alignment, representation learning, and subject-independent evaluation.

7. Sadegh-Zadeh et al. (2026). **Standard Accuracy in EEG Emotion Recognition Is Predictable from Class Distribution Alone: A Systematic Analysis of Labelling Threshold Effects in the DEAP Dataset.** *Frontiers in Neuroinformatics.*

---

# Project Background

NeuroFocus began as a UC San Diego **COGS 189** project exploring machine learning with EEG signals.

The original prototype contained exploratory and synthetic classification experiments.

The project was subsequently rebuilt around:

- Real DEAP valence/arousal labels
- Reproducible EEG preprocessing
- Trial-level spectral features
- Differential entropy
- Participant-independent splitting
- Leakage-resistant evaluation
- Classical ML model comparison
- EEGNet-style neural modeling
- Frozen held-out testing
- Saved inference models
- Interactive Hugging Face deployment

The redesign changed the central question from:

> "Can we train an EEG classifier?"

to:

> **"Does the EEG representation actually generalize to someone the model has never seen?"**

For the models tested here, the answer is:

**Only weakly — and understanding why is the central finding of the project.**
