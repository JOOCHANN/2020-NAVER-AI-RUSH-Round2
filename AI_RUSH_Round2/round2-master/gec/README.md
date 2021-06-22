# Korean Grammatical Error Correction

## Task
User-generated texts are quite noisy because people do not care about grammatical rules much in their writings.
Moreover, grammatical rules (especially of Korean) are often tricky even for natives. 
The goal of this task is to build a grammatical error correction (GEC) system that can recommend more appropriate and grammatically correct sentences given user inputs. 


## Dataset Description
There are three datasets (`rush1-1`, `rush1-2`, and `rush1-3`) corresponds to each week.
Structure of each dataset is as follows:
```
\_ train
    \_ train_data (folder)
        \_ train_data           # Noisy sentences for training
        \_ train_annotation     # Annotation of error types
        \_ train_corpus         # Raw corpus without correction
    \_ train_label              # Corrected sentences for training
\_ test
    \_ test_data                # Noisy sentences for evaluation
    \_ test_label               # Corrected sentences for evaluation
\_ test_submit
    \_ test_data                # Noisy sentences for test submission
    \_ test_label               # Corrected sentences for test submission
```
Noisy sentences and their corrected sentences can be mapped line-by-line.


## Evaluation Metric
We use corpus-level [GLEU](https://www.aclweb.org/anthology/P07-1044/) score for the evaluation of your GEC system with [`nltk.translate.gleu_score.corpus_gleu`](https://www.nltk.org/_modules/nltk/translate/gleu_score.html) script from the `nltk` library.
GLEU is the modified version of BLEU and widely used metric for automatic evaluation of grammatical error correction.


## Baseline
Our baseline is based on a [Transformer](https://arxiv.org/abs/1706.03762) sequence-to-sequence model that generates a correction given a noisy sentence.
You may exploit additional data (`train_annotation` and `train_corpus`) to train a better model, although we did not use them.
The baseline model achieves about `83.4` GLEU score, and takes less than `i` minutes inference time for the submission of dataset `rush1-{i}`.
Runtime is not considered as a criterion for the evaluation.


## Model Training
```
nsml run -d {DATASET} -e train.py
``` 
You may modify the arguments using `-a` option as you wish.


## Model Submission
```
nsml submit {SESSION} {CHECKPOINT}
``` 
