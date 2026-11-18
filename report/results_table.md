# Main results (test set, micro-averaged over all 217 labels)

Mean ± sample standard deviation over seeds 42–44 where three seeds were run;
single-seed runs show one value. P = precision, R = recall.

## v1

| model | seeds | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|---|
| Label-frequency floor | 1 | 0.690 | 0.536 | 0.603 | 0.576 | 0.706 | 0.634 |
| TF-IDF + Random Forest (baseline) | 3 | 0.688 ± 0.002 | 0.485 ± 0.002 | 0.569 ± 0.001 | 0.555 ± 0.001 | 0.706 ± 0.002 | 0.621 ± 0.001 |
| MiniLM embeddings + LR | 1 | 0.689 | 0.537 | 0.604 | 0.579 | 0.700 | 0.634 |
| mpnet embeddings + LR | 1 | 0.689 | 0.536 | 0.603 | 0.581 | 0.698 | 0.634 |
| RoBERTa, fine-tuned | 1 | 0.683 | 0.549 | 0.609 | 0.570 | 0.717 | 0.635 |
| CodeBERT, fine-tuned | 1 | 0.680 | 0.554 | 0.611 | 0.580 | 0.701 | 0.635 |
| CodeBERT + repo name, fine-tuned | 1 | 0.682 | 0.556 | 0.612 | 0.581 | 0.700 | 0.635 |

## v2

| model | seeds | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|---|
| Label-frequency floor | 1 | 0.701 | 0.238 | 0.355 | 0.492 | 0.433 | 0.461 |
| TF-IDF + Random Forest (baseline) | 3 | 0.728 ± 0.004 | 0.310 ± 0.003 | 0.435 ± 0.004 | 0.589 ± 0.017 | 0.509 ± 0.014 | 0.546 ± 0.002 |
| MiniLM embeddings + LR | 1 | 0.658 | 0.461 | 0.542 | 0.614 | 0.552 | 0.582 |
| mpnet embeddings + LR | 1 | 0.675 | 0.477 | 0.559 | 0.643 | 0.548 | 0.592 |
| RoBERTa, fine-tuned | 3 | 0.676 ± 0.014 | 0.491 ± 0.012 | 0.568 ± 0.004 | 0.605 ± 0.020 | 0.587 ± 0.032 | 0.595 ± 0.007 |
| CodeBERT, fine-tuned | 3 | 0.686 ± 0.009 | 0.520 ± 0.013 | 0.592 ± 0.005 | 0.633 ± 0.014 | 0.600 ± 0.018 | 0.616 ± 0.005 |
| CodeBERT + repo name, fine-tuned | 3 | 0.703 ± 0.007 | 0.569 ± 0.017 | 0.629 ± 0.008 | 0.640 ± 0.015 | 0.661 ± 0.024 | 0.650 ± 0.005 |

Precision-constrained threshold (CodeBERT + repo name; best validation F1 with validation precision ≥ 0.75), 3 seeds: precision 0.758 ± 0.003, recall 0.464 ± 0.011, micro-F1 0.575 ± 0.009.

