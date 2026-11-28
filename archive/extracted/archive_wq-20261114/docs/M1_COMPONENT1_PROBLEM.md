# M1 Component-One Problem

## Selected problem

**The parametric RankUp/B2 head ignores a strong, already-learned local VAD geometry, especially for tail and ambiguous samples.**

The failure is not “unlabeled data is useless”. It is that the model’s **decision rule** discards neighbour evidence that is demonstrably present in the UVR representation.

## Why this problem is selected

1. **It is large and measurable.** B2 macro MAE is 0.16762; D Pearson is only 0.403. Tail bins have MAE 0.21–0.32 versus 0.07–0.12 in the middle. The conditional-mean compression is consistent across folds.
2. **The local information exists.** On TRAIN labeled UVR features, pairwise feature distance correlates **0.946** with pairwise VAD distance; 1-NN VAD distance is 0.029 versus 0.153 for random pairs. A model that ignores this is leaving usable structure unused.
3. **It has a concrete, testable mechanism.** A retrieval branch over labeled and unlabeled neighbours changes the decision rule without changing the encoder; it can be ablated by removing the unlabeled pool or randomising neighbours.
4. **It is distinct from the failed families.** It is not ARC pair pseudo-labels, not VAT, not distribution distillation, not feature mixup, not weight soup. It is non-parametric neighbourhood evidence.
5. **It can use the unlabeled corpus.** The unlabeled pool is included as a pseudo-labeled memory; the ablation “labeled-only memory” tests whether unlabeled texts add independent value.

## What the literature already covers

- RankUp and SSR-RCUS use pairwise ranking/mixup to shape the encoder, but do not use a retrieval memory at inference.
- kNN regression and label propagation are classical; retrieval-augmented models are common in NLP. The specific open question here is whether **RankUp’s UVR encoder + pseudo-labeled unlabeled memory** can correct the continuous VAD decision rule under 500 labels.
- This is therefore a **new combination/mechanism for the RankUp setting**, not a claim that kNN itself is new.

## Mechanism hypothesis

For a text \(x\):

\[
p(x) = (1-\lambda) f_\theta(x) + \lambda \sum_{i \in \mathcal{N}_k(x)} w_i\, y_i
\]

where:

- \(f_\theta(x)\) is the B2/RankUp parametric VAD;
- \(\mathcal{N}_k(x)\) contains labeled examples with true VAD and unlabeled examples with teacher pseudo-VAD;
- \(w_i \propto \exp(-d_i/\tau)\) or \(1/(d_i+\epsilon)\);
- \(\lambda\) and \(\tau\) are selected **inside TRAIN only** (cross-fitting), never on the outer holdout.

Hypothesis: the retrieval term reduces tail compression and improves Pearson because it preserves local ordering; unlabeled pseudo-neighbours add coverage that labeled-only retrieval lacks.

## Counter-hypotheses

- The gain may come only from labeled retrieval, not unlabeled data.
- Pseudo-label noise may cancel the benefit.
- The retrieval may improve Pearson but not MAE.
- The gain may not replicate across folds/seeds.

## Success criteria for this problem

The component is accepted only if:

1. matched B2 vs B2+retrieval shows a pre-registered primary gain;
2. the other primary metric is non-inferior;
3. the **unlabeled-memory ablation** reduces the gain;
4. the **random-neighbour ablation** removes the gain;
5. seed0 five-fold direction is stable, then seed1/2 confirm;
6. the gain is not just extra classifier capacity or extra training.

If retrieval fails these tests, the problem statement “underused local geometry” is not sufficient, and the next candidate must target a different failure point.
