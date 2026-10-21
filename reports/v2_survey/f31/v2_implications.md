# What F-31 implies for v2 (draft note, coordinator, 2026-10-21)

**Status: for John to read. Nothing proposed here runs until John approves a design.**

## 1. What F-31 found, and its limits

**Primaries** (pre-registered; 20 seeds per arm; H9):
- **Control** (the frozen-ensemble recipe): 4× is better than 1× by 0.99 kt, 97.5% CI [0.81, 1.18] → **BETTER**.
- **Atlantic** (the N5 atlantic list): 4× is better than 1× by 1.36 kt, 97.5% CI [1.03, 1.70] → **BETTER**, and still BUDGET-LIMITED at 4× (12/20).

**Supporting figures (descriptive or sensitivity):**
- 4× beats 1× on all 9 H9 storms, in both arms.
- >96 kt MAE improves by about 1.4–1.6 kt (S4). >96 kt bias does not change significantly (S3).
- For the control, 2× gets about two thirds of the gain: 0.67 kt, below the SESOI.
- With the smoothed selection rule, the control's gain is 0.52 kt, still BETTER. Part of the native 0.99 kt may be selection optimism from picking among 99 epochs rather than 25 (design §12). The smoothed comparison changes the rule and the budget together, though, so the gap isn't pure optimism either.
  - The point estimates per model run from 0.52 kt (smoothed) to 0.99 kt (native). The smoothed 97.5% CI reaches down to 0.08 kt.
- **Ensembles gained less than single models (descriptive, on H9).** The control's 20-member ensemble went from 8.53 kt (1×) to 8.05 kt (4×), a gain of 0.48 kt. The members gained 0.99 kt: averaging already removes part of what longer training fixes. **Nothing here estimates the gain on new storms.**

**Limits** (design §4, §7, §12):
- H9 has been reused many times, and these are 9 storms. Absolute H9 levels are not estimates for new storms.
- The pre-registered reading of P1 BETTER is "the frozen recipe was budget-limited at 1640 steps, on H9, for this code version". It is not that the frozen ensemble should be replaced.
- **A longer-trained ensemble is a new model.** It needs its own test on an unspent set.

## 2. A longer-trained ensemble recipe for v2

- **Candidate recipe:**
  - the control configuration (`storms_train_expanded.txt`);
  - trained 4× (99 epochs, about 6600 steps), with train.py's native selection on validation;
  - 10 members, new seeds.
- **Why 4× and not more:**
  - It is the budget F-31 tested.
  - The control was no longer BUDGET-LIMITED at 4× (5/20), so validation had levelled off.
  - F-31 does not extend. Any longer budget, or any budget between 2× and 4×, would be chosen on validation only, in a new design. H9 should not be used again to pick a budget.
- **Fixed in advance in that design:**
  - the budget;
  - the selection rule: native, or smoothed if John prefers the more conservative rule;
  - the seeds;
  - train.py pinned. The learning rate must stay independent of total epochs (§11), and any schedule change reopens the budget question.
- **Compute:** F-31's control training at 4× took about 7–9 minutes per run, so a 10-member ensemble is about 1.2–1.5 hours.
  - **For the design:** new seeds, or reuse F-31's 20 control 4× checkpoints.
- **The atlantic list:**
  - It is still improving at 6600 steps.
  - At 4×, S1 shows it 0.26 kt better than the control: significant, but below the SESOI. The arms' training is asymmetric (John's pre-look note), and N5's confounds remain.
  - This doesn't make the added Atlantic storms a v2 lever by itself. Revisiting them would need its own design, with each arm trained to a plateau (each arm's budget chosen on validation).

## 3. Fit with the East Pacific plan (H-EP, protocol 3901f4cf)

- **Slots:**
  - E1: the frozen ensemble (1×);
  - E2c and E2a: the N5 arms (1×). These are open, since N5 closed with a results entry;
  - E3: the later Atlantic + East Pacific arm, with design and checkpoints frozen by about the end of January 2027.
  - F-31's checkpoints fit no slot and are never scored on H-EP.
- **Implication for E3:**
  - If E3 trains at 4× and is compared with E1 (1×), the East Pacific effect is mixed up with the budget effect.
  - **Recommendation:** E3's design adopts the same fixed budget as the v2 recipe, and declares a comparator: the Atlantic-only recipe at that same budget. The protocol already allows this ("plus its comparator if its design declares one").
  - That comparator can be the §2 v2 Atlantic ensemble itself, so one training serves both purposes.
- **Allowed:** using F-31 in E3's design is allowed. F-31 used H9 and validation only, and no H-EP information.
- **Order:** E3 needs H-EP gates 1–7 first, starting with B-v2-1, your antimeridian fix, and B-v2-2, the East Pacific HURDAT2 cache. The v2 Atlantic recipe needs none of those gates and could be trained earlier, once its design is approved.

## 4. Fit with the 2026 clean test (design §4 option D)

- **It is the only unspent set for confirmation.** A 2026-season set is the only place a longer-trained model can be shown better on new storms.
- **Constraints:**
  - no 2026 imagery may be downloaded or cropped under the current rules (I4);
  - 2026 best tracks are not final before spring 2027;
  - the protocol must be written and committed before any 2026 storm is built, with a single pass.
- **Suggested shape, for that protocol:**
  - a pre-registered paired comparison of the frozen ensemble (1×) and the v2 4× ensemble, both frozen before any 2026 build;
  - a SESOI fixed on what difference would matter, not taken from F-31's H9 effect;
  - power computed from the number of 2026 storms and the between-storm spread, not from frames. With two fixed ensembles, the comparison replicates over storms, and frames within a storm are dependent.
    - F-31's H9 ensemble gain (0.48 kt) may be used only as a labelled, H9-derived planning effect for that power calculation.
  - Whether 2026 East Pacific storms are included, and how that interacts with E3, is a question for that design.
- **Timing fits the E3 deadline:** models frozen by about January 2027 could be scored once on H-EP and once on 2026, each under its own protocol.
- **Ordering:** if the v2 ensemble is also E3's comparator, the 2026 protocol is committed before the H-EP pass, so H-EP results can't shape it.

## 5. Decisions for John (no compute in any of them)

1. Make the 4× control recipe the v2 Atlantic candidate, to be confirmed only on 2026. Recommendation: **yes.**
2. Ask hc-survey to draft that recipe's design: budget, selection rule, seeds, train.py pin, validation-only. It then goes through hc-adversary (2 rounds) and to you.
3. For E3's design, when it is drafted: the same fixed budget, plus an Atlantic-only comparator at that budget.
4. Start a 2026 clean-test protocol draft before any 2026 build: list rule, timing, SESOI and power. It is a draft only, and the I4 rule is unchanged.
