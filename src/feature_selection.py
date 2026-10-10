"""
feature_selection.py  (NWAD-2.0)
================================

WHAT THIS FILE DOES
-------------------
This module turns the human decisions written in ``data/FEATURES.md`` into
usable, machine-readable code.

In plain English, it answers two questions:

1. "Which of the 49 columns should the main model actually train on?"
   -> We use the 37 "behavioral" features listed in FEATURES.md.

2. "Are the 'shortcut' columns (ports, protocol, service, connection state...)
   so good at predicting the attack type that the model would just memorise
   how the dataset was built, instead of learning real attack behaviour?"
   -> We build a tiny throw-away model using ONLY those 6 columns and look at
      how accurate it is. If it is suspiciously high, that is a red flag.

WHY THIS FILE EXISTS (the motivation)
-------------------------------------
Dr. Chuprovs' feedback on the proposal warned about two failure modes, and
FEATURES.md was written to address them:

* LEAKAGE  - giving the model information it could not possibly have at test
             time (e.g. the raw source IP, the timestamp, or the answer itself).
             These columns are simply removed.

* SHORTCUT LEARNING - the model finds a cheap, artificial cue (like "this
             destination port always meant this attack in the way this capture
             was generated") and leans on it. It would look accurate in the lab
             but fail on real, different traffic. We do not drop these columns
             blindly - we TEST them with a baseline model and let the numbers
             inform the team's decision.

This module does NOT do the heavier data cleaning (scaling, imputation
strategy, etc.). That is deliberately the job of ``preprocessing.py`` so each
file has one clear responsibility. See the note near the missing-value code
below.

HOW TO RUN IT STANDALONE
------------------------
    python src/feature_selection.py

That will load the dataset, print the feature groups, and run the shortcut
check (a small random-forest comparison). You do not need to do this from a
notebook - but notebook 02 (preprocessing) can also ``import`` the functions
here and reuse them.

TERMINOLOGY CHEAT-SHEET (if you are new to ML)
-----------------------------------------------
* "Feature"     = one input column the model looks at.
* "Target"      = the column the model is trying to predict (the answer).
* "Baseline"    = a simple reference model so we can judge whether a result
                  is good or just lucky.
* "Random Forest" = a popular, easy-to-use algorithm made of many decision
                  trees that vote together. Great default for tabular data and
                  it handles a mix of numeric + categorical columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# 1. THE COLUMN GROUPS  (a direct, coded translation of data/FEATURES.md)
# ---------------------------------------------------------------------------
#
# The whole point of writing these as explicit lists (instead of "keep columns
# 0-36") is that a reader can diff them against FEATURES.md and see exactly
# what the team decided, and why. Keeping them here also means notebooks and
# other scripts all agree on the same feature set.

# Columns we REMOVE because they leak information or are the answer themselves.
# (FEATURES.md section "Excluded")
# NOTE: This list is for DOCUMENTATION only - no code reads it. The exclusion
# happens implicitly by NOT listing these in BEHAVIORAL_COLUMNS / SHORTCUT_CHECK_COLUMNS.
EXCLUDED_COLUMNS = [
    "label",        # binary answer (1=malicious) -> the target, never an input
    "attack_cat",   # 9-way answer (portscan, brute force, ...) -> the target
    "srcip",        # raw source IP -> identifies the host/scenario, not behaviour
    "dstip",        # raw dest IP -> same problem as srcip
    "Stime",        # timestamp -> leaks which capture batch a row came from
    "Ltime",        # timestamp -> same as Stime
]

# Columns that might cause SHORTCUT learning. We keep them in a separate group
# so we can test them in isolation rather than dropping them on a hunch.
# (FEATURES.md section "Flagged for shortcut learning check")
SHORTCUT_CHECK_COLUMNS = [
    "sport",            # source port
    "dsport",           # destination port
    "proto",            # protocol (e.g. 6=TCP, 17=UDP)
    "service",          # named service (e.g. ftp, http)
    "state",            # connection state (e.g. SFIN, REJ)
    "is_sm_ips_ports",  # 1 if src/dst IPs and ports are equal -> tied to the IPs we excluded
]

# The 37 features we treat as genuine "behaviour" and feed to the main model.
# (FEATURES.md section "Kept as behavorial features - 37")
BEHAVIORAL_COLUMNS = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb",                      # TODO(team): still undecided per FEATURES.md - may drop
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

# Two possible "answers" the model can predict. We default to attack_cat
# because FEATURES.md's shortcut test is defined in terms of attack_cat, and
# the project goal is to distinguish the *kind* of attack, not just benign/malicious.
TARGET_BINARY = "label"        # 0/1 -> "is this an attack at all?"
TARGET_MULTICLASS = "attack_cat"  # 9 categories -> "which kind of attack?"


# ---------------------------------------------------------------------------
# 2. SMALL HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _resolve_columns(df, wanted):
    """
    Map our human-written column names onto the *actual* column names in the
    loaded DataFrame, ignoring case and surrounding spaces.

    WHY: FEATURES.md and the raw CSV can disagree on capitalisation / typos
    (for example the raw data has a column literally named 'ct_src_ ltm' with
    a space). Rather than hard-code the exact spelling everywhere, we look each
    name up case-insensitively and return the real name pandas is using.

    It also returns only the columns that actually exist, and warns you about
    any that are missing so you notice a data problem instead of silently
    training on fewer features than you think.
    """
    # Build a fast lookup: lowercased-and-stripped name -> real name in df
    lookup = {str(c).strip().lower(): c for c in df.columns}

    found, missing = [], []
    for name in wanted:
        real = lookup.get(name.strip().lower())
        if real is None:
            missing.append(name)
        else:
            found.append(real)

    if missing:
        # A clear, actionable warning instead of a cryptic KeyError later.
        print(f"[feature_selection] WARNING: these requested columns were not "
              f"found in the data and will be skipped: {missing}")

    return found


def _resolve_target(df, target):
    """
    Resolve a single target column name (e.g. "label") to the exact name the
    DataFrame is actually using (e.g. "Label"), ignoring case and spaces.

    WHY this is separate from _resolve_columns:
      _resolve_columns *silently skips* names it cannot find and warns - that
      is fine for a *feature* (dropping one of 37 features just makes the model
      weaker). But the *target* is non-negotiable: if it is missing, the whole
      training step is impossible, so we raise a loud, actionable error instead
      of quietly returning nothing.

    Pandas column look-up is case-sensitive, and the raw dataset capitalises
    inconsistently (e.g. the feature list has "Label" but we refer to it as
    "label"), so this normalisation is what makes the code robust to that.
    """
    lookup = {str(c).strip().lower(): c for c in df.columns}
    real = lookup.get(str(target).strip().lower())
    if real is None:
        raise KeyError(
            f"Target column '{target}' not found in the data. "
            f"Available columns: {list(df.columns)}")
    return real


def _encode_features(frame):
    """
    Turn a DataFrame of features into a pure-number 2-D array that
    sklearn's Random Forest understands.

    WARNING: The fillna(0) at the end is a diagnostic placeholder. If this
    function is reused in the production model (via select_features), the
    missing-value policy is NOT yet final - that is a preprocessing.py decision.

    WHY: decision trees need numbers. Most of our columns already are numbers,
    but a few are text (e.g. 'state' = 'SFIN', 'service' = 'http',
    'ct_flw_http_mthd' = 'POST'). OrdinalEncoder assigns each distinct text
    value a small integer (SFIN->0, REJ->1, ...). The *order* of the numbers
    does not matter for a tree - it only needs the values to be *distinct* -
    so ordinal encoding is exactly the right (and simplest) choice here.

    We fill any remaining blanks with 0. NOTE ON MISSING VALUES:
    FEATURES.md says the real imputation strategy is a preprocessing.py
    decision and is "not yet implemented". So this fill is ONLY enough for the
    quick diagnostic in this module to run; do not treat it as the project's
    final missing-value policy.
    """
    out = frame.copy()

    # Split columns into two kinds, based on whether their *values* look
    # numeric rather than on the declared dtype (which pandas reports
    # inconsistently - e.g. 'str' vs 'string' - and which lies for columns
    # like sport/dsport/proto that hold numbers or words inside a string type):
    #
    #   numeric-as-string  -> e.g. sport = '5432'. A real quantity; we must
    #                         convert to a float so 5432 is treated as 5432.
    #   true categorical   -> e.g. proto = 'udp'. A label; we ordinal-encode.
    #
    # A column counts as "numeric" if EVERY non-missing value parses as a
    # number. This is exactly what separates '5432' (numeric) from 'udp'
    # (categorical).
    numeric_cols, category_cols = [], []
    for c in out.columns:
        converted = pd.to_numeric(out[c], errors="coerce")
        # If the conversion created new blanks that weren't already blank, the
        # column contains non-numeric text -> it is categorical.
        newly_blank = converted.isna() & out[c].notna()
        if newly_blank.any():
            category_cols.append(c)
        else:
            numeric_cols.append(c)

    # Cast the numeric ones straight to float (keeps their true magnitude).
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Ordinal-encode the categorical ones (distinct values -> distinct ints;
    # the particular integers do not matter to a decision tree).
    if category_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        out[category_cols] = enc.fit_transform(out[category_cols].astype(str))

    # Everything is numeric now. fillna(0) makes any remaining blanks harmless
    # for the tree (diagnostic-only; see note above about preprocessing.py).
    return out.astype(float).fillna(0).to_numpy()


def _sample_rows(df, target, max_rows=None, seed=42):
    """
    Optionally shrink the data to at most `max_rows` random rows.

    WHY: the full UNSW-NB15 set is millions of rows and Random Forest is
    slow on large data. Passing a `max_rows` limit (e.g. 120_000) gives a
    fast, representative estimate of accuracy in seconds instead of minutes.
    We set a random seed (42) so the sample is the same every run, which means
    you and a teammate will see identical numbers (reproducibility).

    Set max_rows=None to use ALL rows (slower, but the full-data estimate).

    Rows whose target is blank are dropped because they cannot be used for
    supervised training (there is no answer to learn from).
    """
    work = df[df[target].notna()].copy()
    if max_rows is not None and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=seed)
    return work


def _class_support(y, labels):
    """
    Return a dict mapping each class label -> how many times it appears in `y`.

    Used to identify the majority class so we can compute a "rare-class" F1
    that excludes it.
    """
    from collections import Counter
    counts = Counter(list(y))
    return {lbl: int(counts.get(lbl, 0)) for lbl in labels}


def _score_for(columns, df, target, max_rows, class_weight=None):
    """
    Train one small Random Forest on the given list of columns and return a
    dict of its held-out scores. This is the workhorse behind the shortcut
    check: we call it with different column groups and compare.

    WHY F1 in addition to accuracy:
      The dataset is heavily class-imbalanced (most rows are "Normal", and
      within attack_cat, "Normal" dominates). A model that simply predicts
      "Normal" for everything already scores high on ACCURACY while catching
      almost no actual attacks. F1 does not have that blind spot:
        * macro F1  - treats every class as equally important (average of the
                      per-class F1). This is the number that best answers
                      "does it actually detect the rare attacks?".
        * weighted F1 - like accuracy, weighs each class by its size. Useful
                      for comparing with the accuracy figure.
        * per-class F1 - shows exactly which attack types are caught and
                      which are missed, so you can see WHERE the behavioral
                      features help or fail.
    """
    real_cols = _resolve_columns(df, columns)
    if len(real_cols) == 0:
        raise ValueError("None of the requested columns exist in the data.")

    real_target = _resolve_target(df, target)     # e.g. "label" -> "Label"
    data = _sample_rows(df, real_target, max_rows=max_rows)
    y = data[real_target]
    X = _encode_features(data[real_cols])

    # train_test_split: we hide ~20% of the data from the model, then ask it to
    # predict those hidden answers. Scores on the *held-out* part are an honest
    # estimate of how well it will do on data it has not seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # n_estimators=100 is deliberately modest: enough to be a fair, stable
    # baseline for a *diagnostic*, without taking minutes to train. For the
    # final research model (train.py) you may want more.
    #
    # class_weight: "balanced" makes the model weigh each training sample in
    # inverse proportion to how common its class is. That directly attacks the
    # imbalance problem - rare attacks (Backdoor, Worms, DoS) get up-weighted
    # so the model is penalised harder for missing them. None = standard,
    # unweighted training.
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42,
                                   class_weight=class_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Per-class F1 (dict: class -> its F1), used both for reporting and for
    # the per-class table printed in run_shortcut_check. f1_score(average=None)
    # returns one number per label (in the order of `labels`), so we zip them.
    labels = list(y.unique())
    per_class = {
        lbl: float(v)
        for lbl, v in zip(labels, f1_score(
            y_test, y_pred, average=None, zero_division=0, labels=labels))
    }

    # "Rare-class" F1: the average F1 over every class EXCEPT the single most
    # common one (e.g. "Normal"). This is the single best lens for "does the
    # model actually catch the attacks, not just the easy majority?" - it
    # strips out the class that inflates plain accuracy.
    support = _class_support(y_test, labels)
    majority = max(per_class, key=lambda c: support[c])
    rare_classes = [c for c in per_class if c != majority]
    f1_rare = float(np.mean([per_class[c] for c in rare_classes])) if rare_classes else 0.0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_rare": f1_rare,
        "per_class_f1": per_class,
        "majority_class": majority,
    }


# ---------------------------------------------------------------------------
# 3. PUBLIC API
# ---------------------------------------------------------------------------

# Expected consumer: src/train.py and notebooks/03_random_forest.ipynb.
# It is NOT called by this file's standalone demo (section 4) - that only
# runs run_shortcut_check.
def select_features(df, target=TARGET_MULTICLASS, include_shortcut=False):
    """
    Return the (X, y) pair for the MAIN model - i.e. the data with the
    leaked/excluded columns removed and only the chosen features kept.

    Parameters
    ----------
    df : the full dataset (e.g. the output of load_data.load_dataset()).
    target : which answer column to predict. "attack_cat" (default, 9 classes)
             or "label" (binary). Pick the one matching the model's goal.
    include_shortcut : if True, also add the 6 shortcut columns to the feature
             set. Default is False: we start with the 37 behavioural features
             only, and let run_shortcut_check() give you the evidence to decide
             whether adding the shortcut columns is worth it.

    Returns
    -------
    (X, y) : X is a 2-D float numpy array of the selected features,
             y is the target numpy array. Ready to hand to any sklearn model.
             NOTE: returns numpy arrays, not a DataFrame.
    """
    if target not in (TARGET_BINARY, TARGET_MULTICLASS):
        raise ValueError(f"target must be '{TARGET_BINARY}' or '{TARGET_MULTICLASS}'.")

    real_target = _resolve_target(df, target)     # e.g. "label" -> "Label"
    wanted = list(BEHAVIORAL_COLUMNS) + (list(SHORTCUT_CHECK_COLUMNS) if include_shortcut else [])
    real_cols = _resolve_columns(df, wanted)

    data = df[df[real_target].notna()].copy()     # drop rows with no answer
    X = _encode_features(data[real_cols])
    y = data[real_target].to_numpy()

    print(f"[feature_selection] Using {X.shape[1]} features, {X.shape[0]} rows, "
          f"target='{target}'.")
    return X, y


def run_shortcut_check(df, target=TARGET_MULTICLASS, max_rows=None):
    """
    The heart of the shortcut-learning test from FEATURES.md, plus a
    class-imbalance check.

    It trains small Random Forests under two settings:

      * UNBALANCED  - standard training (the model sees the raw class
                      distribution, so the majority class dominates).
      * BALANCED    - with class_weight='balanced', so rare attacks are
                      up-weighted. This directly tests whether we can improve
                      detection of the weak attack types (Backdoor, Worms,
                      DoS) that unbalanced training struggles with.

    For each setting it reports, for three feature groups:

      1. shortcut-only   -> ONLY the 6 flagged columns. If its F1 is already
         high, the model can identify the attack type almost entirely from
         dataset-construction cues -> the shortcut-learning red flag.
      2. behavioural-only -> ONLY the 37 behavioural features ("honest" model).
      3. combined        -> all 43 columns.

    METRICS and WHY:
      * accuracy   - inflated by the majority class; good for context.
      * F1 (macro) - treats every class as equally important.
      * F1 (rare)  - average F1 over every class EXCEPT the majority one.
                      This is the single most honest number for "does it catch
                      the attacks?". We report this as the headline.
      * per-class  - shows exactly which attack types are caught or missed.

    Reading:
      * High shortcut-only F1 (rare) = strong shortcut-learning risk.
      * BALANCED vs UNBALANCED gap on the rare-attack F1 = how much the
        imbalance is hurting detection, and whether class-weighting fixes it.

    max_rows: cap on rows used (None = use ALL rows). Slower on the full data
      but the most representative estimate.

    NOTE: The balanced-vs-unbalanced comparison is hard-coded inside this
      function. It is not a user-facing parameter - both settings are always
      reported side by side.

    Returns a dict of the score-sets (both settings) so notebooks can store
      or plot them.
    """
    # Work on a (possibly sampled) subset once, so every model below trains on
    # the same rows and the numbers are directly comparable.
    real_target = _resolve_target(df, target)
    pool = _sample_rows(df, real_target, max_rows=max_rows)
    total_rows = len(pool)
    rows_desc = f"{total_rows:,} rows (all)" if max_rows is None else f"{total_rows:,} rows (sampled)"

    print(f"\n=== SHORTCUT-LEARNING CHECK (target: {target}) ===")
    print(f"Rows used: {rows_desc}")

    feature_groups = [
        ("shortcut-only     ", SHORTCUT_CHECK_COLUMNS),
        ("behavioural-only  ", BEHAVIORAL_COLUMNS),
        ("combined          ", BEHAVIORAL_COLUMNS + SHORTCUT_CHECK_COLUMNS),
    ]

    def _run_setting(weight):
        setting = {}
        for label, cols in feature_groups:
            setting[label.strip()] = _score_for(
                cols, df, target, max_rows=max_rows, class_weight=weight)
        return setting

    unbalanced = _run_setting(None)
    balanced = _run_setting("balanced")

    def _print_setting(name, setting):
        print(f"\n--- {name} ---")
        for label, cols in feature_groups:
            s = setting[label.strip()]
            print(f"  {label}")
            print(f"      accuracy  = {s['accuracy']:.4f}")
            print(f"      F1 macro  = {s['f1_macro']:.4f}")
            print(f"      F1 rare   = {s['f1_rare']:.4f}   <- excludes '{s['majority_class']}'")

    _print_setting("UNBALANCED (standard training)", unbalanced)
    _print_setting("BALANCED (class_weight='balanced')", balanced)

    # Per-class F1 for the behavioural-only model, unbalanced vs balanced, so
    # you can see which attack types the balancing actually improves.
    print("\n--- Per-class F1 for behavioural-only (unbalanced -> balanced) ---")
    ub = unbalanced["behavioural-only"]["per_class_f1"]
    bal = balanced["behavioural-only"]["per_class_f1"]
    for cls in sorted(ub, key=lambda k: ub[k]):
        print(f"      {str(cls):20} {ub[cls]:.4f} -> {bal[cls]:.4f}")

    # Headline takeaways, computed rather than hand-written.
    ub_rare = unbalanced["behavioural-only"]["f1_rare"]
    bal_rare = balanced["behavioural-only"]["f1_rare"]
    ub_short = unbalanced["shortcut-only"]["f1_rare"]
    print(f"\n  Shortcut-only rare-F1 (unbalanced)        = {ub_short:.4f}")
    print(f"  Balancing lifts behavioural rare-F1 by    = {bal_rare - ub_rare:+.4f}")
    print("  (A low shortcut-only rare-F1 = weak shortcut risk; a large")
    print("   positive balancing lift = imbalance is the main enemy to fix.)")

    return {"unbalanced": unbalanced, "balanced": balanced}


# ---------------------------------------------------------------------------
# 4. STANDALONE DEMO  (runs only when you execute this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # We import load_data here (not at the top) so that `import feature_selection`
    # works even on a machine without the dataset present. When you run this
    # file as a script from the project root, `src/` is on the search path, so
    # `import load_data` resolves to src/load_data.py.
    from load_data import load_dataset

    print("Loading dataset... (this reads the 4 raw UNSW-NB15 files)")
    data = load_dataset()

    print("\nColumn groups from FEATURES.md:")
    print(f"  excluded ({len(EXCLUDED_COLUMNS)}):       {EXCLUDED_COLUMNS}")
    print(f"  shortcut check ({len(SHORTCUT_CHECK_COLUMNS)}): {SHORTCUT_CHECK_COLUMNS}")
    print(f"  behavioural ({len(BEHAVIORAL_COLUMNS)}):  {len(BEHAVIORAL_COLUMNS)} features")

    # Run the diagnostic on ALL rows (max_rows=None). This uses the full
    # dataset, so it is slower than a 120k-row sample but is the most
    # representative estimate. Pass e.g. max_rows=120_000 for a quick version.
    run_shortcut_check(data, target=TARGET_MULTICLASS, max_rows=None)
