This document serves as a comprehensive reference explaining the logic, data flow, and underlying reasons behind every single line of code in Model_Training.py.

## 1. The Big Picture (Mapping Diagram)

```
┌───────────────────────┐
│ 1. Data Ingestion      │
└───────────┬────────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Feature-Target Split│
└───────────┬────────────┘
            │
            ▼
┌───────────────────────┐
│ 3. Train-Test Split    │
└───────────┬────────────┘
            │
            ▼
┌───────────────────────┐
│ 4. Impute              │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────┐
│ 5. Feature Transform            │
│                                  │
│   ┌───────────────┐ ┌─────────┐ │
│   │ (1) Encode     │ │(2) Scale│ │
│   │   Categories   │ │ Numbers │ │
│   └───────┬────────┘ └────┬────┘ │
└───────────┼────────────────┼─────┘
            │                │
            └───────┬────────┘
                     ▼
┌───────────────────────┐
│ 6. Assemble Clean Data │
└───────────┬────────────┘
            │
            ▼
┌───────────────────────┐
│ 7. Train & Evaluate    │
│    the Model           │
└───────────┬────────────┘
            │
            ▼
┌───────────────────────┐
│ 8. New Prediction      │
└───────────────────────┘
```

---

## 2. Step-by-Step Explanation (What / Why / How)

### Step 1 — Data Ingestion
```python
laptop_data = pd.read_csv("laptops.csv")
```
- **What:** Loads the CSV into a pandas DataFrame.
- **Why pandas, not raw NumPy:** CSVs have mixed types (text `Brand`, numbers
  `RAM_GB`) and column names. DataFrames handle that naturally; NumPy arrays
  want one uniform dtype.

### Step 2 — Feature-Target Split
```python
Feature = laptop_data[["Brand", "RAM_GB", "Storage_GB"]]  # 2D
Target  = laptop_data["Price_BDT"]                         # 1D
```
- **What:** Separates inputs (`Feature`, what the model uses to predict) from
  the output (`Target`, what the model tries to predict).
- **Why double brackets `[[...]]` for Feature but single `[...]` for Target:**
  Double brackets return a DataFrame (2D table — needed because there are
  multiple feature columns). Single brackets return a Series (1D column —
  correct shape for a single target, since scikit-learn's `.fit(X, y)` expects
  `y` as 1D).

### Step 3 — Train-Test Split
```python
Feature_Train, Feature_Test, Target_Train, Target_Test = train_test_split(
    Feature, Target, test_size=0.2
)
```
- **What:** Randomly reserves 20% of rows as a Test set never touched during
  training.
- **Why:** Without a held-out set, you can't tell whether the model actually
  generalizes or just memorized the training rows. Test-set performance is
  the honest estimate of real-world accuracy.

### Step 4 — Imputation (Filling Missing Values)
```python
number_imputer   = SimpleImputer(strategy="mean")
category_imputer = SimpleImputer(strategy="most_frequent")

train_num      = number_imputer.fit_transform(Feature_Train[["RAM_GB", "Storage_GB"]])
train_category = category_imputer.fit_transform(Feature_Train[["Brand"]])

test_num      = number_imputer.transform(Feature_Test[["RAM_GB", "Storage_GB"]])
test_category = category_imputer.transform(Feature_Test[["Brand"]])
```
- **What:** Fills missing (`NaN`) cells. Numeric columns get the column
  **mean**; the categorical column (`Brand`) gets the **mode** (most frequent
  value) — a mean makes no sense for text categories.
- **Why `fit_transform()` on Train but only `transform()` on Test:**
  - `fit_transform` = *learn the statistic (mean/mode) from this data* AND
    *apply it*.
  - `transform` = *apply a statistic already learned* — no new learning.
  - The imputer must learn its fill-values (e.g., "average RAM is 12.4 GB")
    **only from Train**. If you called `fit_transform` on Test too, the Test
    set's own values would leak into the numbers used to "fix" it — an
    optimistic, unrealistic evaluation. In production, you'll never have a
    "test set average" for genuinely new data anyway — only whatever the
    model learned from training.

### Step 5 — Feature Transform

#### 5(1) Encode Categories
```python
encode = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encode_category_train = encode.fit_transform(train_category)
encode_category_test  = encode.transform(test_category)
```
- **What:** Converts the text column `Brand` (e.g., "Asus", "Dell", "HP")
  into multiple binary (0/1) columns — one per brand.
- **Why encode at all:** `LinearRegression` is pure math — it can't multiply
  weights by the string `"Asus"`. Every input must be numeric.
- **Why One-Hot instead of just numbering brands 1, 2, 3…:** Numbering would
  imply a false order/magnitude (e.g., "Dell=3 is more than Asus=1"), which
  the model would wrongly treat as meaningful. One-hot columns are
  independent 0/1 flags — no fake ranking.
- **Why `sparse_output=False`:** Returns a normal dense NumPy array instead of
  a sparse matrix, so it stacks cleanly with the scaled numeric array in Step
  6.
- **Why `handle_unknown="ignore"`:** If Test contains a brand never seen in
  Train, this tells the encoder to output all-zeros for that row instead of
  crashing.

#### 5(2) Scale Numbers
```python
Scaler = StandardScaler()
Scale_numericals_train = Scaler.fit_transform(train_num)
Scale_numericals_test  = Scaler.transform(test_num)
```
- **What:** Rescales `RAM_GB` and `Storage_GB` so each has mean 0 and
  standard deviation 1 (a "z-score").
- **Why:** `Storage_GB` (e.g., 512) is numerically much larger than `RAM_GB`
  (e.g., 16). Without scaling, the larger-magnitude feature can dominate the
  model's optimization purely because of its scale — not because it's more
  important. Scaling puts both features on equal footing.
- **Why `fit_transform` (Train) / `transform` (Test) again:** Same leakage
  reason as imputation — the scaler's mean/std must come only from Train.

### Step 6 — Assemble Clean Data
```python
clean_train_data = np.hstack([encode_category_train, Scale_numericals_train])
clean_test_data  = np.hstack([encode_category_test,  Scale_numericals_test])
```
- **What:** Horizontally stacks (glues side-by-side) the one-hot encoded
  brand columns and the scaled numeric columns into one final matrix per
  split.
- **Why `np.hstack` instead of `pd.DataFrame`:**
  - By this point, both `encode_category_*` and `Scale_numericals_*` are
    **already plain NumPy arrays** (that's what `.fit_transform`/`.transform`
    return) — there's no pandas object left to combine.
  - `np.hstack` is the direct, lightweight way to glue NumPy arrays
    column-wise. Wrapping them into a `pd.DataFrame` first would add column
    names and indexing overhead for zero benefit — `LinearRegression.fit()`
    only needs a plain 2D numeric array, it doesn't care about column labels.
  - Using a DataFrame here would actually be *extra* work: you'd have to
    invent column names for the one-hot output, and re-align indices between
    two pieces that came from different transformers.
  - In short: **stay in NumPy because you're already in NumPy** — pandas
    would be a detour with no payoff at the model-input stage.

### Step 7 — Train & Evaluate the Model
```python
Model = LinearRegression()
Model.fit(clean_train_data, Target_Train)

predict_price = Model.predict(clean_test_data)

error_measure = np.sqrt(mean_squared_error(Target_Test, predict_price))
```
- **What:** Fits a linear regression (finds the best-fit weights) on the
  clean training matrix, then predicts prices for the unseen test matrix.
  `error_measure` is the **RMSE** (Root Mean Squared Error) — how far off
  predictions are, on average, in the same units as price (BDT).
- **Why RMSE specifically:** Squaring the errors first (inside
  `mean_squared_error`) penalizes big misses more than small ones; taking the
  square root afterward brings the units back to BDT (raw MSE would be in
  "BDT²", which isn't interpretable).

### Seeig the Result
```python
actual_prices = Target_Test.values

for i in range(len(predict_price)):
    print(f"Sample {i+1} predicted {predict_price[i]:.2f} BDT | "
          f"Actual price is {actual_prices[i]:.2f} BDT")
```
- **What:** Prints predicted vs. actual price side-by-side for each test
  sample.
- **Why `Target_Test.values`:** `Target_Test` is a pandas Series, and after
  `train_test_split` it keeps its **original, shuffled row indices** (e.g.,
  row 47, 12, 88…) instead of clean 0,1,2,3…. `predict_price`, on the other
  hand, is a plain NumPy array that scikit-learn always returns starting at
  position 0. `.values` strips the Series down to a plain NumPy array with a
  fresh 0-based position, so `predict_price[i]` and `actual_prices[i]` line
  up on the *same* sample when looped together — matching by **position**,
  not by the old pandas index label.

---

## 3. Your Questions, Answered

### Q1) Why used `np.hstack` instead of `pd.DataFrame`?
Because by Step 6, everything is already a plain NumPy array (the output of
`OneHotEncoder`/`StandardScaler` `.transform()` calls) — there's no pandas
structure left to preserve. `np.hstack` glues NumPy arrays column-wise
directly and cheaply. Rebuilding a `pd.DataFrame` at this point would mean
manually inventing column names for the one-hot output and would add no
value, since `LinearRegression.fit()` consumes a raw numeric matrix and
ignores column labels entirely. **Use pandas while you need labels or mixed
types; drop to NumPy once everything is already pure numbers headed into a
model.**


### Q2) Why is there no need to decode manually?
Decoding (turning one-hot columns back into a `"Brand"` string, or un-scaling
a number back to raw GB) would only be necessary if you needed to **read or
interpret the transformed feature matrix itself** — e.g., inspecting
`clean_train_data` and wanting to know "which row is Asus?"

But the pipeline never does that. It only ever:
1. Feeds `clean_train_data` / `clean_test_data` into the model (which wants
   encoded/scaled numbers — decoding would break it).
2. Prints `predict_price` (the **target**, `Price_BDT`) against
   `actual_prices` (also the raw, original **target** values from
   `Target_Test`, never encoded or scaled in the first place).

Only the **features** (`Brand`, `RAM_GB`, `Storage_GB`) went through
encoding/scaling. The **target** (`Price_BDT`) was never transformed at
all — `Target_Train`/`Target_Test` stayed as plain, original prices the
whole time. So the final printout is comparing two things that were always
in "human-readable" form to begin with — there's simply nothing encoded
here to decode.

### Q3) why fit_transform here but transform there?
Test data must remain a stand-in for
*unseen, real-world* data, so nothing about it may influence how you clean or
transform it. It should only ever be *processed with recipes made from
Train*

---

## 4. Quick-Reference Table

| Concept | Object Used | Fit on Train? | Transform on Test? | Why |
|---|---|---|---|---|
| Fill missing numbers | `SimpleImputer(strategy="mean")` | ✅ `fit_transform` | ✅ `transform` only | Learn mean from Train only — avoid leakage |
| Fill missing category | `SimpleImputer(strategy="most_frequent")` | ✅ `fit_transform` | ✅ `transform` only | Same leakage reason |
| Encode category | `OneHotEncoder` | ✅ `fit_transform` | ✅ `transform` only | Learn category list from Train only |
| Scale numbers | `StandardScaler` | ✅ `fit_transform` | ✅ `transform` only | Learn mean/std from Train only |
| Combine features | `np.hstack` | — | — | Already NumPy; no pandas needed |
| Predict | `LinearRegression` | Trained on `clean_train_data` | Predicts on `clean_test_data` | Standard supervised learning |
| Measure error | `mean_squared_error` + `np.sqrt` | — | — | RMSE = interpretable error in BDT |

---
