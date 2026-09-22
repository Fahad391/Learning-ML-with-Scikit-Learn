This is based on what I understood

Step-8  -> New Prediction

## Purpose

This step takes the model trained in the previous steps and uses it to predict prices for **new laptop data** that the model has never seen before. It's the "real world usage" stage: everything before this was about building the model, this is about *using* it.

## What the code does, in order

### 1. Import the trained pipeline
```python
import Model_Training
```
Instead of retraining anything, this script **reuses** the exact objects created during training:
- `model` → the trained regression/prediction model
- `num_imputer` → fills missing numeric values (RAM, Storage) the same way training data was filled
- `cat_imputer` → fills missing categorical values (Brand) the same way
- `encoder` → converts Brand text into numbers, using the *same encoding rules* learned during training
- `scale` → scales numeric values using the *same mean/range* learned during training

**Why this matters:** If re-fit these tools on new data instead of reusing the trained ones, the numbers would be transformed differently than they were during training, and the model's predictions would be meaningless. Consistency between training and prediction is the whole point.

### 2. Load new data
```python
new_data = pd.read_csv("new_data.csv")
```
This is the unseen data — laptops with unknown prices that you want the model to predict for.

### 3. Clean the new data (impute)
```python
new_num = num_imputer.transform(new_data[["RAM_GB", "Storage_GB"]])
new_cat = cat_imputer.transform(new_data[["Brand"]])
```
Any missing values in RAM, Storage, or Brand are filled in — using the rule already learned from training data (e.g., "fill missing RAM with the training set's median RAM").

Note: `.transform()` is used here, **not** `.fit_transform()`. This is intentional — fitting again would break consistency with the trained model.

### 4. Encode and scale
```python
new_encoded = encoder.transform(new_cat)
new_scaled = scale.transform(new_num)
```
- Brand (text, e.g., "Dell", "HP") is converted into numeric encoded form.
- RAM and Storage are scaled to the same numeric range the model was trained on.

### 5. Assemble the final input
```python
new_clean_data = np.hstack([new_encoded, new_scaled])
```
The encoded brand columns and the scaled numeric columns are stacked side-by-side into one clean array — matching the exact column structure the model expects as input.

### 6. Predict
```python
Predict_prices = model.predict(new_clean_data)
```
The trained model takes the cleaned array and outputs a predicted price for each row (laptop).

### 7. Display results
```python
for i in range(len(Predict_prices)):
    ...
    print(f"Laptop {i+1} [{brand}, {ram}GB RAM, {storage}GB Storage] -> Predicted Price: {Predict_prices[i]:.2f} BDT")
```
For each laptop in the new data, the script prints its original specs alongside the model's predicted price, formatted to 2 decimal places in BDT (Bangladeshi Taka).

## Key logic to remember

| Concept | Why it matters |
|---|---|
| Reuse `Model_Training` objects | Guarantees new data is processed exactly like training data |
| `.transform()` not `.fit_transform()` | Prevents the model from "learning" new rules from unseen data |
| Same column order (`hstack`) | The model expects features in the exact same order as during training |
| `new_data.csv` | Must have the same columns (`Brand`, `RAM_GB`, `Storage_GB`) as the training data, minus the target (Price) |


### Q) why had to impute, encode & scale, assemble and then predict new data when it's already done in training the model?
Because imputing, encoding, and scaling are not part of the model itself, they're separate preprocessing tools that the model depends on. Training the model does not "bake in" the ability to clean raw data. It only teaches the model to find patterns in already-cleaned numeric arrays.

So every single time handing the model a new data, that data has to go through the same cleaning pipeline — because the model has no idea what "Dell" or a missing RAM value means. It only understands numbers, in a specific shape, in a specific range.

(Q) Why can't just skip straight to model.predict()?

Think about what model.predict() actually needs: a 2D array of numbers, structured exactly like the array it was trained on.

the new_data.csv isn't that. It has:

Missing values (needs imputing)
Text like "Brand" (needs encoding into numbers)
Numeric columns on raw scales like RAM=16, Storage=512 (needs scaling)

If fed new_data straight into model.predict(new_data), it would either crash (wrong shape/data types) or silently give garbage predictions.

(Q)Why you reuse the same imputer/encoder/scaler instead of making new ones?

This is the real reason it looks "redundant" — already did imputing/encoding/scaling once during training.
Because why create new rules, when the old rules to new data can be applied?.

During training: cat_imputer.fit(...) learns the rule (e.g., "fill missing Brand with 'Unknown'") and encoder.fit(...) learns the mapping (e.g., "Dell → [1,0,0]", "HP → [0,1,0]")
During prediction: cat_imputer.transform(...) and encoder.transform(...) just apply those already-learned rules to the new rows

If fit a new encoder on new_data, it could assign "Dell → [0,1,0]" instead of "[1,0,0]" just because the new data happens to have brands in a different order or a different set of brands. The model would then be looking at numbers that mean something completely different from what it learned and predictions would be meaningless, even though nothing "crashed."

### The mental model
Stage	What happens	What's learned vs. applied
Training	fit_transform() on training data	Rules are learned (imputer medians, encoding map, scaler mean/std) AND applied
Prediction	transform() on new data	The same learned rules are just applied, nothing new is learned

So the pipeline isn't being redone — it's the same pipeline object, reused, just running on different input. That's exactly why the script imports Model_Training instead of writing new imputers/encoders/scalers from scratch.
## Summary

This step is essentially: **load new laptops → clean/encode/scale them the same way training data was handled → feed them to the trained model → print predicted prices.** It proves the model can generalize beyond the data it was trained on.
