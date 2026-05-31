# Why Are LSTM and Transformer Underpredicting Large Migration Jumps?

## Observation

Both the LSTM and Transformer tend to predict small or moderate daily movement distances, while the real trajectories contain many large migration jumps.

This behavior is likely caused more by the dataset, loss function, and prediction task than by the model architecture itself.

---

# 1. Movement Distance Imbalance

Most samples in the dataset correspond to:

- Stationary days
- Stopover days
- Small daily movements

Large migration jumps are relatively rare.

When trained with standard regression losses (MSE, MAE, Huber), the model minimizes average error by predicting the most common behavior.

Mathematically:

\[
\hat{y} \approx E[y|x]
\]

If similar historical contexts sometimes lead to:

- 0 km movement
- 150 km movement

the optimal regression prediction may become:

- 40–60 km movement

which is neither outcome but minimizes average loss.

### Consequence

The model becomes conservative and tends to predict moderate movement.

---

# 2. Missing Weather Information

The current experiment intentionally excludes weather features.

However, large migration jumps are often triggered by:

- Tailwinds
- Headwinds
- Pressure changes
- Precipitation
- Other environmental conditions

Without these variables, two samples may appear nearly identical:

```text
same bird
same region
same season
same recent movement history
same stopover duration
```

yet one day results in:

```text
0 km movement
```

while another results in:

```text
200 km migration jump
```

From the model's perspective these outcomes are impossible to distinguish.

### Consequence

The safest prediction becomes moderate movement.

---

# 3. Current Metrics Favor Typical Days

The best models are selected primarily using overall GPS error.

This rewards performance on the majority of samples:

- stationary days
- short movements

rather than rare migration events.

As observed in the experiment:

- Triline models achieve the best overall mean GPS error.
- Direct regression models achieve better performance on large migration movements.

This suggests the current optimization objective may prioritize average behavior rather than migration behavior.

### Consequence

The model learns what is most common, not necessarily what is most biologically interesting.

---

# 4. Distance Compression in Triline Models

The triline architecture predicts:

```text
fly probability
log1p(distance_km)
heading
```

Using:

\[
\log(1+d)
\]

compresses large distances.

For example:

| Distance | log1p(distance) |
|-----------|----------------|
| 10 km | 2.40 |
| 50 km | 3.93 |
| 150 km | 5.02 |
| 300 km | 5.71 |

Large errors in migration distance produce relatively small differences in log-space.

### Consequence

The model may learn:

- stationary vs moving
- small vs medium movement

better than:

- medium vs very large migration jumps

---

# 5. One-Step Training Does Not Encourage Long-Term Migration Behavior

The model is trained to predict only the next day.

Training objective:

```text
Day t-k ... Day t
      ↓
Predict Day t+1
```

The model is not explicitly rewarded for reproducing:

- complete migration routes
- long-range movement patterns
- migration timing decisions

A model can achieve low one-step error while still failing to reproduce realistic long-distance migrations during rollout.

### Consequence

Rollout trajectories often drift and underpredict major jumps.

---

# Recommended Improvements

## 1. Distance-Weighted Loss (Highest Priority)

Give larger migration days more importance during training.

Example:

| True Distance | Weight |
|---------------|---------|
| 0–5 km | 1 |
| 5–30 km | 2 |
| 30–100 km | 4 |
| >100 km | 8 |

Example:

```python
loss = weight(distance_true) * HuberLoss(...)
```

### Expected Benefit

- Stronger gradients from migration days.
- Less tendency toward conservative predictions.

---

## 2. Predict Movement Categories First

Instead of directly predicting distance:

```text
stationary (0–5 km)
short (5–30 km)
medium (30–100 km)
large (>100 km)
```

Then predict distance within the selected category.

Architecture:

```text
Shared Encoder
 ├─ Movement Classifier
 ├─ Distance Regressor
 └─ Heading Regressor
```

### Expected Benefit

Helps the model distinguish migration states before estimating distance.

---

## 3. Oversample Large-Jump Examples

Increase the frequency of large-movement training samples.

Example:

```python
weight = 1
if distance > 30:
    weight += 4
if distance > 100:
    weight += 8
```

or use weighted sampling.

### Expected Benefit

Prevents large migration days from being overwhelmed by stationary days.

---

## 4. Add Weather Features

Most promising long-term improvement.

Potential features:

```text
tailwind support
headwind support
crosswind
wind speed
pressure change
precipitation
temperature
```

### Expected Benefit

Allows the model to learn:

```text
When does migration happen?
```

rather than only:

```text
Where has the bird been recently?
```

---

## 5. Optimize for Migration Metrics

In addition to mean GPS error, track:

```text
migration50_error_mean_km
migration100_error_mean_km
large_jump_recall
distance MAE for >50 km days
distance MAE for >100 km days
```

### Expected Benefit

Model selection becomes aligned with migration prediction quality.

---

# Overall Interpretation

The models have learned a statistically reasonable strategy:

> "Most days birds do not move very far."

This minimizes average prediction error but leads to underprediction of large migration jumps.

The most likely causes are:

1. Strong class imbalance toward stationary days.
2. Missing weather information.
3. Loss functions that favor average behavior.
4. Compression of large distances through log-distance prediction.
5. One-step training objectives that do not emphasize long-range migration behavior.

The most promising next experiments are:

1. Distance-weighted loss.
2. Large-jump oversampling.
3. Distance-bin classification + regression.
4. Adding weather features.
5. Migration-focused evaluation metrics.