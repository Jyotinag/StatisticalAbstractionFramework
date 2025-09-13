# Statistical Abstraction Framework for Ordinal Cybersickness Prediction

# Methodology

This section discusses the **data preprocessing**, the **statistical abstraction framework**, and explains the **model architecture and training setup**.

---

## Data Preprocessing

We downsampled both the **VRWalking** (originally 60Hz) and the **VR.net** (originally 30Hz) datasets to **1 Hz**.  
This was done for two reasons:  
1. To reduce data complexity and processing overhead.  
2. To avoid resampling issues when labeling sensor data with **FMS values**.  

For example, if we retained the original 60Hz sampling, a 60-second interval would contain **3600 rows** with the same FMS score, leading to redundancy.

---

### Outlier Removal

We employed three techniques to remove extreme FMS values:

1. **Range check** → Removed values outside **1–10**.  
2. **Isolation Forest** → For multivariate outlier detection.  
   - Removed unusual sensor combinations (e.g., Tobii eye trackers recording `-1` when eyes are closed).  
   - Max removal capped at **10%** of data.  
3. **IQR Method** → Removed statistically extreme FMS scores.  

- **VR.net:** 18.7% removed  
- **VRWalking:** 13.5% removed  

After removal:  
- Max FMS = **6**  
- Min FMS = **1**

---

### Creating Sequences

- Labeled sequences using **60-second overlapping windows** with **50% overlap**.  
- This captures transitions where FMS evolves within a time window.  

**After preprocessing:**  
- **129,982 rows → 3,789 sequences**  
- Each sequence labeled with its corresponding FMS score.  

---

### Data Oversampling

We used a **hybrid SMOTE-Tomek** method:

- **SMOTE** → Generates synthetic samples with random factors.  
- **Tomek Link Cleaning** → Removes borderline cases that confuse the model.  

This avoids the problem of models always predicting **low FMS scores** due to imbalance.

**Final class distribution:**

| FMS Score | Original Count | Resampled Count | Change |
|-----------|----------------|-----------------|--------|
| 1         | 2,097          | 1,716           | -381   |
| 2         | 1,023          | 1,803           | +780   |
| 3         | 570            | 1,884           | +1,314 |
| 4         | 59             | 1,762           | +1,703 |
| 5         | 19             | 1,584           | +1,565 |
| 6         | 21             | 1,793           | +1,772 |
| **Total** | **3,789**      | **10,545**      | **+6,753** |

---

## Statistical Feature Abstraction

We treat each sequence as an **n × m matrix**:  
- **n** = number of features  
- **m** = window size (60 frames)

We extract four types of statistical features:

### 1. Basic Statistical Features
- Mean, median, std, range, min, max, 25th percentile, 75th percentile, skewness, kurtosis  
- **10 features per sequence**

### 2. Contextual Features
- Based on prior work ([Kundu et al.](#), [Setu et al.](#))  
- Features grouped into 5 categories via keyword matching:
  - Time-series  
  - Video frame  
  - Eye-tracking  
  - Spatial  
  - Movement  

- Extracted **4 descriptive stats per group → 20 features**

### 3. Temporal Features
- Fit linear regression on each column to extract slopes (trend).  
- Calculate:
  - Velocity (1st derivative)  
  - Acceleration (2nd derivative)  
- Extract 9 temporal features.

### 4. Frequency Features
- **FFT** applied on:
  - Head position  
  - Eye position  
  - Gaze direction  

- Extracted:
  - Mean spectral power  
  - Dominant frequency index  

- Total **6 frequency features**

**Final feature vector size = 45 (10 + 20 + 9 + 6)**

---

## Model Architecture

We design a **dual-head ordinal regression model**.

### Ordinal Regression Model

- Input = **45 feature vector**  
- Standardized per batch  
- **4 fully connected hidden layers** with:
  - Batch Normalization  
  - ReLU activation  
  - Dropout  
- Multi-Head Self-Attention for feature relationships  

**Parameters: ~216k**

---

### Dual Prediction Heads

#### Ordinal Head
- Outputs severity score **s** in an abstract space  
- Uses **learnable thresholds** to partition into FMS classes  
- Probabilities calculated as:
# Equations

This section lists all mathematical formulations used in the methodology.

---

## Projection to a Single Score
\[
s = \mathbf{w}_{\text{attn}}^T \cdot \mathbf{h}_{\text{attn}} + b
\]

---

## Cumulative Probability for Ordinal Regression
\[
P(Y \leq k \mid \mathbf{x}) = \sigma(\tilde{\theta}_k - s) = \frac{1}{1 + \exp(-(\tilde{\theta}_k - s))}
\]
for \(k = 1, 2, \ldots, K-1\).

---

## Individual Class Probability
\[
P(Y = k \mid \mathbf{x}) =
\begin{cases}
P(Y \leq 1 \mid \mathbf{x}) & \text{if } k = 1 \\[6pt]
P(Y \leq k \mid \mathbf{x}) - P(Y \leq k-1 \mid \mathbf{x}) & \text{if } 2 \leq k \leq K-1 \\[6pt]
1 - P(Y \leq K-1 \mid \mathbf{x}) & \text{if } k = K
\end{cases}
\]

---

## Expected Ordinal Value
\[
\hat{y}_{\text{ord}} = \mathbb{E}[Y \mid \mathbf{x}] = \sum_{k=1}^{K} k \cdot P(Y = k \mid \mathbf{x})
\]

---

## Regression Loss
\[
\mathcal{L}_{\text{reg}} = \text{SmoothL1}(\hat{y}_{\text{reg}}, y_{\text{true}})
\]

---

## Smooth L1 Loss
\[
\text{SmoothL1}(x) =
\begin{cases}
0.5x^2 & \text{if } |x| < 1 \quad \text{(quadratic for small errors)} \\[6pt]
|x| - 0.5 & \text{if } |x| \geq 1 \quad \text{(linear for large errors)}
\end{cases}
\]

---

## Final Prediction
\[
\text{Final Score} = W_{\text{ord}} \times \hat{y}_{\text{ord}} + W_{\text{reg}} \times \hat{y}_{\text{reg}}
\]

---

