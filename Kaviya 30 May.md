## Problem – Defective Items in a Factory

A factory has recorded the number of defective items produced per day over **1000 production days**. The number of defective items per day is randomly generated between **0 and 20** to simulate real-world variability in quality.

Using this dataset, calculate the probability that **exactly 5 defective items** will be produced on a new day. Use Python to:

- Generate the data
- Calculate the mean and standard deviation
- Compute the probability using the normal distribution (with continuity correction)

---

### Step-by-step Solution in Python

````python
from numpy.random import randint as ri
import pandas as pd

#Generate random data for 1000 days (defective items between 0 and 20)
defects = ri(0, 21, 1000)
defects = pd.Series(defects)




```python
#Calculate mean and Standard Deviation
mean=defects.mean()
std_dev=defects.std()
print(f"Mean:{mean}")
print(f"Standard Deviation:{std_dev}")

````

```python
#Probability to calculate exactly 5 defects
from scipy.stats import norm
p_4_5=norm.cdf(5.5,loc=mean,scale=std_dev)
p_3_5=norm.cdf(4.5,loc=mean,scale=std_dev)
probability=p_4_5-p_3_5
print(f"Probability of exactly 5 defects:{probability:.4f}")
```

```python

```

## Problem – Testing the Claim About Delivery Time

A food delivery company claims that its average delivery time is **30 minutes**. Based on historical data, the **population standard deviation** is known to be **4 minutes**.

To evaluate this claim, a consumer rights group decides to test the null hypothesis that the average delivery time is **at most 30 minutes**. They observe a sample of **40 deliveries**, and the average delivery time for the sample comes out to be **31.2 minutes**.

### Objective:

Test the null hypothesis using the z-test.

- **Null Hypothesis (H₀): μ ≤ 30** (Average delivery time is 30 minutes or less)
- **Alternative Hypothesis (H₁): μ > 30** (Average delivery time is more than 30 minutes)

---

### Step-by-step Solution in Python

````python
import numpy as np


# Known values
population_mean = 30        # Claimed average delivery time
sample_mean = 31.2          # Observed sample mean
std_dev = 4                 # Known population standard deviation
n = 40                      # Sample size



```python
#Calculate the Z statistic
z=(sample_mean-popu;ation_mean)/(std_dev/np.sqrt(n))
print(f"Z-statistic:{z:.2f}")

````

```python
#Compute the p-value
p_value=1-norm.cdf(z)
print(f"P-VALUE:{p_value.4f}")
```

```python
#Conclusion
alpha=0.05
if p_value < alpha:
   print("Reject the H0:Delivery time is significantly more than 30 minutes.")
else:
   print("Fail to reject the null hypothesis: Not enough evidence to say delivery time is more than 30 minutes.")

```

## Problem – Fitness Program Impact Analysis

A health and wellness company is evaluating the impact of its **6-week fitness training program**. They collect performance data (in terms of fitness scores out of 100) from participants **before and after** the program.

You are provided with a dataset of **150 participants**, with the following information:

- **Initial Score** (before the program)
- **Final Score** (after the program)
- **Gender** of the participant (0 = Female, 1 = Male)

---

### Your Task:

Using the dataset provided below, perform the following statistical tests:

1. **One-Sample t-Test**  
   Test whether the **average initial fitness score** is at least **65**.

2. **Two-Sample Independent t-Test**  
   Compare the **initial fitness scores of male and female participants** to check if there's a significant difference.

3. **Paired Sample t-Test**  
   Test whether the **final scores are significantly higher than the initial scores**, i.e., whether the fitness program had a measurable impact.

---

Hypotheses
1️⃣ One-Sample t-Test:

**Null Hypothesis** H₀: μ ≥ 65 (Average initial score is at least 65)

**Alternate Hypothesis** H₁: μ < 65 (Average initial score is less than 65)

2️⃣ Two-Sample Independent t-Test:

**Null Hypothesis** H₀: μ₁ = μ₂ (No difference in average initial scores between males and females)

**Alternate Hypothesis** H₁: μ₁ ≠ μ₂ (There is a difference in average initial scores)

3️⃣ Paired Sample t-Test:

**Null Hypothesis** H₀: μ_diff = 0 (No change in scores before and after the program)

**Alternate Hypothesis** H₁: μ_diff < 0 (Final scores are higher than initial scores)

### Data Setup – Generate your dataset using the code below:

````python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(100)

# Sample size
n = 150

# Gender (0 = Female, 1 = Male)
gender = np.random.choice([0, 1], size=n)

# Initial scores (mean slightly < 65 to create realistic test)
initial_scores = np.random.normal(loc=64, scale=6, size=n)

# Final scores (showing average improvement)
final_scores = initial_scores + np.random.normal(loc=5, scale=3, size=n)

# Create DataFrame
df = pd.DataFrame({
    'Gender': gender,
    'Initial_Score': initial_scores,
    'Final_Score': final_scores
})

df.head()


# T-Test Instructions

## Statistical Tests with `scipy.stats`

Use the appropriate test based on your data and hypothesis:

### 1. One-Sample T-Test
```python
from scipy.stats import ttest_1samp
# Perform One-Sample t-Test
t_stat, p_value = ttest_1samp(df['Initial_Score'], popmean=65)

print(f"One-Sample t-Test: t-statistic = {t_stat:.3f}, p-value (two-tailed) = {p_value:.4f}")

# For one-tailed test (H1: mean < 65), divide the p-value
if p_value / 2 < 0.05 and t_stat < 0:
    print("Reject H₀: The average initial score is significantly less than 65.")
else:
    print("Fail to reject H₀: No significant evidence that the average initial score is less than 65.")



### 2. Two-Sample Independent T-Test

```python
from scipy.stats import ttest_ind  # For independent t-test
from scipy.stats import ttest_rel  # For paired sample t-test (used later)
# Separate initial scores by gender
initial_male = df[df['Gender'] == 1]['Initial_Score']
initial_female = df[df['Gender'] == 0]['Initial_Score']

# Perform independent t-test
t_stat, p_value = ttest_ind(initial_male, initial_female)

print(f"Two-Sample Independent t-Test: t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: There is a significant difference in initial scores between males and females.")
else:
    print("Fail to reject H₀: No significant difference in initial scores between males and females.")



```python
# Perform paired t-test
t_stat, p_value = ttest_rel(df['Initial_Score'], df['Final_Score'])

print(f"Paired Sample t-Test: t-statistic = {t_stat:.3f}, p-value (two-tailed) = {p_value:.4f}")

# One-tailed test (H1: Final > Initial → mean difference < 0)
if p_value / 2 < 0.05 and t_stat < 0:
    print("Reject H₀: Final scores are significantly higher than initial scores.")
else:
    print("Fail to reject H₀: No significant improvement after the program.")


````

```python

```

```python

```

## Problem – ANOVA Analysis of Customer Satisfaction Across Store Branches

A retail company wants to analyze whether the **average customer satisfaction scores** vary significantly across its three store branches: **Branch A, Branch B, and Branch C**.

You are provided with data containing:

- **Customer_ID**
- **Branch** (Categorical Variable)
- **Satisfaction_Score** (Continuous Variable on a scale from 0 to 500)

---

### Objective:

Use **One-Way ANOVA** to test the following hypotheses:

- **H₀ (Null Hypothesis)**: The average satisfaction scores across all three branches are **equal**.
- **H₁ (Alternative Hypothesis)**: At least one branch has a **different average** satisfaction score.

---

### Dataset Generation (Run this code block)

````python
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Sample size per branch
n = 70

# Create satisfaction scores for three branches
branch_a = np.random.normal(loc=420, scale=30, size=n)
branch_b = np.random.normal(loc=400, scale=35, size=n)
branch_c = np.random.normal(loc=430, scale=25, size=n)

# Combine into a DataFrame
data = pd.DataFrame({
    'Customer_ID': range(1, n*3 + 1),
    'Branch': ['A'] * n + ['B'] * n + ['C'] * n,
    'Satisfaction_Score': np.concatenate([branch_a, branch_b, branch_c])
})

data.head()



```python
#Hypothesis Assumption
H0(Null):Mean satisfaction scores for all branches are equal
H1(Alternative):At least one branch has a different mean satisfaction score

````

```python
from scipy.stats import f_oneway

# Separate satisfaction scores by branch
scores_a = data[data['Branch'] == 'A']['Satisfaction_Score']
scores_b = data[data['Branch'] == 'B']['Satisfaction_Score']
scores_c = data[data['Branch'] == 'C']['Satisfaction_Score']

# One-Way ANOVA test
f_stat, p_value = f_oneway(scores_a, scores_b, scores_c)

print(f"F-statistic = {f_stat:.3f}")
print(f"p-value = {p_value:.4f}")
if p_value < 0.05:
    print("Reject H₀: There is a significant difference in satisfaction scores between at least two branches.")
else:
    print("Fail to reject H₀: No significant difference in satisfaction scores across branches.")


```

```python
#Interpretation
if p-value<0.05:
   You reject the null hypothesis
elif:
   You failed to reject the null hypothesis
```

```python

```

## Problem – Evaluate Forecast Accuracy Using the Chi-Square Goodness of Fit Test

The city’s public transportation authority uses a forecasting model to estimate the number of metro passengers for each day of the week. These forecasts help manage train schedules, staffing, and platform operations.

Recently, actual passenger counts were collected and compared to the forecasted values to evaluate how well the model performs.

---

### Question

You are provided with the forecasted and observed number of passengers (in thousands) for each day of a week:

- **Forecasted Values (Expected):**  
  `[95, 110, 100, 130, 160, 210, 230]`

- **Observed Values (Actual):**  
  `[90, 105, 98, 135, 165, 205, 225]`

Using a **Chi-Square Goodness of Fit Test**, determine whether the forecast model provides an accurate estimate of daily passenger traffic.

---

### Hypotheses

- **Null Hypothesis (H₀):** There is no significant difference between the forecasted and observed values (i.e., the model is accurate).
- **Alternative Hypothesis (H₁):** There is a significant difference between the forecasted and observed values (i.e., the model is inaccurate).

---

### Test Parameters

- **Significance Level (α):** 0.10
- **Degrees of Freedom (df):** 6

---

### Instructions

1. **Perform the Chi-Square Goodness of Fit Test** using the given data.
2. **Calculate**:
   - Chi-Square Test Statistic
   - Critical Value at α = 0.10
3. **Compare** the test statistic with the critical value.
4. **State your conclusion**:
   - Do you **reject** or **fail to reject** the null hypothesis?
   - What does this imply about the **accuracy of the forecasting model**?

---

### Python Starter Code

````python
import numpy as np
from scipy.stats import chi2

# Data
expected = np.array([95, 110, 100, 130, 160, 210, 230])
observed = np.array([90, 105, 98, 135, 165, 205, 225])





```python
chi_square_statistic = 0
for o, e in zip(observed, expected):
    chi_square_statistic += ((o - e) ** 2) / e

print(f"Chi-Square Statistic = {chi_square_statistic:.4f}")

````

```python
alpha = 0.10
df = 6
critical_value = chi2.ppf(1 - alpha, df)

print(f"Critical Value at α = 0.10 (df=6): {critical_value:.4f}")

```

```python
if chi_square_statistic > critical_value:
    print("Reject H₀: There is a significant difference → Model is inaccurate.")
else:
    print("Fail to reject H₀: No significant difference → Model is reasonably accurate.")

```

Solve both questions below without using any built-in functions.

## Problem – Manual Covariance Calculation Between Study Hours and Exam Scores

A school counselor wants to understand how strongly the number of hours a student studies is related to their exam score.

She collected the following data:

| Student | Hours_Studied | Exam_Score |
| ------- | ------------- | ---------- |
| A       | 2             | 65         |
| B       | 4             | 70         |
| C       | 6             | 75         |
| D       | 8             | 85         |
| E       | 10            | 95         |

---

### Objective

Manually compute the **covariance** between `Hours_Studied` and `Exam_Score` **without using built-in functions** like `.cov()` or NumPy methods.

---

### Python Code for Manual Covariance Calculation

````python
# Dataset
import numpy as np
hours = [2, 4, 6, 8, 10]
scores = [65, 70, 75, 85, 95]
```python
cov_matrix = np.cov(hours, scores)
covariance = cov_matrix[0][1]
print("Covariance (Sample) using NumPy:", covariance)
````

```python
import pandas as pd
df = pd.DataFrame({
    'Hours_Studied': hours,
    'Exam_Score': scores
})
covariance = df['Hours_Studied'].cov(df['Exam_Score'])

print("Covariance (Sample) using Pandas:", covariance)

```

```python

```

## Problem – Manual Correlation Calculation Between Exercise Hours and Stress Level

A health researcher is analyzing the relationship between how many hours a person exercises per week and their reported stress level (on a scale of 1–100, where higher is more stress).

She collects data from 5 participants:

| Person | Exercise_Hours | Stress_Level |
| ------ | -------------- | ------------ |
| A      | 1              | 85           |
| B      | 3              | 75           |
| C      | 5              | 60           |
| D      | 7              | 55           |
| E      | 9              | 40           |

---

### Objective

Manually compute the **Pearson correlation coefficient** between `Exercise_Hours` and `Stress_Level` without using built-in correlation functions.

---

### Python Code for Manual Correlation Calculation

````python
# Data
import numpy as np
exercise = [1, 3, 5, 7, 9]
stress = [85, 75, 60, 55, 40]




```python
correlation_matrix = np.corrcoef(exercise, stress)
correlation = correlation_matrix[0, 1]

print(f"Correlation (NumPy): {correlation:.4f}")
import pandas as pd

df = pd.DataFrame({
    'Exercise': exercise,
    'Stress': stress
})

correlation = df['Exercise'].corr(df['Stress'])
print(f"Correlation (Pandas): {correlation:.4f}")
````
