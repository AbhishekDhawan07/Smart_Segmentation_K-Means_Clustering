<div align="center">

# 🛍️ Smart Segmentation — K-Means Clustering for Customer Segmentation

### *Turning Raw Customer Data into Actionable Business Intelligence*

<br>

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

<br>

> 🧠 *"Not all customers are the same. Smart businesses know exactly who they're talking to."*

<br>

---

### 🏆 Project Snapshot

| 👥 Customers | 📐 Features Used | 🔢 Optimal Clusters | 🤖 Algorithm | 🔍 Method |
|:---:|:---:|:---:|:---:|:---:|
| **200** | **2 (Income + Spending)** | **K = 5** | **K-Means++** | **Elbow Method** |

---

</div>

<br>

## 📁 Repository Structure

```
🗂️ Smart_Segmentation_K-Means_Clustering/
│
└── 📂 K-Means Clustering Project - Customer Segmentation/
    │
    ├── 📓 K_-_Means_Clustering_Project_-_Customer_Segmentation.ipynb   ← Full Notebook
    └── 📊 Mall_Customers.csv                                            ← Mall Dataset
```

---

## 🎯 What Is This Project?

This project applies **K-Means Clustering** — an unsupervised machine learning algorithm — to segment mall customers into **distinct behavioral groups** based on their **Annual Income** and **Spending Score**.

> 💡 Unlike classification, clustering has **no predefined labels**. The algorithm discovers hidden patterns entirely on its own!

**The Business Question Being Answered:**

```
🤔  "Which types of customers visit our mall — and how should we market to each group?"
```

---

## 📊 Dataset Deep Dive — Mall Customers

<div align="center">

| # | 🔢 Feature | 📋 Description | 🔬 Type | 🎯 Used for Clustering |
|:---:|:---|:---|:---:|:---:|
| 1 | `CustomerID` | Unique customer identifier | Integer | ❌ Excluded |
| 2 | `Gender` | Customer's gender (Male/Female) | Categorical | ⚙️ Encoded |
| 3 | `Age` | Customer age in years | Numerical | ✅ Yes |
| 4 | `Annual Income (k$)` | Yearly income in thousands USD | Numerical | ✅ Yes |
| 5 | `Spending Score (1-100)` | Mall-assigned score based on spending behavior | Numerical | ✅ Yes |

</div>

<br>

```
📦 Dataset at a Glance
├── 🗃️  Total Records         →  200 customers
├── 📐  Total Features         →  5 columns
├── 👩  Female Customers       →  112  (56%)
├── 👨  Male Customers         →   88  (44%)
├── 📅  Age Range              →  18 – 70 years
├── 💰  Income Range           →  $15k – $137k
└── 🛒  Spending Score Range   →  1 – 99
```

---

## 🔬 End-to-End ML Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  📥  LOAD DATA           →   Read Mall_Customers.csv         │
│         ↓                                                    │
│  🔍  EDA                 →   Shape, Info, Stats              │
│         ↓                                                    │
│  📊  UNIVARIATE ANALYSIS →   Gender, Income Distributions    │
│         ↓                                                    │
│  📈  BIVARIATE ANALYSIS  →   Income vs Spending, Age vs Score│
│         ↓                                                    │
│  🛠️  FEATURE ENGINEERING →   Outlier Capping + Encoding      │
│         ↓                                                    │
│  ⚖️  FEATURE SCALING     →   StandardScaler                  │
│         ↓                                                    │
│  📐  ELBOW METHOD        →   Find Optimal K                  │
│         ↓                                                    │
│  🤖  K-MEANS TRAINING    →   K=5, K-Means++ Init             │
│         ↓                                                    │
│  🎨  VISUALIZE CLUSTERS  →   Color-coded Scatter Plot        │
│         ↓                                                    │
│  💡  BUSINESS INSIGHTS   →   Interpret Each Segment          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧪 Step-by-Step Notebook Breakdown

<details>
<summary><b>📦 Step 1 — Importing Libraries</b> 🖱️ click to expand</summary>
<br>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

> 🔧 All major data science and ML libraries loaded — no deep learning frameworks needed. K-Means is lightweight and fast!

</details>

---

<details>
<summary><b>🔍 Step 2 — Exploratory Data Analysis (EDA)</b> 🖱️ click to expand</summary>
<br>

| 🔎 Check | 📋 Finding |
|:---|:---|
| Dataset Shape | `(200, 5)` |
| Missing Values | ✅ Zero null values |
| Duplicate Rows | ✅ Handled and dropped |
| Data Types | Integer + Object (Gender) |
| Gender Split | 112 Female 👩 \| 88 Male 👨 |

```python
df.shape       # → (200, 5)
df.info()      # → Column types & null counts
df.isnull().sum()  # → All zeros ✅
df.duplicated().sum()  # → Duplicates caught & removed
```

</details>

---

<details>
<summary><b>📊 Step 3 — Univariate & Bivariate Analysis</b> 🖱️ click to expand</summary>
<br>

**📌 Univariate Plots (Single Feature at a Time):**

| 📊 Plot | 🔍 Feature | 💡 Purpose |
|:---|:---|:---|
| Bar Chart | `Gender` | Count of Male vs Female customers |
| Histogram | `Annual Income (k$)` | Income distribution shape |
| Box Plot | `Annual Income (k$)` | Detect income outliers |

**📌 Bivariate Plots (Two Features Together):**

| 📈 Plot | 🔍 Features | 💡 Insight |
|:---|:---|:---|
| Scatter Plot | Income vs Spending Score | Reveals natural clusters visually |
| Scatter Plot | Age vs Spending Score | Shows age-based spending patterns |

```python
# Income vs Spending Score — the key clustering visualization
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
```

</details>

---

<details>
<summary><b>🛠️ Step 4 — Feature Engineering</b> 🖱️ click to expand</summary>
<br>

**🔹 Outlier Capping (IQR Method)**

Instead of dropping outliers (which loses data), extreme values were **capped** at the IQR boundaries:

```python
def cap_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(column, lower_bound, upper_bound)

df['Annual Income (k$)']      = cap_outliers(df['Annual Income (k$)'])
df['Spending Score (1-100)']  = cap_outliers(df['Spending Score (1-100)'])
df['Age']                     = cap_outliers(df['Age'])
```

**🔹 Gender Encoding**

Categorical text encoded to numerical values for compatibility:

```python
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# Male → 0  |  Female → 1
```

</details>

---

<details>
<summary><b>⚖️ Step 5 — Feature Selection & Scaling</b> 🖱️ click to expand</summary>
<br>

**Features selected for clustering:**

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

> 💡 These two features were chosen because they capture **purchasing power** and **spending behavior** — the most business-relevant dimensions for customer segmentation.

**StandardScaler applied:**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Final shape → (200, 2)
```

> ⚠️ K-Means is **distance-based**, so scaling is critical — without it, high-range features (like Income) dominate the clustering!

</details>

---

<details>
<summary><b>📐 Step 6 — Elbow Method (Finding Optimal K)</b> 🖱️ click to expand</summary>
<br>

The **Elbow Method** plots WCSS (Within-Cluster Sum of Squares) against different values of K. The "elbow" — where the rate of decrease sharply slows — reveals the optimal number of clusters.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method — Finding Optimal K")
```

```
WCSS vs K
│
│ ●
│   ●
│     ●
│       ●  ← Elbow at K=5 ✅
│          ● ● ● ● ● ●
└─────────────────────────── K
  1  2  3  4  5  6  7  8  9 10
```

> 🎯 **Optimal K = 5** identified from the elbow point!

</details>

---

<details>
<summary><b>🤖 Step 7 — K-Means Model Training</b> 🖱️ click to expand</summary>
<br>

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Attach cluster labels back to original dataframe
df['Cluster'] = y_kmeans
```

| ⚙️ Parameter | 🔧 Value | 💡 Why |
|:---|:---|:---|
| `n_clusters` | `5` | Elbow method result |
| `init` | `k-means++` | Smarter centroid initialization → faster convergence |
| `random_state` | `42` | Reproducible results |

</details>

---

<details>
<summary><b>🎨 Step 8 — Cluster Visualization</b> 🖱️ click to expand</summary>
<br>

Each of the 5 customer segments plotted with a unique color:

```python
plt.figure(figsize=(7, 6))
for cluster_id in range(5):
    plt.scatter(
        X_scaled[y_kmeans == cluster_id, 0],
        X_scaled[y_kmeans == cluster_id, 1],
        label=f'Cluster {cluster_id + 1}'
    )

plt.title("Customer Segments (K=5)")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.show()
```

</details>

---

## 💡 The 5 Customer Segments — Business Insights

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🎯 CLUSTER INTERPRETATION                                 │
├────────┬────────────────────────┬────────────────────────────────────────── ┤
│ Cluster│ 💰 Income | 🛒 Spending│ 🏷️  Customer Profile                      │
├────────┼────────────────────────┼───────────────────────────────────────────┤
│  🟢 1  │   Low    |   High      │ 🤑 Impulsive Buyers — spend beyond means   │
│  🟠 2  │   High   |   High      │ 💎 Premium Targets — the ideal segment!    │
│  🔴 3  │   High   |   Low       │ 💼 Careful Spenders — potential to unlock  │
│  🟣 4  │   Low    |   Low       │ 💤 Passive Browsers — minimal engagement   │
│  🔵 5  │   Mid    |   Mid       │ 📊 Average Joes — steady, balanced segment │
└────────┴────────────────────────┴───────────────────────────────────────────┘
```

### 📣 What Should Marketing Do?

| 🎯 Segment | 📢 Strategy |
|:---|:---|
| 🟢 Impulsive Buyers | Flash sales, time-limited offers, loyalty points |
| 🟠 Premium Targets | VIP programs, luxury products, exclusive deals |
| 🔴 Careful Spenders | Value-for-money messaging, ROI-driven campaigns |
| 🟣 Passive Browsers | Re-engagement campaigns, discount nudges |
| 🔵 Average Joes | Broad promotions, seasonal offers, bundles |

---

## 🧠 Why K-Means for Customer Segmentation?

```
┌───────────────────────────────┬────────────────────────────────────────────┐
│      ✨ K-Means Strength       │       🛍️  Why It Works Here                │
├───────────────────────────────┼────────────────────────────────────────────┤
│  Unsupervised — no labels     │  No pre-existing customer categories needed │
│  Scalable to large datasets   │  Works beyond our 200-row example           │
│  Fast & interpretable         │  Business teams can understand the output   │
│  Works well in 2D space       │  Income + Spending → clean 2D clusters      │
│  K-Means++ init               │  Better cluster quality, faster convergence │
└───────────────────────────────┴────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Smart_Segmentation_K-Means_Clustering.git
cd "Smart_Segmentation_K-Means_Clustering/K-Means Clustering Project - Customer Segmentation"
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3️⃣ Launch the Notebook
```bash
jupyter notebook "K_-_Means_Clustering_Project_-_Customer_Segmentation.ipynb"
```

> ✅ **Good news:** The dataset path in Cell 3 is already set to `"Mall_Customers.csv"` — just place the CSV in the same directory and you're good to go!

---

## 📋 Requirements

```
Python         >= 3.7
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📂 File Reference

| 📄 File | 📋 What's Inside |
|:---|:---|
| `K_-_Means_Clustering_Project_-_Customer_Segmentation.ipynb` | Complete pipeline: EDA → Feature Engineering → Elbow Method → K-Means → Visualization → Insights (56 cells) |
| `Mall_Customers.csv` | 200 mall customers with Gender, Age, Annual Income, and Spending Score |

---

## 🌍 Real-World Applications of Customer Segmentation

<div align="center">

> 🏬 **Retail & E-commerce** — personalized promotions per segment
> 🏦 **Banking** — credit card offers tailored to spending profiles
> 🎬 **Streaming Platforms** — recommendation engines by viewing behavior
> 🏨 **Hospitality** — loyalty programs targeting high-value guests
>
> Customer segmentation is a **$XX billion market** — and K-Means is the workhorse behind it all.

</div>

---

## 📌 Key Takeaways

```
✅  Unsupervised pipeline — no labels, no problem!
✅  Elbow Method used scientifically to choose K = 5
✅  Outlier capping preserves all 200 records (no data loss!)
✅  K-Means++ initialization for smarter, faster clustering
✅  5 distinct, business-interpretable customer segments discovered
✅  Color-coded scatter visualization for clear cluster separation
✅  Ready-to-present marketing strategy per segment
```

---

## 📜 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 💬 *"The goal is to turn data into information, and information into insight."*
#### — Carly Fiorina, Former CEO of HP

<br>

⭐ **Found this useful? Drop a star and share the knowledge!** ⭐

`🛍️ Built with passion for ML + Business Intelligence`

</div>
