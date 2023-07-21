# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

### Problem Statement:
The dataset used in this analysis contains information about cars, including their make, model, year, engine, and other properties. The goal is to perform data cleaning to ensure that the data is suitable for further analysis and modeling.

### Steps:
1. Importing the necessary libraries
2. Download the dataset and load it into a pandas DataFrame
3. Checking the datatypes
4. Dropping irrelevant columns
5. Renaming the columns
6. Dropping duplicate rows
7. Dropping null or missing values
8. Removing outliers

### 1. Importing the Necessary Libraries

We start by importing the required libraries for data manipulation, visualization, and statistical analysis.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
```

### 2. Download the Dataset and Load into DataFrame

We download the dataset from the provided link, extract the CSV file, and load it into a pandas DataFrame.

```python
df = pd.read_csv("/content/data.csv")
```

### 3. Check the Datatypes

We inspect the data types of each column in the DataFrame to understand the nature of the features.

```python
df.info()
```

### 4. Dropping Irrelevant Columns

Some columns in the dataset might not be relevant for our analysis. We identify such columns and drop them from the DataFrame.

```python
cols_to_drop = ["Engine Fuel Type", "Market Category", "Vehicle Style", "Popularity", "Number of Doors", "Vehicle Size"]
df = df.drop(cols_to_drop, axis=1)
```

### 5. Renaming the Columns

We rename the remaining columns to have more descriptive and useful names for model training.

```python
rename_cols = {"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode", "highway MPG": "MPG_H", "city mpg": "MPG_C", "MSRP": "Price"}
df.rename(columns=rename_cols, inplace=True)
```

### 6. Dropping the Duplicate Rows

Duplicate rows in the dataset can impact the analysis. We identify and remove duplicate rows.

```python
df = df.drop_duplicates()
```

### 7. Dropping the Null or Missing Values

Missing values in the data can affect the model training. We identify and remove rows with missing values.

```python
df = df.dropna()
```

### 8. Removing Outliers

Outliers can significantly impact the model's accuracy. We use two techniques, IQR and Z-score, to identify and remove outliers.

```python
# Using IQR Technique
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df2 = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Using Z-score Technique
z = np.abs(stats.zscore(df[l]))
threshold1 = 3
threshold2 = -3
df3 = df[((z < threshold1) | (z > threshold2)).all(axis=1)]
```

## Exploratory Data Analysis

### Problem Statement:
After data cleaning, we perform Exploratory Data Analysis (EDA) to understand the dataset better, identify patterns, and visualize relationships between different features.

### Steps:
1. Visualizing Univariate Distributions
   - Histograms & Density Plots
   - Bar Plots
   
2. Visualizing Bivariate Distributions
   - Scatterplots
   - Count Plots
   - Joint Distributions

3. Visualizing Multivariate Distributions
   - Pairplot
   - Heatmap

### 1. Visualizing Univariate Distributions

We use histograms and box plots to visualize the distribution of individual features.

```python
plt.figure(figsize=(15, 10))
for i in l:
    sns.displot(data=df, x=df[i], kde=True)
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x="Cylinders", y="Price", data=df)
plt.show()
```

### 2. Visualizing Bivariate Distributions

We use scatter plots, count plots, and joint distributions to explore relationships between two variables.

```python
sns.scatterplot(x="HP", y="Price", data=df)
plt.show()

sns.countplot(x="Transmission", hue="Drive Mode", data=df)
plt.show()

plt.figure(figsize=(12, 8))
df.Make.value_counts().nlargest(40).plot(kind="bar", figsize=(12, 8))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make')
plt.show()
```

### 3. Visualizing Multivariate Distributions

We use a pair plot to visualize multiple pairwise bivariate distributions in the dataset.

```python
sns.pairplot(df)
plt.show()
```

### 4. Heatmap

We create a heatmap to show the correlation between different features in the dataset.

```python
corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='BrBG', annot=True)
plt.show()
```

This combined Data Cleaning and Exploratory Data Analysis process helps us understand the data, identify patterns, and prepare the dataset for further analysis and modeling.
