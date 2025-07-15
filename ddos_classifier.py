import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import pickle

import warnings
warnings.filterwarnings('ignore')


dfps_train = []
dfps_test = []

for dirname, _, filenames in os.walk('./Datasets/'):
    for filename in filenames:
        if filename.endswith('-training.csv'):
            dfp = os.path.join(dirname, filename)
            dfps_train.append(dfp)
            #print(dfp)
        elif filename.endswith('-testing.csv'):
            dfp = os.path.join(dirname, filename)
            dfps_test.append(dfp)
            #print(dfp)

train_prefixes = [dfp.split('/')[-1].split('-')[0] for dfp in dfps_train]
test_prefixes = [dfp.split('/')[-1].split('-')[0] for dfp in dfps_test]

common_prefixes = list(set(train_prefixes).intersection(test_prefixes))

# Filter the dataframes to only include the common prefixes
dfps_train = [dfp for dfp in dfps_train if dfp.split('/')[-1].split('-')[0] in common_prefixes]
dfps_test = [dfp for dfp in dfps_test if dfp.split('/')[-1].split('-')[0] in common_prefixes]


train_df = pd.concat([pd.read_csv(dfp) for dfp in dfps_train], ignore_index=True)
test_df = pd.concat([pd.read_csv(dfp) for dfp in dfps_test], ignore_index=True)


#print(train_df.shape, test_df.shape)

# Map the labels to the same format

label_mapping = {
    'DrDoS_UDP': 'UDP',
    'BENIGN': 'BENIGN'  # Already matches
}

test_df[" Label"] = test_df[" Label"].map(label_mapping)


train_df = train_df[train_df[" Label"] != "MSSQL"]

#print(train_df[" Label"].value_counts())
#print(test_df[" Label"].value_counts()) 

def grab_col_names(data, cat_th=10, car_th=20):

    # Categorical columns and categorical but high-cardinality columns
    cat_cols = [col for col in data.columns if data[col].dtypes == "O"]
    num_but_cat = [col for col in data.columns if data[col].nunique() < cat_th and data[col].dtypes != "O"]
    high_card_cat_cols = [col for col in data.columns if data[col].nunique() > car_th and data[col].dtypes == "O"]

    # Combine Object type columns and Low-unique-value numeric columns into cat_cols
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in high_card_cat_cols]

    # Numerical columns excluding those considered as categorical
    num_cols = [col for col in data.columns if data[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

     # Display information about the dataset
    print(f"Observations: {data.shape[0]}")
    print(f"Variables: {data.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"High Cardinality Categorical Columns: {len(high_card_cat_cols)}")
    print(f"Number but Categorical Columns: {len(num_but_cat)}")
    print("\n")

    return cat_cols, num_cols, high_card_cat_cols

cat_cols, num_cols, high_card_cat_cols = grab_col_names(train_df)


""" print(f"Catergorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"High Cardinality Categorical Columns: {high_card_cat_cols}") """


""" for i in cat_cols:
    print(i, train_df[i].unique())

for i in train_df.columns:
    if train_df[i].nunique() == 1:
        print(i)
 """
# Total number of missing values

train_df = train_df.dropna()

single_val_cols = [col for col in train_df.columns if train_df[col].nunique() == 1]
single_val_cols

# Remove columns with a single unique value
train_df.drop(single_val_cols, axis=1, inplace = True)
test_df.drop(single_val_cols, axis=1, inplace = True)

#Remove object variables

colunas_nao_usadas = ['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', 
                      ' Destination IP', ' Destination Port','SimillarHTTP', ' Timestamp']
train_df = train_df.drop(columns=[col for col in colunas_nao_usadas if col in train_df.columns])
test_df = test_df.drop(columns=[col for col in colunas_nao_usadas if col in test_df.columns])

# Shape of the dataset after removing columns with a single unique value
print(train_df.shape, test_df.shape)

# Select only numeric columns
numerical_df = train_df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numerical_df.corr().abs()

# Generate a boolean mask for the upper triangle
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

# Select the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(mask)

# Find the columns with correlation of 0.8 or higher
high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]

train_df.drop(high_corr_cols, axis=1, inplace=True)
test_df.drop(high_corr_cols, axis=1, inplace=True)

# 3. Tratar inf/NaN
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)



X_train, X_val, y_train, y_val = train_test_split(train_df.drop(" Label", axis=1), train_df[" Label"], test_size=0.2, random_state=42)
X_test, y_test = test_df.drop(" Label", axis=1), test_df[" Label"]


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Label mapping for the target variable
label_map = {index: Label for index, Label in enumerate(le.classes_)}


# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

""" 
# Train and evaluate models
def train_model(X_train, X_test, y_train, y_test):
    # Initialize models
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "Extra Trees": ExtraTreesClassifier(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        "XGBoost": XGBClassifier(),
    }

    scores_list = []

    # Initialize plot for ROC curves
    plt.figure(figsize=(10, 8))

     # Train and evaluate models with progress bar
    for name, model in tqdm(classifiers.items(), desc=f"Training Models"):
        print(f"Training {name}......")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate predictions for multiclass
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")

        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))

        # ROC curve for the model (for each class)
        y_proba = model.predict_proba(X_test)
        for i in range(len(np.unique(y_train))):
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
            plt.plot(fpr, tpr, label=f'{name} - Class {i} (AUC = {roc_auc:.4f})')

        # Append scores to the list
        scores_list.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV Score": cv_score
        })

     # Finalize the ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models (Multiclass)')
    plt.legend(loc='lower right')
    plt.show()

    # Create and display the DataFrame of scores
    scores = pd.DataFrame(scores_list)
    return scores

# Run the function
scores = train_model(X_train, X_val, y_train, y_val)
print("Testing Scores between different models:")
 """