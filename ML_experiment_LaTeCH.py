# %%
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
!pip install nltk
!pip install seaborn
!pip install datasets
!pip install openpyxl

# %%
# importing libraries
import pandas as pd
import numpy as np
import json

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# %%
# load the Memo-corpus dataset w. metadata
ds = load_dataset("chcaa/memo-canonical-novels")

# make df
meta = pd.DataFrame(ds['train'])

# %%
# we want to use only 2 categories (O, CANON)

# define the nice labels for the categories
nice_labels = {'O': 'Other', 'HISTORICAL': 'Historical', 'CANON': 'Canon'}

# Combine categories in the 'CATEGORY' column
meta['category'] = meta['category'].replace({
    'LEX_CANON': 'CANON',
    'CE_CANON': 'CANON',
    'CANON_HISTORICAL': 'CANON',
    'HISTORICAL': 'O' # canon books that are also historical will be considered canon
})

# replace error in nationality
meta['nationality'] = meta['nationality'].replace({'teacher in denmark, maybe german': 'de'})

# %%
# Load the embeddings data (previous work)
# for embedding extraction, see: https://github.com/centre-for-humanities-computing/memo-canonical-novels
with open('data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author.json', 'r') as f:
    embeddings_data = [json.loads(line) for line in f]

embeddings_df = pd.DataFrame(embeddings_data)

# %%
# make sure that the embeddings are in the right format
embeddings_df['embedding'] = embeddings_df['embedding'].apply(np.array)

# Merge embeddings with the main dataframe
merged_df = pd.merge(meta, embeddings_df, left_on='filename', right_on='filename')

# add sentence length as a baseline feature for the model (use nltk)
merged_df['avg_sentence_length'] = merged_df['text'].apply(lambda x: np.mean([len(sent_tokenize(s)) for s in sent_tokenize(x)]))

# drop all columns that we do not need
merged_df = merged_df[['filename',
                        'published_under_gender','real_gender', 'nationality',
                        'publisher', 'price', 'category',
                        'embedding', 'avg_sentence_length']].copy()

# %%
## ML CONFIGURATION
# define the column used for the class labels
class_column = 'category'
print(merged_df[class_column].value_counts())

# define the testset size and the number of iterations
test_size = 0.1
num_iterations = 50
print('test size:', test_size)
print('num iterations:', num_iterations)

# %%
# ML run in which we try to distinguish only two classes

# copy our df for ML
df_two_classes = merged_df.copy()

print(df_two_classes[class_column].value_counts())

encoder_dict = {}
# fit encoder to reshape certain cols of the dataset 
for col in ['publisher', 'published_under_gender', 'real_gender', 'nationality']:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_col = encoder.fit(df_two_classes[col].values.reshape(-1, 1))
    print(df_two_classes[col].value_counts())
    print()
    encoder_dict[col] = encoder_col

# %%

feature_combinations = {
    # Single features
    'embeddings': lambda df: np.stack(df['embedding'].values),
    'avg_sentence_length': lambda df: df['avg_sentence_length'].values.reshape(-1, 1),
    'price': lambda df: df['price'].values.reshape(-1, 1),
    'publisher': lambda df: encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
    'published_gender': lambda df: encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
    'real_gender': lambda df: encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
    'nationality': lambda df: encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1)),

    # Two-feature combinations
    'embeddings_price': lambda df: np.hstack([np.stack(df['embedding'].values), df['price'].values.reshape(-1, 1)]),
    'embeddings_publisher': lambda df: np.hstack([np.stack(df['embedding'].values), encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1))]),
    'embeddings_published_gender': lambda df: np.hstack([np.stack(df['embedding'].values), encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))]),
    'embeddings_real_gender': lambda df: np.hstack([np.stack(df['embedding'].values), encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))]),
    'embeddings_nationality': lambda df: np.hstack([np.stack(df['embedding'].values), encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))]),
    'price_publisher': lambda df: np.hstack([df['price'].values.reshape(-1, 1), encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1))]),
    'price_published_gender': lambda df: np.hstack([df['price'].values.reshape(-1, 1), encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))]),
    'price_real_gender': lambda df: np.hstack([df['price'].values.reshape(-1, 1), encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))]),
    'price_nationality': lambda df: np.hstack([df['price'].values.reshape(-1, 1), encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))]),
    'publisher_published_gender': lambda df: np.hstack([encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)), encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))]),
    'publisher_real_gender': lambda df: np.hstack([encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)), encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))]),
    'publisher_nationality': lambda df: np.hstack([encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)), encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))]),

    # Three-feature combinations
    'embeddings_price_publisher': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1))
    ]),
    'embeddings_price_published_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_price_real_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_price_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_publisher_published_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_publisher_real_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_publisher_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_published_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_real_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'price_publisher_published_gender': lambda df: np.hstack([
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))
    ]),
    'price_publisher_real_gender': lambda df: np.hstack([
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))
    ]),
    'price_publisher_nationality': lambda df: np.hstack([
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    
    # Four-feature combinations
    'embeddings_price_publisher_published_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_price_publisher_real_gender': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1))
    ]),
    'embeddings_price_publisher_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_price_published_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_price_real_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_publisher_published_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_publisher_real_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'price_publisher_published_gender_nationality': lambda df: np.hstack([
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'price_publisher_real_gender_nationality': lambda df: np.hstack([
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),

    # Five-feature combinations
    'embeddings_price_publisher_published_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),
    'embeddings_price_publisher_real_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ]),

    # Six-feature combination
    'embeddings_price_publisher_published_gender_real_gender_nationality': lambda df: np.hstack([
        np.stack(df['embedding'].values),
        df['price'].values.reshape(-1, 1),
        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
        encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
        encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))
    ])
}

#%%

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}

# dictionary to store results for the confusion matrix
confusion_matrix_results = {feature_set: None for feature_set in feature_combinations}

print('full class sizes', df_two_classes[class_column].value_counts())
print('sample size:', df_two_classes[class_column].value_counts().min())

# %% Run experiments

# Store all predictions for all feature sets
all_predictions = []

for i in range(num_iterations):
    print(f"Iteration: {i + 1}/{num_iterations}")
    
    # Create balanced dataset and split into train/test
    min_class_size = df_two_classes[class_column].value_counts().min() # this will always be 114, the size of the canon group
    
    balanced_dfs = [
        group.sample(n=min_class_size, random_state=i)
        for _, group in df_two_classes.groupby(class_column)
    ]

    balanced_df = pd.concat(balanced_dfs, ignore_index=True) # put together
    balanced_df = balanced_df.sample(frac=1, random_state=i).reset_index(drop=True) # shuffle
    

    X_full = balanced_df.drop(columns=[class_column])
    y_full = balanced_df[class_column].values

    # train/test split
    train_idx, test_idx = train_test_split(X_full.index,
        test_size=test_size, random_state=i, stratify=y_full
    )
    train_df = balanced_df.loc[train_idx]
    test_df = balanced_df.loc[test_idx]

    # loop over feature sets
    for feature_set_name, feature_set_func in feature_combinations.items():
        print(f"Evaluating feature set: {feature_set_name}")
        
        # Apply feature transformation and train classifier
        X_train = feature_set_func(train_df)
        X_test = feature_set_func(test_df)
        y_train = train_df[class_column].values
        y_test = test_df[class_column].values

        clf = RandomForestClassifier(n_estimators=100, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Save predictions
        for idx, true_label, pred_label in zip(test_idx, y_test, y_pred):
            all_predictions.append({
                'index': idx,
                'filename': balanced_df.loc[idx, 'filename'],
                'true_class': true_label,
                'predicted_class': pred_label,
                'feature_set': feature_set_name,
                'iteration': i
            })

        # Update confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if confusion_matrix_results[feature_set_name] is None:
            confusion_matrix_results[feature_set_name] = cm
        else:
            confusion_matrix_results[feature_set_name] += cm

        # Track class-wise performance
        if feature_set_name not in results: # creating a dict in results for current run (if does not exist)
            results[feature_set_name] = {} 

        class_performance = results[feature_set_name].get('class_performance', {})
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if class_name not in class_performance:
                class_performance[class_name] = {'precision': [], 'recall': [], 'f1-score': []}
            class_performance[class_name]['precision'].append(metrics['precision'])
            class_performance[class_name]['recall'].append(metrics['recall'])
            class_performance[class_name]['f1-score'].append(metrics['f1-score'])
        
        # Save updated class performance back to results
        results[feature_set_name]['class_performance'] = class_performance

# Finalize metrics (average over iterations)
for feature_set_name, feature_set_results in results.items():
    class_performance = feature_set_results['class_performance']
    results[feature_set_name]['final_metrics'] = {
        class_name: {
            'mean_precision': np.mean(scores['precision']),
            'std_precision': np.std(scores['precision']),
            'mean_recall': np.mean(scores['recall']),
            'std_recall': np.std(scores['recall']),
            'mean_f1': np.mean(scores['f1-score']),
            'std_f1': np.std(scores['f1-score']),
        }
        for class_name, scores in class_performance.items()
    }
    # Normalize confusion matrix
    confusion_matrix_results[feature_set_name] = (
        confusion_matrix_results[feature_set_name] / num_iterations
    )

# %%
confusion_matrix_results
# %%
# Identify false positives for 'embeddings'
embeddings_false_positives = [
    pred for pred in all_predictions
    if pred['feature_set'] == 'embeddings' and pred['true_class'] == 'O' and pred['predicted_class'] == 'CANON'
]

# Check how they are predicted by other feature sets
comparison_results = []
for fp in embeddings_false_positives:
    filename = fp['filename']

    # Find predictions for the same filename and iteration across all feature sets
    for pred in all_predictions:
        if pred['filename'] == filename:
            comparison_results.append({
                'filename': filename,
                'iteration': i,
                'true_class': pred['true_class'],
                'predicted_class': pred['predicted_class'],
                'feature_set': pred['feature_set']
            })

date = '250124'

# Save all false positive embeddings to a CSV
pd.DataFrame(embeddings_false_positives).to_csv(f"results/embeddings_false_positives_{date}.csv", index=False)

# Save comparison results to a CSV
pd.DataFrame(comparison_results).to_csv(f"results/comparison_results_correct_{date}.csv", index=False)


# %%
# Display results
for feature_set_name, class_metrics in results.items():
    print(f"Feature Set: {feature_set_name}")
    for class_name, metrics in class_metrics['final_metrics'].items():
        print(f"  Class {class_name}:")
        print(f"    Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    Mean Recall: {metrics['mean_recall']:.3f}")
        print(f"    Mean F1-Score: {metrics['mean_f1']:.3f}")
        # and get the SD of the F1 score
        print('    ..')
        print(f"    STD F1-Score: {metrics['std_f1']:.3f}")
    print()

# save them to a txt in results folder
with open(f'results/ML_results_LaTeCH_{date}.txt', 'w') as f:
    for feature_set_name, class_metrics in results.items():
        f.write(f"Feature Set: {feature_set_name}\n")
        for class_name, metrics in class_metrics['final_metrics'].items():
            f.write(f"  Class {class_name}:\n")
            f.write(f"    Mean Precision: {metrics['mean_precision']:.3f}\n")
            f.write(f"    Mean Recall: {metrics['mean_recall']:.3f}\n")
            f.write(f"    Mean F1-Score: {metrics['mean_f1']:.3f}\n")
            f.write('    ..\n')
            f.write(f"    STD F1-Score: {metrics['std_f1']:.3f}\n")
        f.write('\n')

# %%

# Generate LaTeX table
latex_table = r"""
\begin{table*}
    \centering
    \small
    \begin{tabular}{l|cc|cc|cc}
    \toprule
    & \multicolumn{2}{c|}{\textbf{Precision}} & \multicolumn{2}{c|}{\textbf{Recall}} & \multicolumn{2}{c}{\textbf{F1-score}} \\
    \textbf{Feature set} & \textit{Canon} & \textit{Other} & \textit{Canon}  & \textit{Other} & \textit{Canon}  & \textit{Other} \\
    \midrule
"""

for feature_set, metrics in results.items():
    # Access final_metrics for CANON and O
    canon_metrics = metrics['final_metrics']['CANON']
    other_metrics = metrics['final_metrics']['O']
    
    # Format the row
    row = f"    {feature_set} & {canon_metrics['mean_precision']:.3f} & {other_metrics['mean_precision']:.3f} & "
    row += f"{canon_metrics['mean_recall']:.3f} & {other_metrics['mean_recall']:.3f} & "
    row += f"{canon_metrics['mean_f1']:.3f} & {other_metrics['mean_f1']:.3f} \\\\"
    latex_table += row + "\n"

latex_table += r"""
    \bottomrule
    \end{tabular}
    \caption{Performance metrics for different feature sets.}
    \label{tab:performance_metrics}
\end{table*}
"""

# %%

# Save the LaTeX table to a file
output_file = "latex_tables/latex_table.tex"
with open(output_file, "w") as file:
    file.write(latex_table)

print(f"LaTeX table saved to {output_file}")
