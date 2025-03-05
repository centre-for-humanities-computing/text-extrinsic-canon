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
print()

encoder_dict = {}
# fit encoder to reshape certain cols of the dataset 
for col in ['publisher', 'published_under_gender', 'real_gender', 'nationality']:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_col = encoder.fit(df_two_classes[col].values.reshape(-1, 1))
    print(df_two_classes[col].value_counts())
    print()
    encoder_dict[col] = encoder_col

# %%
# Define feature combinations with consistent encoder usage
feature_combinations = {
    'embeddings': lambda df: np.stack(df['embedding'].values),
    'avg_sentence_length': lambda df: df['avg_sentence_length'].values.reshape(-1, 1),
    'price': lambda df: df['price'].values.reshape(-1, 1),
    'publisher': lambda df: encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),#publisher_encoder.transform(df['publisher'].values.reshape(-1, 1)),
    'embeddings_price': lambda df: np.hstack([np.stack(df['embedding'].values), df['price'].values.reshape(-1, 1)]),
    'embeddings_publisher': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                  encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1))]),
    'publisher_price': lambda df: np.hstack([encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
                                             df['price'].values.reshape(-1, 1)]),
    'embeddings_publisher_price': lambda df: np.hstack([np.stack(df['embedding'].values), 
                                                        encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
                                                        df['price'].values.reshape(-1, 1)]),
    'published_gender': lambda df: encoder_dict['published_under_gender'].transform(df['published_under_gender'].values.reshape(-1, 1)),
    'real_gender': lambda df: encoder_dict['real_gender'].transform(df['real_gender'].values.reshape(-1, 1)),
    'nationality': lambda df: encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1)),
    'publisher_nationality': lambda df: np.hstack([encoder_dict['publisher'].transform(df['publisher'].values.reshape(-1, 1)),
                                        encoder_dict['nationality'].transform(df['nationality'].values.reshape(-1, 1))])
}

# print list of feature combinations
print('feature combinations:', list(feature_combinations.keys()))

# Dictionary to store class-wise metrics for all feature combinations
results = {feature_set: {} for feature_set in feature_combinations}

# dictionary to store results for the confusion matrix
confusion_matrix_results = {feature_set: None for feature_set in feature_combinations}

print('full class sizes', df_two_classes[class_column].value_counts())
print('sample size:', df_two_classes[class_column].value_counts().min())


# %%

# we set a ceiling
# this is the max number that a book can be included in the "Other" sample
ceiling = 15 # should be 15

iteration_counter=0

# add filename counts, start at 0 (full and "other" category only)
filename_count_dict = {filename: 0 for filename in df_two_classes['filename']}
other_df = df_two_classes.loc[df_two_classes['category'] == 'O']
other_dict_count = {filename: 0 for filename in other_df['filename']}

# Store all predictions for all feature sets
all_predictions = []

class_column = 'category'

max_times = 10860

while sum(other_dict_count.values()) < max_times:
    print(f"Iteration: {iteration_counter + 1}") # let us know where we are
    iteration_counter += 1
    
    ### PART I: we create a balanced test and train set ###

    # Create balanced dataset and split into train/test
    min_class_size = df_two_classes[class_column].value_counts().min() # this will always be 114, the size of the canon group

    # create sample groups
    canon_group = df_two_classes[df_two_classes[class_column] == 'CANON'].sample(
        n=min_class_size, random_state=iteration_counter)  # this is always the same 114 books
    # Create other group
    other_group = df_two_classes[df_two_classes[class_column] == 'O'].sample(
        n=min_class_size, random_state=iteration_counter) 

    # merge
    merged_groups = pd.concat([other_group, canon_group], ignore_index=True) # put together
    # shuffle
    balanced_df = merged_groups.sample(frac=1, random_state=iteration_counter).reset_index(drop=False)

    X_full = balanced_df.drop(columns=[class_column])
    y_full = balanced_df[class_column].values

    # create train, test, based on filenames
    train_filenames, test_filenames = train_test_split(
    balanced_df['filename'],  # Use filenames for stratified splitting
    test_size=test_size,
    random_state=iteration_counter,
    stratify=y_full)

    train_df = balanced_df[balanced_df['filename'].isin(train_filenames)]
    test_df = balanced_df[balanced_df['filename'].isin(test_filenames)]

    ### PART II: we want to resample so that every "O" book is in the test-set 15 times ###
    picked_test_set_with_filenames = []

    while len(picked_test_set_with_filenames) < len(test_df):

        # fill up pool so it does not break the very first time it runs
        pool = pd.DataFrame({})

        # Track filenames added in this iteration to prevent duplicates
        stuff_that_was_added = []

        for _, row in test_df.iterrows():
            filename = row['filename']
            class_of_book = row['category']

            # Always append "CANON" books to the test set
            if class_of_book == 'CANON':
                picked_test_set_with_filenames.append(row)
                filename_count_dict[filename] += 1

            # Append "O" books if they haven't reached the ceiling
            elif filename_count_dict[filename] < ceiling:
                picked_test_set_with_filenames.append(row)
                other_dict_count[filename] += 1
                filename_count_dict[filename] += 1

            else:
                print(f'!! Ceiling hit for {filename}')

                # Define the pool of available "O" books
                pool = df_two_classes[
                    (df_two_classes[class_column] == 'O') &                      # Must be in the "O" category
                    (~df_two_classes['filename'].isin(train_df['filename'])) &  # Exclude train filenames
                    (~df_two_classes['filename'].isin(test_df['filename'])) &   # Exclude current test filenames
                    (~df_two_classes['filename'].isin(stuff_that_was_added))    # Exclude stuff already sampled in this iteration
                ]

                # Exclude books that already hit the ceiling
                pool = pool[~pool['filename'].isin(
                    [k for k, v in other_dict_count.items() if v >= ceiling]
                )]

                if pool.empty:
                    print("No 'O' samples left in the pool to resample. Stopping iteration.")
                    print("Cannot create a last full test-set")
                    break

                # Sample a new book from the pool
                temp_sample = pool.sample(n=1, random_state=iteration_counter).iloc[0]

                # Add the sampled row to the test set
                picked_test_set_with_filenames.append(temp_sample)
                sample_from_pool_filename = temp_sample['filename']

                # note down that this book was added in this iteration
                stuff_that_was_added.append(sample_from_pool_filename)

                # Update counters
                filename_count_dict[sample_from_pool_filename] += 1
                other_dict_count[sample_from_pool_filename] += 1

        # If the inner loop exited due to pool being empty, we stop trying to fill the test set
        if pool.empty:
            print("Stopping due to insufficient 'O' books available in the pool.")
            break  # Break the outer while loop as well, no more samples to add

    # Check if the test set was completed
    if len(picked_test_set_with_filenames) < len(test_df):
        print("Test set could not be fully resampled due to lack of available samples.")
        break  # Exit the outer while loop as well if the test set wasn't completed

    # Convert the final test set to a DataFrame if needed
    final_test_df = pd.DataFrame(picked_test_set_with_filenames)

    # make sure that this is the same length
    print('len of originally sampled testset:', len(test_df))
    print('len of revised testset:', len(final_test_df))
    print('len of train_set:', len(train_df))

    ### PART III: Now we loop over feature sets:
    for feature_set_name, feature_set_func in feature_combinations.items():
        
        # Apply feature transformation and train classifier
        X_train = feature_set_func(train_df)
        X_test = feature_set_func(final_test_df)
        y_train = train_df[class_column].values
        y_test = final_test_df[class_column].values

        clf = RandomForestClassifier(n_estimators=100, random_state=iteration_counter)
        # train
        clf.fit(X_train, y_train) 
        # test
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) ####### PF: what does zero_division do?

        # Save predictions
        for item, true_label, pred_label in zip(final_test_df['filename'], y_test, y_pred):
            all_predictions.append({
            'filename': item,
            'true_class': true_label,
            'predicted_class': pred_label,
            'feature_set': feature_set_name,
            'iteration': iteration_counter,
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

    print(f'Sum of filename count dict: {sum(filename_count_dict.values())}')
    print(f'Sum of other count dict: {sum(other_dict_count.values())}')


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
                'iteration': iteration_counter,
                'true_class': pred['true_class'],
                'predicted_class': pred['predicted_class'],
                'feature_set': pred['feature_set']
            })

date = '250124'


# Save all false positive embeddings to a CSV
pd.DataFrame(embeddings_false_positives).to_csv(f"results/embeddings_false_positives_{date}_15_ceiling.csv", index=False)

# Save comparison results to a CSV
pd.DataFrame(comparison_results).to_csv(f"results/comparison_results_{date}_15_ceiling.csv", index=False)

# save all predictions to a CSV
pd.DataFrame(all_predictions).to_csv(f"results/all_predictions_{date}_15_ceiling.csv", index=False)
