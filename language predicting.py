import pandas as pd
import numpy as np
import pickle
import utils

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

pair_train_sample = pickle.load(open('samples/pair_train_sample.pkl', mode='rb'))
pair_holdout_sample = pickle.load(open('samples/pair_holdout_sample.pkl', mode='rb'))

interplay_features = utils.INTERPLAY_COLS
style_features = utils.ARG_COLS
combined_features = interplay_features + ['# words'] + style_features

# Merging the data altogether
'''
pos_root_int_df = pd.read_pickle('language indicators results/interplay/data/pos_root_int_df')
pos_full_int_df = pd.read_pickle('language indicators results/interplay/data/pos_full_int_df')
neg_root_int_df = pd.read_pickle('language indicators results/interplay/data/neg_root_int_df')
neg_full_int_df = pd.read_pickle('language indicators results/interplay/data/neg_full_int_df')


pos_root_words_df = pd.read_pickle('language indicators results/words/data/pos_root_words_df')
pos_full_words_df = pd.read_pickle('language indicators results/words/data/pos_full_words_df')
neg_root_words_df = pd.read_pickle('language indicators results/words/data/neg_root_words_df')
neg_full_words_df = pd.read_pickle('language indicators results/words/data/neg_full_words_df')


pos_root_arg_df = pd.read_pickle('language indicators results/argument/data/pos_root_arg_df')
pos_full_arg_df = pd.read_pickle('language indicators results/argument/data/pos_full_arg_df')
neg_root_arg_df = pd.read_pickle('language indicators results/argument/data/neg_root_arg_df')
neg_full_arg_df = pd.read_pickle('language indicators results/argument/data/neg_full_arg_df')

interplay_features = utils.INTERPLAY_COLS
style_features = ['# words'] + utils.ARG_COLS

pos_root_df = pd.concat(objs=[pos_root_int_df.reset_index(drop=True), pos_root_words_df.reset_index(drop=True), pos_root_arg_df.reset_index(drop=True)], axis=1)
pos_full_df = pd.concat(objs=[pos_full_int_df.reset_index(drop=True), pos_full_words_df.reset_index(drop=True), pos_full_arg_df.reset_index(drop=True)], axis=1)
neg_root_df = pd.concat(objs=[neg_root_int_df.reset_index(drop=True), neg_root_words_df.reset_index(drop=True), neg_root_arg_df.reset_index(drop=True)], axis=1)
neg_full_df = pd.concat(objs=[neg_full_int_df.reset_index(drop=True), neg_full_words_df.reset_index(drop=True), neg_full_arg_df.reset_index(drop=True)], axis=1)

pos_root_df['effective'] = pd.DataFrame(['effective' for _ in range(500)])
pos_full_df['effective'] = pd.DataFrame(['effective' for _ in range(500)])
neg_root_df['effective'] = pd.DataFrame(['not effective' for _ in range(500)])
neg_full_df['effective'] = pd.DataFrame(['not effective' for _ in range(500)])

pd.to_pickle(pos_root_df, 'pos_root_df')
pd.to_pickle(pos_full_df, 'pos_full_df')
pd.to_pickle(neg_root_df, 'neg_root_df')
pd.to_pickle(neg_full_df, 'neg_full_df')
'''

# training data
pos_root_df = pd.read_pickle('language indicators results/combined/pos_root_df')
pos_full_df = pd.read_pickle('language indicators results/combined/pos_full_df')
neg_root_df = pd.read_pickle('language indicators results/combined/neg_root_df')
neg_full_df = pd.read_pickle('language indicators results/combined/neg_full_df')


# merge the positive and negative examples on top of each other -> this ensures in K-Fold that comments with same OP are in the same fold together.
root_df = pd.concat([pos_root_df,neg_root_df]).sort_index()
root_df = root_df.reset_index(drop=True)

full_df = pd.concat([pos_full_df,neg_full_df]).sort_index()
full_df = full_df.reset_index(drop=True)


# pd.to_pickle(root_df, 'root_df')
# pd.to_pickle(full_df, 'full_df')

root_df = pd.read_pickle('language indicators results/combined/root_df')
full_df = pd.read_pickle('language indicators results/combined/full_df')
# separate into X and y
X_train_root_df = root_df.drop('effective', axis=1)
y_train_root = root_df['effective']
X_train_full_df = full_df.drop('effective', axis=1)
y_train_full = full_df['effective']


# standardize the features to unit variance
scaler_1 = StandardScaler()
scaler_2 = StandardScaler()
X_train_root = pd.DataFrame(scaler_1.fit_transform(X_train_root_df), columns=combined_features)
X_train_full = pd.DataFrame(scaler_2.fit_transform(X_train_full_df), columns=combined_features)


# calculate all feature scores for holdout data, root and full and standardize based on our scalers for the train data
holdout_root = pd.read_pickle('language indicators results/holdout/root_df')
holdout_full = pd.read_pickle('language indicators results/holdout/full_df')

X_holdout_root_df = holdout_root.drop('effective', axis=1)
y_holdout_root = holdout_root['effective']
X_holdout_full_df = holdout_full.drop('effective', axis=1)
y_holdout_full = holdout_full['effective']


X_holdout_root = pd.DataFrame(scaler_1.transform(X_holdout_root_df), columns=combined_features)
X_holdout_full = pd.DataFrame(scaler_2.transform(X_holdout_full_df), columns=combined_features)

# features that passed in the paired t-test with Bonferroni correction and thus are significant
interplay_features_root = ['# common all', '# common stop', 'jaccard stop', 'op frac all', 'op frac content', 'op frac stop', 'reply frac all', 'reply frac stop']
interplay_features_full = ['# common all', '# common content', '# common stop', 'jaccard stop', 'op frac all', 'op frac content', 'op frac stop', 'reply frac all', 'reply frac content', 'reply frac stop']

style_features_root = ['# indefinite articles', '# definite articles', '# 1st person plural pronouns', '# of links', 'V', 'D', '# of sentences', '# of paragraphs', 'Flesch-Kincaid Readability']
style_features_full = ['# indefinite articles', '# definite articles', '# 1st person pronouns', '# 1st person plural pronouns', '# 2nd person pronouns', '# of links', '# of quotes', 'V', 'A', 'D', '# of sentences', '# of paragraphs', 'Flesch-Kincaid Readability']


# # words passed significance for both root and full
combined_features_root = interplay_features_root + ['# words'] + style_features_root
combined_features_full = interplay_features_full + ['# words'] + style_features_full

root_feature_sets = [['# words'], interplay_features_root, style_features_root, combined_features_root]
full_feature_sets = [['# words'], interplay_features_full, style_features_full, combined_features_full]

# logistic regression on each feature set for root and for full
root_scores = []
full_scores = []

for feature in root_feature_sets:
    X_root = X_train_root[feature]
    log_reg_1 = LogisticRegressionCV(cv=KFold(n_splits=5), solver='liblinear', random_state=None, penalty='l1', scoring='accuracy')
    
    log_reg_1.fit(X_root, y_train_root)

    root_score = log_reg_1.score(X_holdout_root[feature], y_holdout_root)
    
    root_scores.append(root_score)

for feature in full_feature_sets:
    X_full = X_train_full[feature]
    log_reg_2 = LogisticRegressionCV(cv=KFold(n_splits=5), solver='liblinear', random_state=None, penalty='l1', scoring='accuracy')
    
    log_reg_2.fit(X_full, y_train_full)

    full_score = log_reg_2.score(X_holdout_full[feature], y_holdout_full)

    full_scores.append(full_score)

feature_category_list = ['# words', 'interplay features', 'style features', 'combined']
scores_df = pd.DataFrame({feature_category_list[i]:[root_scores[i], full_scores[i]] for i in range(4)})
print(scores_df)
pd.to_pickle(scores_df, 'scores.pkl')