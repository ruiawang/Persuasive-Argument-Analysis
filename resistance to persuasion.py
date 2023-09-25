import pandas as pd
import utils
import pickle
import re
from scipy import stats


op_train_sample = pickle.load(open('samples/op_train_sample.pkl', mode='rb'))
# test_sub = op_train_sample[4]

arg_features = ['# words'] + utils.ARG_COLS
'''
op_data = {feature:[] for feature in arg_features+['delta awarded']}
for submission in op_train_sample:
    
    tokens = utils.tokenize(utils.remove_footer(submission))
    op_data['# words'].append(utils.word_count(tokens))

    for feature, value in utils.word_category_info(tokens).items():
        op_data[feature].append(value)
    for feature, value in utils.word_score_info(tokens).items():
        op_data[feature].append(value)
    for feature, value in utils.entire_text_features(submission['selftext']).items():
        op_data[feature].append(value)

    if submission['delta_label'] == True:
        op_data['delta awarded'].append('malleable')
    else:
        op_data['delta awarded'].append('resistant')
op_df = pd.DataFrame(op_data)
pd.to_pickle(op_df, 'op_df')
print(op_df.head())
'''
op_df = pd.read_pickle('persuasion data/op_df')

pos_df = op_df.where(op_df['delta awarded'] == 'malleable').dropna().reset_index(drop=True)
neg_df = op_df.where(op_df['delta awarded'] == 'resistant').dropna().reset_index(drop=True)

# print(pos_df.info(), neg_df.info()) # 141 malleable, 359 resistant

alpha = 0.05
adjusted_alpha = alpha/len(arg_features)
p_value_dict = {feature:[] for feature in arg_features}
for feature in arg_features:
    t_statistic, p_value = stats.ttest_ind(pos_df[feature], neg_df[feature], equal_var=False)
    
    pos_mean = pos_df[feature].mean()
    neg_mean = neg_df[feature].mean()
    if pos_mean > neg_mean:
        direction = 1
    elif pos_mean < neg_mean:
        direction = -1
    else:
        direction = 0
    p_value_dict[feature] = [p_value, direction]
pickle.dump(p_value_dict, open('p_values.pkl', mode='wb'))
# print(sorted(p_value_dict.keys(), key = lambda feature: p_value_dict[feature][0]))