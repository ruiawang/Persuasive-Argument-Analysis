import pickle
import utils
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

concreteness = utils.CONCRETENESS
vad = utils.VAD

pair_train_sample = pickle.load(open('samples/pair_train_sample.pkl', mode='rb'))
# test_sub = pair_train_sample[0]

'''
Interplay features
'''
interplay_bases = [utils.id, utils.remove_stop_words, utils.only_stop_words]
interplay_features = [utils.common_words, utils.jaccard, utils.reply_fraction, utils.op_fraction]
interplay_cols = utils.INTERPLAY_COLS


pos_root_data = {col:[] for col in interplay_cols}
neg_root_data = {col:[] for col in interplay_cols}

pos_full_data = {col:[] for col in interplay_cols}
neg_full_data = {col:[] for col in interplay_cols}


for submission in pair_train_sample:
    op_text = utils.tokenize(submission['op_text'])

    pos, neg = submission['positive'], submission['negative']
    pos_root, pos_full = utils.separate(pos)
    neg_root, neg_full = utils.separate(neg)


    pos_root = utils.tokenize(pos_root)
    pos_full = utils.tokenize(pos_full)
    neg_root = utils.tokenize(neg_root)
    neg_full = utils.tokenize(neg_full)

    for i, feature in enumerate(interplay_features):
        for j, base in enumerate(interplay_bases):
            curr_feature = interplay_cols[i*len(interplay_bases)+j]
            
            pos_root_data[curr_feature].append(feature(base(pos_root), base(op_text)))
            pos_full_data[curr_feature].append(feature(base(pos_full), base(op_text)))
            neg_root_data[curr_feature].append(feature(base(neg_root), base(op_text)))
            neg_full_data[curr_feature].append(feature(base(neg_full), base(op_text)))

pos_root_df = pd.DataFrame(pos_root_data)
pos_full_df = pd.DataFrame(pos_full_data)
neg_root_df = pd.DataFrame(neg_root_data)
neg_full_df = pd.DataFrame(neg_full_data)

pd.to_pickle(pos_root_df, 'pos_root_int_df')
pd.to_pickle(pos_full_df, 'pos_full_int_df')
pd.to_pickle(neg_root_df, 'neg_root_int_df')
pd.to_pickle(neg_root_df, 'neg_full_int_df')

root_p_values = {}
full_p_values = {}

alpha = 0.05

for col in interplay_cols:
    root_t_statistic, root_p_value = stats.ttest_rel(pos_root_df[col], neg_root_df[col])
    full_t_statistic, full_p_value = stats.ttest_rel(pos_full_df[col], neg_full_df[col])
    
    mean_pos_root = pos_root_df[col].mean()
    mean_neg_root = neg_root_df[col].mean()
    mean_pos_full = pos_full_df[col].mean()
    mean_neg_full = neg_full_df[col].mean()

    if mean_pos_root > mean_neg_root:
        root_direction = 1
    elif mean_pos_root < mean_neg_root:
        root_direction = -1
    else:
        root_direction = 0

    if mean_pos_full > mean_neg_full:
        full_direction = 1
    elif mean_pos_full < mean_neg_full:
        full_direction = -1
    else:
        full_direction = 0

    root_p_values[col] = [root_p_value, root_direction]
    full_p_values[col] = [full_p_value, full_direction]

adjusted_alpha = alpha/len(interplay_cols)

pickle.dump(root_p_values, open('root_p_values.pkl', mode='wb'))
pickle.dump(full_p_values, open('full_p_values.pkl', mode='wb'))

root_significant_features = [feature for feature, p_value in root_p_values.items() if p_value[0] < adjusted_alpha]
full_significant_features = [feature for feature, p_value in full_p_values.items() if p_value[0] < adjusted_alpha]

# print(root_significant_features)
# print(full_significant_features)


'''
Argument-Only features

The original paper had a robust and wide selection of argument/word-based features.
I will choose to replicate some of them - these will be:

# words

from the Word category–based features:
# indefinite pronouns
# definite pronouns
# links
# 1st person pronouns (singular and plural)
# 2nd person pronouns
# quotations
# of questions

from the Word score–based features:
Valence, Arousal, Dominance, and Concreteness

Entire argument features:
# sentences, # paragraphs, Flesch-Kincaid Readability
'''

# # words analysis
pos_root_word_lengths = []
pos_full_word_lengths = []
neg_root_word_lengths = []
neg_full_word_lengths = []

for submission in pair_train_sample:
    pos, neg = submission['positive'], submission['negative']
    
    pos_root, pos_full = utils.separate(pos)
    neg_root, neg_full = utils.separate(neg)
    
    pos_root_tokens = utils.tokenize(pos_root)
    pos_full_tokens = utils.tokenize(pos_full)
    neg_root_tokens = utils.tokenize(neg_root)
    neg_full_tokens = utils.tokenize(neg_full)

    pos_root_word_lengths.append(utils.word_count(pos_root_tokens))
    pos_full_word_lengths.append(utils.word_count(pos_full_tokens))
    neg_root_word_lengths.append(utils.word_count(neg_root_tokens))
    neg_full_word_lengths.append(utils.word_count(neg_full_tokens))

pos_root_words_df = pd.DataFrame({'# words': pos_root_word_lengths})
pos_full_words_df = pd.DataFrame({'# words': pos_full_word_lengths})
neg_root_words_df = pd.DataFrame({'# words': neg_root_word_lengths})
neg_full_words_df = pd.DataFrame({'# words': neg_full_word_lengths})

pd.to_pickle(pos_root_words_df, 'pos_root_words_df')
pd.to_pickle(pos_full_words_df, 'pos_full_words_df')
pd.to_pickle(neg_root_words_df, 'neg_root_words_df')
pd.to_pickle(neg_full_words_df, 'neg_full_words_df')

t_statistic, root_words_p_value = stats.ttest_rel(pd.DataFrame(pos_root_word_lengths), pd.DataFrame(neg_root_word_lengths))
t_statistic, full_words_p_value = stats.ttest_rel(pd.DataFrame(pos_full_word_lengths), pd.DataFrame(neg_full_word_lengths))

mean_pos_root = np.average(pos_root_word_lengths)
mean_pos_full = np.average(pos_full_word_lengths)
mean_neg_root = np.average(neg_root_word_lengths)
mean_neg_full = np.average(neg_full_word_lengths)

if mean_pos_root > mean_neg_root:
    root_direction = 1
elif mean_pos_root < mean_neg_root:
    root_direction = -1
else:
    root_direction = 0

if mean_pos_full > mean_neg_full:
    full_direction = 1
elif mean_pos_full < mean_neg_full:
    full_direction = -1
else:
    full_direction = 0

root_words_p_value = [root_words_p_value, root_direction]
full_words_p_value = [full_words_p_value, full_direction]

# print(root_words_p_value, full_words_p_value) [[4.12198849e-12], 1] [[1.07780303e-16], 1]


# Other Argument Features
argument_cols = utils.ARG_COLS

pos_root_arg_data = {col:[] for col in argument_cols}
neg_root_arg_data = {col:[] for col in argument_cols}

pos_full_arg_data = {col:[] for col in argument_cols}
neg_full_arg_data = {col:[] for col in argument_cols}

word_category_features = ['# indefinite articles', '# definite articles', '# 1st person pronouns', '# 1st person plural pronouns', 
                          '# 2nd person pronouns', '# of links', '# of quotes', '# questions']
word_score_features = ['V', 'A', 'D', 'C']
entire_text_features = ['# of sentences', '# of paragraphs', 'Flesch-Kincaid Readability']
for submission in pair_train_sample:
    pos, neg = submission['positive'], submission['negative']
    pos_root, pos_full = utils.separate(pos)
    neg_root, neg_full = utils.separate(neg)

    pos_root_tokens = utils.tokenize(pos_root)
    pos_full_tokens = utils.tokenize(pos_full)
    neg_root_tokens = utils.tokenize(neg_root)
    neg_full_tokens = utils.tokenize(neg_full)

    pos_root_category = utils.word_category_info(pos_root_tokens)
    pos_full_category = utils.word_category_info(pos_full_tokens)
    neg_root_category = utils.word_category_info(neg_root_tokens)
    neg_full_category = utils.word_category_info(neg_full_tokens)
    
    for feature in word_category_features:
        pos_root_arg_data[feature].append(pos_root_category[feature])
        pos_full_arg_data[feature].append(pos_full_category[feature])
        neg_root_arg_data[feature].append(neg_root_category[feature])
        neg_full_arg_data[feature].append(neg_full_category[feature])
    
    pos_root_score = utils.word_score_info(pos_root_tokens)
    pos_full_score = utils.word_score_info(pos_full_tokens)
    neg_root_score = utils.word_score_info(neg_root_tokens)
    neg_full_score = utils.word_score_info(neg_full_tokens)
    
    for feature in word_score_features:
        pos_root_arg_data[feature].append(pos_root_score[feature])
        pos_full_arg_data[feature].append(pos_full_score[feature])
        neg_root_arg_data[feature].append(neg_root_score[feature])
        neg_full_arg_data[feature].append(neg_full_score[feature])

    pos_root_entire = utils.entire_text_features(pos_root)
    pos_full_entire = utils.entire_text_features(pos_full)
    neg_root_entire = utils.entire_text_features(neg_root)
    neg_full_entire = utils.entire_text_features(neg_full)
    
    for feature in entire_text_features:
        pos_root_arg_data[feature].append(pos_root_entire[feature])
        pos_full_arg_data[feature].append(pos_full_entire[feature])
        neg_root_arg_data[feature].append(neg_root_entire[feature])
        neg_full_arg_data[feature].append(neg_full_entire[feature])      
    

pos_root_arg_df = pd.DataFrame(pos_root_arg_data)
pos_full_arg_df = pd.DataFrame(pos_full_arg_data)
neg_root_arg_df = pd.DataFrame(neg_root_arg_data)
neg_full_arg_df = pd.DataFrame(neg_full_arg_data)

# print(pos_root_arg_df.head())
# print(pos_full_arg_df.head())
# print(neg_root_arg_df.head())
# print(neg_full_arg_df.head())

# pd.to_pickle(pos_root_arg_df, 'pos_root_arg_df')
# pd.to_pickle(pos_full_arg_df, 'pos_full_arg_df')
# pd.to_pickle(neg_root_arg_df, 'neg_root_arg_df')
# pd.to_pickle(neg_full_arg_df, 'neg_full_arg_df')

# pos_root_arg_df = pd.read_pickle('pos_root_arg_df')
# pos_full_arg_df = pd.read_pickle('pos_full_arg_df')
# neg_root_arg_df = pd.read_pickle('neg_root_arg_df')
# neg_full_arg_df = pd.read_pickle('neg_full_arg_df')

arg_root_p_values = {}
arg_full_p_values = {}

alpha = 0.05

for col in argument_cols:
    root_t_statistic, root_p_value = stats.ttest_rel(pos_root_arg_df[col], neg_root_arg_df[col])
    full_t_statistic, full_p_value = stats.ttest_rel(pos_full_arg_df[col], neg_full_arg_df[col])
    
    mean_pos_root = pos_root_arg_df[col].mean()
    mean_neg_root = neg_root_arg_df[col].mean()
    mean_pos_full = pos_full_arg_df[col].mean()
    mean_neg_full = neg_full_arg_df[col].mean()

    if mean_pos_root > mean_neg_root:
        root_direction = 1
    elif mean_pos_root < mean_neg_root:
        root_direction = -1
    else:
        root_direction = 0

    if mean_pos_full > mean_neg_full:
        full_direction = 1
    elif mean_pos_full < mean_neg_full:
        full_direction = -1
    else:
        full_direction = 0

    arg_root_p_values[col] = [root_p_value, root_direction]
    arg_full_p_values[col] = [full_p_value, full_direction]

adjusted_alpha = alpha/len(argument_cols)

# pickle.dump(arg_root_p_values, open('arg_root_p_values.pkl', mode='wb'))
# pickle.dump(arg_full_p_values, open('arg_full_p_values.pkl', mode='wb'))

root_significant_features = [feature for feature, p_value in arg_root_p_values.items() if p_value[0] < adjusted_alpha]
full_significant_features = [feature for feature, p_value in arg_full_p_values.items() if p_value[0] < adjusted_alpha]

# print(root_significant_features)
# print(full_significant_features)



'''
Quarter-by-Quarter Analysis
'''
word_score_features = ['V', 'A', 'D', 'C']
pos_quarters_data = [{feature:[] for feature in word_score_features} for i in range(4)]
neg_quarters_data = [{feature:[] for feature in word_score_features} for i in range(4)]

for submission in pair_train_sample:
    pos, neg = submission['positive'], submission['negative']
    pos_root, _ = utils.separate(pos)
    neg_root, _ = utils.separate(neg)

    pos_quarters = utils.quarters(pos_root)
    neg_quarters = utils.quarters(neg_root)
    for i in range(4):
        pos_quarter_scores = utils.word_score_info(pos_quarters[i])
        neg_quarter_scores = utils.word_score_info(neg_quarters[i])
        for feature in word_score_features:
            pos_quarters_data[i][feature].append(pos_quarter_scores[feature])
            neg_quarters_data[i][feature].append(neg_quarter_scores[feature])

pos_quarters_results = {feature:[] for feature in word_score_features}
neg_quarters_results = {feature:[] for feature in word_score_features}

for feature in word_score_features:
    for i in range(4):
        pos_quarters_results[feature].append(np.average(pos_quarters_data[i][feature]))
        neg_quarters_results[feature].append(np.average(neg_quarters_data[i][feature]))

# print(pos_quarters_results)
# print(neg_quarters_results)
# pickle.dump(pos_quarters_results, open('pos_quarters_results.pkl', 'wb'))
# pickle.dump(neg_quarters_results, open('neg_quarters_results.pkl', 'wb'))

# plotting
pos_quarters_results = {'V': [3.306997699177231, 3.183554925435482, 3.2251098089614643, 3.3324367587440875], 'A': [2.4181050182402806, 2.34284643034643, 2.36935104596941, 2.4743568760526777], 'D': [3.283346325593583, 3.145545636941952, 3.189554771607879, 3.297779558730065], 'C': [4.779526933338892, 4.636754283505107, 4.706297415347186, 4.842514326549288]}
neg_quarters_results = {'V': [3.3825888919248244, 3.2734072340222244, 3.320281487178333, 3.3796866170836894], 'A': [2.468079088927753, 2.4058344851240956, 2.451230392832782, 2.495786493071553], 'D': [3.339112548935306, 3.239760601713777, 3.276001354215834, 3.364692950442815], 'C': [4.776390897689387, 4.677317183206738, 4.718014084730121, 4.788618696947202]}

pos_quarters_results = pd.DataFrame(pos_quarters_results)
neg_quarters_results = pd.DataFrame(neg_quarters_results)

print(pos_quarters_results)
print(neg_quarters_results)

v_results = pd.DataFrame(zip(range(1,5), pos_quarters_results['V'], neg_quarters_results['V']), columns=['quarter', 'effective', 'not effective'])
a_results = pd.DataFrame(zip(range(1,5), pos_quarters_results['A'], neg_quarters_results['A']), columns=['quarter', 'effective', 'not effective'])
d_results = pd.DataFrame(zip(range(1,5), pos_quarters_results['D'], neg_quarters_results['D']), columns=['quarter', 'effective', 'not effective'])
c_results = pd.DataFrame(zip(range(1,5), pos_quarters_results['C'], neg_quarters_results['C']), columns=['quarter', 'effective', 'not effective'])
print(v_results)

v_results.plot(x='quarter', y=['effective', 'not effective'], color=['red', 'blue'], title='Valence', xticks=range(1,5), ylabel='Valence', style='o-')
a_results.plot(x='quarter', y=['effective', 'not effective'], color=['red', 'blue'], title='Arousal', xticks=range(1,5), ylabel='Arousal', style='o-')
d_results.plot(x='quarter', y=['effective', 'not effective'], color=['red', 'blue'], title='Dominance', xticks=range(1,5), ylabel='Dominance', style='o-')
c_results.plot(x='quarter', y=['effective', 'not effective'], color=['red', 'blue'], title='Concreteness', xticks=range(1,5), ylabel='Concreteness', style='o-')
plt.show()



pos_quarters_data = [[] for i in range(4)]
neg_quarters_data = [[] for i in range(4)]
for submission in pair_train_sample:
    op_text = submission['op_text']
    pos, neg = submission['positive'], submission['negative']
    _, pos_full = utils.separate(pos)
    _, neg_full = utils.separate(neg)
    
    op_quarters = utils.quarters(op_text)
    pos_quarters = utils.quarters(pos_full)
    neg_quarters = utils.quarters(neg_full)

    for i in range(4):
        pos_quarters_data[i].append(utils.reply_fraction(pos_quarters[i], op_quarters[i]))
        neg_quarters_data[i].append(utils.reply_fraction(neg_quarters[i], op_quarters[i]))

pos_quarters_results = [0 for _ in range(4)]
neg_quarters_results = [0 for _ in range(4)]
for i in range(4):
    pos_quarters_results[i] = np.average(pos_quarters_data[i])
    neg_quarters_results[i] = np.average(neg_quarters_data[i])
# print(pos_quarters_results) [0.27625485817502476, 0.2528198055168628, 0.23991167651276243, 0.22865056719050258]
# print(neg_quarters_results) [0.3009880800316708, 0.2882306043718262, 0.27058499953597853, 0.25364724667556726]

# plotting
results = pd.DataFrame(zip(range(1,5), [0.27625485817502476, 0.2528198055168628, 0.23991167651276243, 0.22865056719050258], 
                       [0.3009880800316708, 0.2882306043718262, 0.27058499953597853, 0.25364724667556726]), columns=['quarter', 'effective', 'not effective'])

results.plot(x='quarter', y=['effective', 'not effective'], color=['red', 'blue'], title='Reply Fraction similarity', xticks=range(1,5), ylabel='reply fraction', style='o-')
plt.show()
