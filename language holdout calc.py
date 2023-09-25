import pickle
import pandas as pd
import utils

pair_holdout_sample = pickle.load(open('samples/pair_holdout_sample.pkl', mode='rb'))


interplay_features = utils.INTERPLAY_COLS
style_features = ['# words'] + utils.ARG_COLS
combined_features = interplay_features + style_features

interplay_bases = [utils.id, utils.remove_stop_words, utils.only_stop_words]
interplay_features = [utils.common_words, utils.jaccard, utils.reply_fraction, utils.op_fraction]
interplay_cols = utils.INTERPLAY_COLS


pos_root_int_data = {col:[] for col in interplay_cols}
neg_root_int_data = {col:[] for col in interplay_cols}

pos_full_int_data = {col:[] for col in interplay_cols}
neg_full_int_data = {col:[] for col in interplay_cols}


for submission in pair_holdout_sample:
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
            
            pos_root_int_data[curr_feature].append(feature(base(pos_root), base(op_text)))
            pos_full_int_data[curr_feature].append(feature(base(pos_full), base(op_text)))
            neg_root_int_data[curr_feature].append(feature(base(neg_root), base(op_text)))
            neg_full_int_data[curr_feature].append(feature(base(neg_full), base(op_text)))

pos_root_int_df = pd.DataFrame(pos_root_int_data)
pos_full_int_df = pd.DataFrame(pos_full_int_data)
neg_root_int_df = pd.DataFrame(neg_root_int_data)
neg_full_int_df = pd.DataFrame(neg_full_int_data)

pos_root_word_lengths = []
pos_full_word_lengths = []
neg_root_word_lengths = []
neg_full_word_lengths = []

for submission in pair_holdout_sample:
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

argument_cols = utils.ARG_COLS

pos_root_arg_data = {col:[] for col in argument_cols}
neg_root_arg_data = {col:[] for col in argument_cols}

pos_full_arg_data = {col:[] for col in argument_cols}
neg_full_arg_data = {col:[] for col in argument_cols}

word_category_features = ['# indefinite articles', '# definite articles', '# 1st person pronouns', '# 1st person plural pronouns', 
                          '# 2nd person pronouns', '# of links', '# of quotes', '# questions']
word_score_features = ['V', 'A', 'D', 'C']
entire_text_features = ['# of sentences', '# of paragraphs', 'Flesch-Kincaid Readability']
for submission in pair_holdout_sample:
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

pos_root_df = pd.concat(objs=[pos_root_int_df.reset_index(drop=True), pos_root_words_df.reset_index(drop=True), pos_root_arg_df.reset_index(drop=True)], axis=1)
pos_full_df = pd.concat(objs=[pos_full_int_df.reset_index(drop=True), pos_full_words_df.reset_index(drop=True), pos_full_arg_df.reset_index(drop=True)], axis=1)
neg_root_df = pd.concat(objs=[neg_root_int_df.reset_index(drop=True), neg_root_words_df.reset_index(drop=True), neg_root_arg_df.reset_index(drop=True)], axis=1)
neg_full_df = pd.concat(objs=[neg_full_int_df.reset_index(drop=True), neg_full_words_df.reset_index(drop=True), neg_full_arg_df.reset_index(drop=True)], axis=1)

pos_root_df['effective'] = pd.DataFrame(['effective' for _ in range(234)])
pos_full_df['effective'] = pd.DataFrame(['effective' for _ in range(234)])
neg_root_df['effective'] = pd.DataFrame(['not effective' for _ in range(234)])
neg_full_df['effective'] = pd.DataFrame(['not effective' for _ in range(234)])

root_df = pd.concat([pos_root_df,neg_root_df]).sort_index()
root_df = root_df.reset_index(drop=True)

full_df = pd.concat([pos_full_df,neg_full_df]).sort_index()
full_df = full_df.reset_index(drop=True)

print(root_df.head())
print(root_df.info())
print(full_df.head())
print(full_df.info())

pd.to_pickle(root_df, 'root_df')
pd.to_pickle(full_df, 'full_df')