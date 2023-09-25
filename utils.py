import re
import pickle

from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from sklearn import preprocessing
from fkscore import fkscore
import numpy as np

'''
Interaction Dynamics utils
'''
# refined_sample = pickle.load(open('samples/interaction_dynamics_sample.pkl', mode='rb'))

# returns all comment authors that received a delta
def deltas(submission):
    delta_authors = []
    id_map = {}
    for comment in submission['comments']:
        if 'author' in comment.keys():
            id_map[comment['id']] = comment['author']
            if comment['author'] == 'DeltaBot':
                author = re.search(r"(?<=\/u\/).+(?=\.)", comment['body'])
                if author is not None:
                    delta_authors.append(author.group(0))
    return delta_authors

# sorts all comments in a submission by creation time
def sort_comments(submission):
    comments = []
    for comment in submission['comments']:
        if 'created' in comment.keys() and 'author' in comment.keys():
            comments.append(comment)
    return sorted(comments, key= lambda comment: comment['created'])

# counts the number of comments made by each challenger
def num_author_comments(submission):
    id_map = {}
    op = submission['author']
    for comment in submission['comments']:
        if 'author' in comment.keys():
            id_map[comment['id']] = comment['author']
    author_comment_map = defaultdict(int)
    for comment in submission['comments']:
        if 'author' in comment.keys():
            if comment['author'] not in ['DeltaBot', op]:
                author_comment_map[comment['author']] += 1
    return author_comment_map

# counts the length of back-and-forths between challenger and op
def op_challenger_chain_counts(submission):
    id_map = {}
    op = submission['author']
    for comment in submission['comments']:
        if 'author' in comment.keys():
            id_map[comment['id']] = comment['author']
    op_chal_chains = defaultdict(int)
    for comment in submission['comments']:
        if 'author' in comment.keys() and comment['replies']:
            for reply_id in comment['replies']['data']['children']:
                if reply_id in id_map.keys():
                    if id_map[reply_id] == op:
                        op_chal_chains[comment['author']] += 1
                    elif comment['author'] == op:
                        op_chal_chains[id_map[reply_id]] += 1
    return op_chal_chains

'''
Language Indicator utils
'''
stop_words = set(stopwords.words('english'))

# simple implementation of jaccard score and other measures for two lists
def jaccard(A, O):
    set1 = set(A)
    set2 = set(O)
    return len(set1.intersection(set2))/len(set1.union(set2))

def common_words(A, O):
    return len(set(A).intersection(set(O)))

def reply_fraction(A, O):
    set1 = set(A)
    set2 = set(O)
    return len(set1.intersection(set2))/len(set1)

def op_fraction(A, O):
    set1 = set(A)
    set2 = set(O)
    return len(set1.intersection(set2))/len(set2)

# tokenizing and cleaning of comments text, replacing quotes and links with special tokens
def tokenize(comment):
    comment = comment.replace('&gt;', 'QUOTE')
    comment = re.sub(r'http\S+', 'LINK', comment)
    comment = comment.replace('(LINK', 'LINK')
    tokens = word_tokenize(comment)
    return tokens
# identity function (just return)
def id(obj):
    return obj
# function to remove stop words from tokenized
def remove_stop_words(tokens):
    return [word for word in tokens if word.lower() not in stop_words]
# function to only retrieve the stopw ords from tokenized
def only_stop_words(tokens):
    return [word for word in tokens if word.lower() in stop_words]
# separate a root-path unit into root reply and full path
def separate(rooted_path_unit):
    return rooted_path_unit['comments'][0]['body'], '\n\n'.join([comment['body'] for comment in rooted_path_unit['comments']])

interplay_features = ['# common ', 'jaccard ', 'op frac ', 'reply frac ']
interplay_bases = ['all', 'content', 'stop']
interplay_cols = []
for feature in interplay_features:
    for base in interplay_bases:
        interplay_cols.append(feature + base)
INTERPLAY_COLS = interplay_cols

# load VAD and concreteness csv as pd, keep the mean columns for each word
CONCRETENESS = pd.read_csv('docs/concreteness.csv')[['Word', 'Rating.Mean']]
VAD = pd.read_csv('docs/VAD.csv')[['Word', 'V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]

ARG_COLS = ['# indefinite articles', '# definite articles', '# 1st person pronouns', '# 1st person plural pronouns', 
            '# 2nd person pronouns', '# of links', '# of quotes', '# questions',
            'V', 'A', 'D', 'C',
            '# of sentences', '# of paragraphs', 'Flesch-Kincaid Readability']

def word_count(tokens):
    return len([w for w in tokens if w.isalnum()])

# return info for word category-based features in tokenized
def word_category_info(tokens): 
    quote_link_counter = Counter(tokens)
    num_links = quote_link_counter['LINK']
    num_quotes = quote_link_counter['QUOTE']
    lowered = [token.lower() for token in tokens]
    counter = Counter(lowered)
    features_map = {'# indefinite articles':['a', 'an'], '# definite articles': ['the'], '# 1st person pronouns': ['i', 'me', 'my', 'mine', 'myself'], 
                    '# 1st person plural pronouns': ['us', 'we', 'ours', 'ourselves', 'our'],
                    '# 2nd person pronouns': ['you', 'your', 'yours', 'yourself', 'yourselves'], '# questions': ['?']}

    count_map = {feature:sum([counter[token] for token in value]) for feature, value in features_map.items()}
    count_map['# of links'] = num_links
    count_map['# of quotes'] = num_quotes
    return count_map

# returns average V,A,D,C for content words in tokenized passage
def word_score_info(tokens):
    results = {feature:0.0 for feature in ['V', 'A', 'D','C']}
    tokens = remove_stop_words(tokens)
    lowered_tokens = [token.lower() for token in tokens]
    new_length = word_count(lowered_tokens)
    for token in lowered_tokens:
        if token in VAD['Word'].values:
            results['V'] += VAD.loc[VAD['Word'] == token, 'V.Mean.Sum'].item()
            results['A'] += VAD.loc[VAD['Word'] == token, 'A.Mean.Sum'].item()
            results['D'] += VAD.loc[VAD['Word'] == token, 'D.Mean.Sum'].item()
        if token in CONCRETENESS['Word'].values:
            results['C'] += CONCRETENESS.loc[CONCRETENESS['Word'] == token, 'Rating.Mean'].item()
    return {key:value/new_length for key, value in results.items()}

def num_sentences(text):
    return len(sent_tokenize(text))

def num_paragraphs(text):
    return len(text.split('\n\n'))

def flesch_kincaid_score(text):
    return fkscore(text).score['readability']

# entire text information for comments
def entire_text_features(text):
    results = {}
    results['# of sentences'] = num_sentences(text)
    results['# of paragraphs'] = num_paragraphs(text)
    results['Flesch-Kincaid Readability'] = flesch_kincaid_score(text)
    return results


def quarters(text):
    tokens = tokenize(text)
    return [list(array) for array in np.array_split(tokens, 4)]


# Persuaion utils

# CMV posts insert a footer at the end of the post, remove it and add title to the body of post
def remove_footer(submission):
    lines = [line for line in submission['selftext'].splitlines() 
             if not line.lstrip().startswith('&gt;') and not line.lstrip().startswith('_____')
             and "edit" not in " ".join(line.lower().split())
            ]
    return submission['title'] + "\n\n" + "\n".join(lines)

def remove_CMV(text):
    return text.replace('CMV:', '', 1)
