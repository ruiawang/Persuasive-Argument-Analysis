import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils


train_sample = pickle.load(open('samples/train_sample.pkl', mode='rb'))

# refinement critieria was submissions with >= 10 replies and 1 reply from OP
refined_sample = []
for submission in train_sample:
    comment_authors = []
    for comment in submission['comments']:
        if 'author' in comment.keys():
            comment_authors.append(comment['author'])
    if submission['num_comments'] >= 10 and submission['author'] in comment_authors:
        refined_sample.append(submission)
# print(len(refined_sample)) 411
pickle.dump(refined_sample, open('interaction_dynamics_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)

# refined_sample = pickle.load(open('samples/interaction_dynamics_sample.pkl', mode='rb'))

# counting delta ratio by challenger entry rank
delta_counts_by_challenger_rank = [0 for _ in range(10)]
num_challenger_ranks = [0 for _ in range(10)]

for submission in refined_sample:
    ordered_comments = utils.sort_comments(submission)
    ordered_authors = []
    for comment in ordered_comments:
        if len(ordered_authors) < 10 and comment['author'] not in ordered_authors and comment['author'] not in [submission['author'], 'DeltaBot']:
            ordered_authors.append(comment['author'])
    for i, author in enumerate(ordered_authors):
        num_challenger_ranks[i] += 1
        if author in utils.deltas(submission):
            delta_counts_by_challenger_rank[i] += 1

delta_ratios_by_challenger_rank = [delta_counts_by_challenger_rank[i]/num_challenger_ranks[i] for i in range(10)]

# plotting
deltas_by_challenger_df = pd.DataFrame(zip(range(1,11), delta_ratios_by_challenger_rank), columns=['challenger rank', 'delta percentage'])
fig, ax = plt.subplots()
ax.set_xlim(0,10.5)
ax.set_ylim(0,0.05)
sns.lineplot(deltas_by_challenger_df, ax=ax, x='challenger rank', y='delta percentage', marker='o', errorbar='se')
plt.show()



# counting delta ratios by number of comments a challenger makes
delta_counts_by_num_comments = [0 for _ in range(9)]
num_comment_lengths = [0 for _ in range(9)]

for submission in refined_sample:
    delta_authors = utils.deltas(submission)
    author_interactions = utils.num_author_comments(submission)
    for author in author_interactions.keys():
        if author_interactions[author] >= 9:
            num_comment_lengths[-1] += 1
            if author in delta_authors:
                delta_counts_by_num_comments[-1] += 1
        else:
            num_comment_lengths[author_interactions[author]-1] += 1
            if author in delta_authors:
                delta_counts_by_num_comments[author_interactions[author]-1] += 1

delta_ratios_by_num_comments = [delta_counts_by_num_comments[i]/num_comment_lengths[i] for i in range(9)]

# plotting
delt_ratios_by_num_comments_df = pd.DataFrame(zip(['1','2','3','4','5','6','7','8','9+'], delta_ratios_by_num_comments), columns=['number of comments made', 'delta percentage'])
fig, ax = plt.subplots()
sns.lineplot(delt_ratios_by_num_comments_df, ax=ax, x='number of comments made', y='delta percentage', marker='o', errorbar='se')
plt.show()



# counting delta ratio length of back-and-forths between ops and challenger
delta_counts_by_chain_length = [0 for _ in range(6)]
num_chain_lengths = [0 for _ in range(6)]

for submission in refined_sample:
    delta_authors = utils.deltas(submission)
    author_interactions = utils.op_challenger_chain_counts(submission)
    for author in author_interactions.keys():
        if author_interactions[author] >= 6:
            num_chain_lengths[-1] += 1
            if author in delta_authors:
                delta_counts_by_chain_length[-1] += 1
        else:
            num_chain_lengths[author_interactions[author]-1] += 1
            if author in delta_authors:
                delta_counts_by_chain_length[author_interactions[author]-1] += 1
delta_ratios_by_chain_length = [delta_counts_by_chain_length[i]/num_chain_lengths[i] for i in range(6)]

# plotting
delta_ratios_by_chain_length_df = pd.DataFrame(zip(['1','2','3','4','5','6+'], delta_ratios_by_chain_length), columns=['# replies in back and forth', 'delta percentage'])
fig, ax = plt.subplots()
sns.lineplot(delta_ratios_by_chain_length_df, ax=ax, x='# replies in back and forth', y='delta percentage', marker='o', errorbar='se')
plt.show()