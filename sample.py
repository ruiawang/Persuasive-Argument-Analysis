import bz2
import json
import random
import pickle

# dataset statistics from Table 1 of the paper.
TRAIN_LEN = 18363
HOLDOUT_LEN = 2263

# setting a seed so the sampling is reproducible.
random.seed(0)

# We will pursue a random sample of 500 training examples
train_sample_lines = random.sample(range(TRAIN_LEN), 500)

# similarly, we will scale down our holdout. 500/18363 * 2263 yields around 62 samples.
holdout_sample_lines = random.sample(range(HOLDOUT_LEN), 62)

# for ease of manipualtion, we will hold the samples as lists of dictionaries
train_sample = []
holdout_sample = []

with bz2.open('all/train_period_data.jsonlist.bz2', mode='rt') as train:
    for i, line in enumerate(train):
        if i in train_sample_lines:
            submission = json.loads(line)
            train_sample.append(submission)

with bz2.open('all/heldout_period_data.jsonlist.bz2', mode='rt') as holdout:
    for i, line in enumerate(holdout):
        if i in holdout_sample_lines:
            submission = json.loads(line)
            holdout_sample.append(submission)


pickle.dump(train_sample, open('train_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(holdout_sample, open('holdout_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)


PAIR_TRAIN_LEN = 3456
PAIR_HOLDOUT_LEN = 807

# We will pursue a random sample of 500 training examples
pair_train_sample_lines = random.sample(range(PAIR_TRAIN_LEN), 500)

# similarly, we will scale down our holdout. 500/3456 * 807 yields around 117 samples.
pair_holdout_sample_lines = random.sample(range(PAIR_HOLDOUT_LEN), 117)

pair_train_sample = []
pair_holdout_sample = []
with bz2.open('pair_task/train_pair_data.jsonlist.bz2', mode='rt') as pair_train:
    for i, line in enumerate(pair_train):
        if i in pair_train_sample_lines:
            submission = json.loads(line)
            pair_train_sample.append(submission)
with bz2.open('pair_task/heldout_pair_data.jsonlist.bz2', mode='rt') as pair_holdout:
    for i, line in enumerate(pair_holdout):
        if i in pair_holdout_sample_lines:
            submission = json.loads(line)
            pair_holdout_sample.append(submission)

pickle.dump(pair_train_sample, open('pair_train_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(pair_holdout_sample, open('pair_holdout_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)


OP_TRAIN_LEN = 10743
OP_HOLDOUT_LEN = 1529

# We will pursue a random sample of 500 training examples
op_train_sample_lines = random.sample(range(OP_TRAIN_LEN), 500)

# similarly, we will scale down our holdout. 500/10743 * 1529 yields around 71 samples.
op_holdout_sample_lines = random.sample(range(OP_HOLDOUT_LEN), 71)

op_train_sample = []
op_holdout_sample = []
with bz2.open('op_task/train_op_data.jsonlist.bz2', mode='rt') as op_train:
    for i, line in enumerate(op_train):
        if i in op_train_sample_lines:
            submission = json.loads(line)
            op_train_sample.append(submission)
with bz2.open('op_task/heldout_op_data.jsonlist.bz2', mode='rt') as op_holdout:
    for i, line in enumerate(op_holdout):
        if i in op_holdout_sample_lines:
            submission = json.loads(line)
            op_holdout_sample.append(submission)

pickle.dump(op_train_sample, open('op_train_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(op_holdout_sample, open('op_holdout_sample.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)