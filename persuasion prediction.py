import pickle
import utils
import config
import openai

API_KEY = config.API_KEY
openai.api_key = API_KEY
op_train_sample = pickle.load(open('samples/op_train_sample.pkl', mode='rb'))
op_holdout_sample = pickle.load(open('samples/op_holdout_sample.pkl', mode='rb'))
responses = []
delta_results = []
'''
for submission in op_holdout_sample:
    current_opinion = utils.remove_CMV(utils.remove_footer(submission))
    prompt = f"""Please predict if the user who wrote the following opinion had their opinion changed.\n\n{current_opinion}\n\nPrediction:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': 'You are trying to predict whether a user will change their opinion about a topic.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        max_tokens=1024)
    responses.append(response.choices[0].message)
    delta_results.append(submission['delta_label'])
'''
for i in range(400,500):
    submission = op_train_sample[i]
    current_opinion = utils.remove_CMV(utils.remove_footer(submission))
    prompt = f"""Please predict if the user who wrote the following opinion had their opinion changed.\n\n{current_opinion}\n\nPrediction:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': 'You are trying to predict whether a user will change their opinion about a topic.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        max_tokens=1024)
    responses.append(response.choices[0].message)
    delta_results.append(submission['delta_label'])
pickle.dump(delta_results, open('delta results.pkl', mode='wb'))
pickle.dump(responses, open('gpt-3.5 responses.pkl', mode='wb'))