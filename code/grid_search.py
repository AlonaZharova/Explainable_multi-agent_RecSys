import json
from agents import Performance_Evaluation_Agent
from copy import deepcopy

shiftable_devices = {
    1: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    2: ['Washing Machine', 'Dishwasher'],
    3: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    4: ['Washing Machine (1)', 'Washing Machine (2)'],
    5: ['Tumble Dryer'], # , 'Washing Machine' --> consumes energy constantly; , 'Dishwasher' --> noise at 3am
    6: ['Washing Machine', 'Dishwasher'],
    7: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    8: ['Washing Machine'], # 'Dryer' --> consumes constantly
    9: ['Washer Dryer', 'Washing Machine', 'Dishwasher'],
    10: ['Washing Machine'] #'Dishwasher'
}

active_appliances = {
    1: deepcopy(shiftable_devices[1]) + ['Television Site', 'Computer Site'],
    2: deepcopy(shiftable_devices[2]) + ['Television', 'Microwave', 'Toaster', 'Hi-Fi', 'Kettle'],
    3: deepcopy(shiftable_devices[3]) + ['Toaster', 'Television', 'Microwave', 'Kettle'],
    4: deepcopy(shiftable_devices[4]) + ['Television Site', 'Kettle'], #'Microwave', 'Computer Site' --> consume energy constantly
    5: deepcopy(shiftable_devices[5]) + ['Television Site', 'Combination Microwave', 'Kettle', 'Toaster'], # 'Computer Site', --> consumes energy constantly
    6: deepcopy(shiftable_devices[6]) + ['MJY Computer', 'Kettle', 'Toaster'], #, 'PGM Computer', 'Television Site' 'Microwave' --> consume energy constantly
    7: deepcopy(shiftable_devices[7]) + ['Television Site', 'Toaster', 'Kettle'],
    8: deepcopy(shiftable_devices[8]) + ['Toaster', 'Kettle'], # 'Television Site', 'Computer' --> consume energy constantly
    9: deepcopy(shiftable_devices[9]) + ['Microwave', 'Kettle'], #'Television Site', 'Hi-Fi' --> consume energy constantly
    10: deepcopy(shiftable_devices[10]) + ['Magimix (Blender)', 'Microwave'] # 'Television Site' --> consume energy constantly
}

thresholds = {
    1: 0.15,
    2: 0.01,
    3: 0.01,
    4: 0.01,
    5: 0.025,
    6: 0.065,
    7: 0.01,
    8: 0.01, # washing machine over night
    9: 0.01,
    10: 0.01
}

DATA_PATH = '../data/'
EXPORT_PATH = '../export/'

household_id = 3

config = {'data': {'household': deepcopy(household_id)}}
config['user_input'] = {
    'shiftable_devices': deepcopy(shiftable_devices[config['data']['household']]),
    'active_appliances': deepcopy(active_appliances[config['data']['household']]),
    'threshold': deepcopy(thresholds[config['data']['household']])
}

out = []

# #%% Logit
evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="logit", load_data=True, weather_sel=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')

evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])

max_iters = [10, 100, 1000]
for i in range(0, len(max_iters)):
    scores, preds = evaluation.get_agent_scores(max_iter=max_iters[i])
    out = evaluation.model_type + "," + str(max_iters[i]) + str(scores)
    with open("Output.txt", "a") as f:
        f.write(out)

# #%% Random Forest
evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="random forest", load_data=True, weather_sel=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')

evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])

max_depth = [10, 20]
n_estimators = [100, 500, 1000]
max_features = ["auto", "sqrt", "log2"]

for i in range(0, len(max_depth)):
    for j in range(0, len(n_estimators)):
        for k in range(0, len(max_features)):
            scores, preds = evaluation.get_agent_scores(max_depth=max_depth[i], n_estimators=n_estimators[j], max_features=max_features[k])
            out = evaluation.model_type + "," + str(max_depth[i]) + str(n_estimators[j]) + str(max_features[k]) + str(scores)
            with open("Output.txt", "a") as f:
                f.write(out)

#%% Ada
evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="ada", load_data=True, weather_sel=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')

evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])

lr = [0.001, 0.01, 0.1, 1.0]
n_estimators = [50, 100, 500]

for i in range(0, len(lr)):
    for j in range(0, len(n_estimators)):
        scores, preds = evaluation.get_agent_scores(learning_rate=lr[i], n_estimators=n_estimators[j])
        out = evaluation.model_type + "," + str(lr[i]) + str(n_estimators[j]) + str(scores) + "\n"
        with open("Output.txt", "a") as f:
            f.write(out)

#%% KNN
evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="knn", load_data=True, weather_sel=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')

evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])

n_neighbors = [1, 5, 10]
leaf_size = [30, 60, 90]

for i in range(0, len(leaf_size)):
    for j in range(0, len(n_neighbors)):
        scores, preds = evaluation.get_agent_scores(leaf_size=leaf_size[i], n_neighbors=n_neighbors[j])
        out = evaluation.model_type + "," + str(leaf_size[i]) + str(n_neighbors[j]) + str(scores) + "\n"
        with open("Output.txt", "a") as f:
            f.write(out)

#%% XGBoost
#%% XGBoost
evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="xgboost", load_data=True, weather_sel=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')

evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])

learning_rate = [0.01, 0.1, 0.3, 1.0]
max_depth = [6, 12, 18]
lam = [0.0, 0.5, 1.0]

for i in range(0, len(learning_rate)):
    for j in range(0, len(max_depth)):
        for k in range(0, len(lam)):
            scores, preds = evaluation.get_agent_scores(learning_rate=learning_rate[i], max_depth=max_depth[j], reg_lambda=lam[k], reg_alpha=(1-lam[k]))
            out = evaluation.model_type + "," + str(learning_rate[i]) + str(max_depth[j]) + str(lam[k]) + str(scores) + "\n"
            with open("Output.txt", "a") as f:
                f.write(out)