##########################################################
# Script for running an IDS agent from the command line #
##########################################################

import os
import argparse
import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


from ids_env.common.config_workspace import config_device, config_seed
from ids_env.common.config_ids_env import make_testing_env, make_training_env
from ids_env.common.config_agent import Agent
from ids_env.common.utils import calcul_rates, print_stats

if __name__=='__main__':

    ####----Parameters----####

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, default="KDD", help="Dataset to use: 'KDD' or 'AWID'.")
    parser.add_argument("-m", "--model", required=True, default="DQN", help="RL algorithm: 'DQN', 'PPO', 'A2C', 'TRPO', 'QRDQN.")
    parser.add_argument("-l", "--layers", required=True, default=1, type=int, help="Number of hidden layers in the policy.")
    parser.add_argument("-u", "--units", required=True, default=64, type=int, help="Number of units in each layer of the policy. If -1, then the network is composed with layers of increasing size.")
    parser.add_argument("-b", "--binary", default=1, type=int, help="Binary classification (True) or multi-class (False)")

    args = parser.parse_args()

    dataset= args.data
    model = args.model
    hidden_layers = args.layers
    nb_units = args.units if args.units>0 else "custom"
    binary=args.binary

     # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Load parameters
    log_dir = "../logs"
    load_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    if not os.path.exists(load_dir):
        raise(ValueError("The wanted model doesn't exist yet: try with other hyperparameter values"))

    ####----Device----######
    
    device = config_device(device_name)

    ####----Seed----#####

    config_seed(seed)

    ####----Environment----####

    testing_env = make_testing_env(dataset, binary=binary)() 
    training_env = make_training_env(dataset, binary=binary)()
    obs_shape = testing_env.reset().shape

    ####----Agent----####

    agent = Agent(testing_env, obs_shape, hidden_layers=hidden_layers, nb_units=nb_units, model=model, device=device, seed=seed)

    #agent_name='best_model_f1_bin.zip'
    if binary:
        agent_name='last_model_binary.zip'
    else:
        agent_name='last_model.zip'
    agent.load(os.path.join(load_dir, agent_name))

    ####----Test----####

    test_set=np.array(testing_env.X, dtype='float32')
    train_set=np.array(training_env.X, dtype='float32')
    dict_attack=dict((testing_env.unwrapped.attack_types[i], i) for i in range(len(testing_env.unwrapped.attack_types)))
    test_labels=testing_env.y.replace(dict_attack)
    train_labels=training_env.y.replace(dict_attack)
    if binary:
        test_labels=np.sign(test_labels)

    actions = agent.model.predict(train_set)[0]

    fpr, fnr = calcul_rates(train_labels, actions)
    f1_avg = f1_score(train_labels, actions, average='weighted')
    print('\nPerformance of '+ model + ' with ' + str(hidden_layers) + ' layers and ' + str(nb_units) + ' units in each layer on the ' + dataset + ' dataset (train set):')
    print('\nF1 score (avg): '+str(f1_avg))
    print('False positive rate: '+str(fpr))
    print('False negative rate: '+str(fnr))
    print('\nDetailed information: ')
    if binary:
        print_stats(['Normal', 'Attack'], train_labels, actions)
    else:
        print_stats(dict_attack, train_labels, actions)
    
    actions = agent.model.predict(test_set)[0]

    fpr, fnr = calcul_rates(test_labels, actions)
    f1_avg = f1_score(test_labels, actions, average='weighted')
    print('\nPerformance of '+ model + ' with ' + str(hidden_layers) + ' layers and ' + str(nb_units) + ' units in each layer on the ' + dataset + ' dataset (test set):')
    print('\nF1 score (avg): '+str(f1_avg))
    print('False positive rate: '+str(fpr))
    print('False negative rate: '+str(fnr))
    print('\nDetailed information: ')
    if binary:
        print_stats(['Normal', 'Attack'], test_labels, actions)
    else:
        print_stats(dict_attack, test_labels, actions)






