import numpy as np
from sklearn.metrics import f1_score

from stable_baselines3.common.callbacks import EventCallback
import wandb

from ids_env.common.utils import calcul_rates



class CustomWandbCallback(EventCallback):
    '''
    Custom callback to log training data in wandb and save best model.
    '''
    def __init__(self, eval_freq, test_set, train_set, dict_attack, test_labels, train_labels, binary, nb_class, epsilon_range, model_name, verbose=0):
        '''
        Params:
        -------
            -training_env: gym.Env
                Agent's training environment.
            -eval_env: gym.Env
                Environment used for evaluation (testing set).
            -eval_freq: int
                Callback frequency.
            -save_dir: str
                Directory used to save best model.
        '''
        super().__init__(verbose=verbose)
        self.eval_freq=eval_freq
        self.test_set=test_set
        self.train_set = train_set
        self.dict_attack=dict_attack
        self.test_labels=test_labels
        self.train_labels=train_labels
        self.nb_class=nb_class
        self.epsilon_range=epsilon_range
        self.binary=binary
        self.model_name=model_name
    
    def _on_step(self) -> bool:

        if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            print('Logging metrics for epoch ' + str(self.n_calls // self.eval_freq))
            
            #### Training metrics ####

            train_actions = self.model.predict(self.train_set, deterministic=True)[0]
            fpr, fnr = calcul_rates(self.train_labels, train_actions)
            f1 = f1_score(self.train_labels, train_actions) if self.binary else f1_score(self.train_labels, train_actions, average='weighted')

            wandb.log({
                'training_metrics':{
                    "FPR":fpr,
                    "FNR":fnr,
                    "F1 score":f1
                    },
                'epoch':self.n_calls // self.eval_freq,
                })
            
            #### Clean testing metrics ####

            test_actions = self.model.predict(self.test_set, deterministic=True)[0]
            fpr, fnr = calcul_rates(self.test_labels, test_actions)
            f1 = f1_score(self.test_labels, test_actions) if self.binary else f1_score(self.test_labels, test_actions, average='weighted')

            wandb.log({
                'testing_metrics':{
                    "FPR":fpr,
                    "FNR":fnr,
                    "F1 score":f1
                    },
                'epoch':self.n_calls // self.eval_freq,
                })

        return True
