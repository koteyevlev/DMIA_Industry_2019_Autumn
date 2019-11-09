import json
import sys

for gamma in [1.5]:
    for reg_lambda in [0.1]:
        for n_estimators in [300]:
            for subsample in [0.5]:
                for max_depth in [5]:
                    for colsample_bylevel in [0.4]:
                        for colsample_bytree in [0.9]:
                            for learning_rate in [0.04]:
                                for child_weight in [2]:
                                    for seed in range(1, 100):
                                        path = "xgboost_params_example.json"

                                        with open(path, 'r') as f:
                                        #    print(f.read())
                                            data = json.loads(f.read())
                                            data['tree_method'] = 'auto'
                                            data['seed'] = seed
                                            data['subsample'] = subsample
                                            data['min_child_weight'] = child_weight
                                            data['learning_rate'] = learning_rate
                                            data['colsample_bytree'] = colsample_bytree
                                            data['colsample_bylevel'] = colsample_bylevel
                                            data['max_depth'] = max_depth
                                            data['n_estimators'] = n_estimators
                                            data['reg_lambda'] = reg_lambda
                                            data['gamma'] = gamma
                                            print()
                                            print(data)
                                            
                                        with open('xgboost_params_example.json', 'w') as f:
                                            json.dump(data, f)
                                        #sys.argv = [data]
                                        exec(open("xgboost_params_checker.py").read())
