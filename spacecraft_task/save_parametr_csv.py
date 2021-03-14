import pandas as pd
A = 1
parametrs_dict = {
                    'eps': [0.1, 0.1, 0.1, 0.01, 0.05],
                    'c' : [520, 520, 520, 840, 900],
                    'A' : [0, 100, 0, 0, 0],
                    'lambdas_1_x' : [0, 20, 0, 0, 0],
                    'lambdas_2_x' : [0, 0, 20, 0, 0],
                    'lambdas_3_x' : [0, 0, 0, 10.5, 0],
                    'lambdas_4_x' : [0, 0, 0, 0, 119]
                }
df = pd.DataFrame(parametrs_dict)
df.to_csv('parametrs.csv')