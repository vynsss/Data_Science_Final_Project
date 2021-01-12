import pandas as pd

dfg = pd.read_csv('final_data/germany.csv')
dfc = pd.read_csv('final_data/china.csv')
dfs = pd.read_csv('final_data/spain.csv')
dfi = pd.read_csv('final_data/italy.csv')
dff = pd.read_csv('final_data/france.csv')

frames = [dfg, dfc, dfs, dfi, dff]
result = pd.concat(frames)

print(result)
# result.to_csv(r'final_data/data.csv', index=False, header=True)



