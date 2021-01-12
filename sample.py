import pandas as pd

df = pd.read_csv('csv_files/Germany.csv')

ds = df.sample(frac=1)

dn = ds.head(1000)

print(dn)
dn.to_csv(r'final_data/germany.csv', index=False, header=True)