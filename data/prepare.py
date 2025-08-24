import pandas as pd

IMB_FACT = 1

df = pd.read_json('data/allpainters.json').T
df['nbmuseum'] = df['nbmuseum'].astype(int)
df['inmuseum'] = (df.nbmuseum > 0).astype(int)

PE = df.loc[df['inmuseum'] == 1, ['desc', 'inmuseum']]
PE = PE.drop_duplicates()
PN = df.loc[df['inmuseum'] == 0, ['desc', 'inmuseum']]
PN = PN.drop_duplicates()
PN = PN.sample(int(len(PE)*IMB_FACT), random_state=42)
df = pd.concat([PE, PN])
df = df.sample(frac=1, random_state=42)

print(df.duplicated().sum(), "duplicated entries")

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train.to_csv('data/train.csv', index=True)
test.to_csv('data/test.csv', index=True)

