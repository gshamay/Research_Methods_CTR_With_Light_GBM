# Adi Paz
# Neriya Mazzuz
# Gil Shamay 

import pandas as pd

d = {'C': [1, 2, 3], 'f': ['b', 'c', 'a']}
df1 = pd.DataFrame(data=d)
d = {'C': [2, 3, 4], 'f': ['a', 'd', 'e']}
df2 = pd.DataFrame(data=d)
d = {'C': [4, 5, 6], 'f': ['b', 'c', 'd']}
df3 = pd.DataFrame(data=d)

d = {'n': ['a', 'b', 'e', 'c', 'd']}
df = pd.DataFrame(data=d)

df['n'] = pd.Categorical(df['n'])
df['n'] = df.n.cat.codes

df1['f'] = pd.Categorical(df1['f']) #[1,2,3]
# df1.cat.rename_categories([22, 33, 11])

print(df1.info())
dfCatSeries = df1['f']

print(str(df1.max(axis = 1, skipna = True)))
df1['lala'] = [5,3,2]
df3 =  {'g':df1['C'],'k':df1['lala']}
df3 = pd.DataFrame(data=df3)