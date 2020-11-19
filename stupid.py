import pandas as pd 
import os 

df = pd.read_csv('public-test.csv')

dir = os.listdir('../public-test_new/')
mp = {}
for x in dir:
    mp[x[x.find('_') + 1:-6]] = x

au_1 = []
au_2 = []
for a, b in df.iterrows():
    # print(mp[b['audio_1'][:-5]])
    # break
    au_1.append('public-test_new/' + mp[b['audio_1'][:-5]])
    au_2.append('public-test_new/' + mp[b['audio_2'][:-5]])

df['loc_1'] = au_1
df['loc_2'] = au_2 

df.to_csv('new_public-test.csv', index = False) 