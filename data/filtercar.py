import pandas as pd
import os

meta = pd.read_csv('car/02958343.csv')
count = 0
for fname in os.listdir('car'):
	if not fname.endswith('.obj'):
		continue

	modelId = '3dw.'+fname[:-4]
	data = meta.loc[meta['fullId'] == modelId]
	lemma = data['wnlemmas'].values[0]
	
	if 'ambulance' in lemma:
		os.rename('car/'+fname, 'amb/'+fname)
	
	cls = ['sedan', 'convertible', 'coupe']
	if not any([c in lemma for c in cls]):
		count += 1
		os.remove('car/'+fname)
		
		
print(count)
		
	
	
