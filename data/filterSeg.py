import pandas as pd
import subprocess

meta = pd.read_csv('metadata_seg.csv')

#ignore missing wnlemmas
meta.dropna(subset=['category'], inplace=True)

#p = meta.loc[meta['category'].str.contains('Dog|Horse|Cat')]
#p = meta.loc[meta['wnlemmas'] == 'truck']
p = meta.loc[meta['category'].str.contains('Truck')]


ids = [i[4:] for i in p['fullId'].values]
print(ids)

fpaths = ['models/{}.obj'.format(i) for i in ids]
call = ['unzip', 'ShapeNet-Seg.zip'] + fpaths + ['-d', 'truck/']
subprocess.run(call)
	
