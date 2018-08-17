import os
import subprocess

def unzip(synid, cname):

	os.mkdir(cname)

	#unzip folder
	subprocess.run(['unzip','ShapeNetCore.v2.zip', 'ShapeNetCore.v2/{}/*/models/*.obj'.format(synid),'-d','tmp/'])
	
	#walk through tmp and get obj files to root
	for root, dirs, files in os.walk('tmp'):
		for f in files:
			modelid = root.split('/')[-2]
			os.rename(os.path.join(root,f), '{}/{}.obj'.format(cname, modelid))
			
	#remove tmp folder
	subprocess.run(['rm','-rf', 'tmp'])


#unzip('03790512', 'motorbike')
#unzip('02924116', 'bus')
#unzip('03710193', 'mailbox')
#unzip('02747177', 'bin')

#unzip('03119396', 'coupe')
#unzip('03100240', 'convertible')
#unzip('04166281', 'sedan')
#unzip('02958343', 'car')



	


