import os
import sys
import bpy
import math
import mathutils
import random

def importModel(fpath):
	'''Import OBJ model (from fpath) into Blender. Set origin, position and uniformely resize'''

	bpy.ops.wm.read_factory_settings(use_empty=True)
	imported_obj = bpy.ops.import_scene.obj(filepath=fpath, split_mode='OFF', axis_up='Z')
	obj = bpy.context.selected_objects[0]

	#Set origin
	bpy.ops.object.origin_set()

	#Set position on origin
	obj.location = [0,0,0]

	#Uniform resize
	msize = max(obj.dimensions)
	obj.dimensions = [d/msize for d in obj.dimensions]
	
	bpy.ops.export_scene.obj(filepath=fpath)

	return obj
	
def processDirRecursively(modelDir):
	for root, dirs, files in os.walk(modelDir):
		for name in files:
			if name.endswith('.obj'):
				path  = os.path.join(root,name)
				print("Processing {}".format(path))
				importModel(path)
				
				
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
modelDir = argv[0]
processDirRecursively(modelDir)
