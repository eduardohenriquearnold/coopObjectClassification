import os
import sys
import bpy
import math
import mathutils
import random

cameras = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
random.seed(100) #Reproducible results for occlusion box placement

def setOcclusion(size):
    '''Set occlusion model with corresponding size and position'''
    obj = bpy.data.objects['Occlusion']
    obj.dimensions = 3*[size]

    #Set random position
    r = 0.6
    ang = random.uniform(-math.pi/2, math.pi/2)
    obj.location = [r*math.cos(ang), r*math.sin(ang), 0]

def importModel(fpath):
    '''Import OBJ model (from fpath) into Blender. Set origin, position and uniformely resize'''

    imported_obj = bpy.ops.import_scene.obj(filepath=fpath, split_mode='OFF')
    obj = bpy.context.selected_objects[0]

    #Set origin
    #bpy.ops.object.origin_set()
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    #Set position on origin
    obj.location = [0,0,0]

    #Random rotation
    if globals()['randomRot'] == 1:
        obj.rotation_euler[2] = random.uniform(-math.pi, math.pi)

    #Uniform resize
    msize = max(obj.dimensions)
    obj.dimensions = [d/msize for d in obj.dimensions]

    return obj

def render(outpath):
    '''Render scene to file (given by outpath) '''
    bpy.context.scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)

def generateView(fpath, dst):
    '''Renders all cameras images for model given in fpath'''

    #Import model
    obj = importModel(fpath)

    #Incorporate occlusion model
    setOcclusion(globals()['occSize'])

    #Generate views
    for i, cam in enumerate(cameras):
        bpy.context.scene.camera = cam
        fname = os.path.basename(fpath)[:-4]
        render(dst+'/'+fname+'-{}.png'.format(i))

    #Delete model
    bpy.data.objects.remove(obj, True)

    #Purge unused data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.scenes:
        if block.users == 0:
            bpy.data.scenes.remove(block)


def processDir(modelDir):
    modelPaths = [modelDir+f for f in os.listdir(modelDir) if f.endswith('.obj')]
    list(map(generateView, modelPaths))

def processDirRecursively(modelDir, dstDir):
    for root, dirs, files in os.walk(modelDir):
        for name in files:
            if name.endswith('.obj'):
                path  = os.path.join(root,name)
                print("Processing {}".format(path))
                dpath = os.path.join(dstDir, os.path.relpath(root, modelDir))
                #print(dpath)
                generateView(path, dpath)


argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
modelDir = argv[0]
occSize = float(argv[1])
randomRot = int(argv[2])
dstDir = argv[3]
processDirRecursively(modelDir, dstDir)
