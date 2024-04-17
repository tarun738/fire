import numpy as np
import argparse
import json
import os
import cv2
import time
from multiprocessing import Process
from pathlib import Path
import trimesh as tm
import sys
import torch
import pickle
from ddfSampling import sampleDDFsFromMeshOnSphere, sampleMissingDDFsFromMeshOnSphere
from multiprocessing import Pool
from mesh_utils import loadAndNormalizeMesh, rot, normalizePC

def meshToColorSDF(tupFiles):
    outPath = tupFiles[0]
    mesh = tupFiles[1]
    print(mesh)
    numSamples=tupFiles[2]
    numDirs=tupFiles[3]
    samplingMode=tupFiles[4]
    inFolderSdf=tupFiles[5]
    tmmesh = loadAndNormalizeMesh(mesh,normalize=1.0/1.03)
    startTime = time.time()
    if samplingMode == 0:
        print("Sampling DDFs from Sphere")
        samples = sampleDDFsFromMeshOnSphere(tmmesh, numSamples=numSamples, numDirs=numDirs)
    else:
        print("Sampling DDF missing directions")
        samples = sampleMissingDDFsFromMeshOnSphere(tmmesh, numSamples=numSamples, numDirs=numDirs)
        np.savez_compressed(outPath,dsdfmiss=samples[:,:7].astype('float32'))
        return

    # loading from dsdf samples
    if not os.path.isfile(inFolderSdf):
        errFile = os.path.join(os.path.dirname(outPath),"notfound.pkl")
        if os.path.isfile(errFile):
            errFilePtr = open(errFile, "rb")
            errs = pickle.load(errFilePtr)
            errFilePtr.close()
        else:
            errs = []
        errs.append([inFolderSdf])
        errFilePtr = open(errFile, "wb")
        pickle.dump(errs, errFilePtr)
        errFilePtr.close()
        return
    
    sdfNpz = np.load(inFolderSdf,allow_pickle=True)
    if 'sdf' in sdfNpz:
        sdfNpz = sdfNpz['sdf'][None][0]
    else:
        print("Found error in SDF file")
    sdfSamples = np.append(sdfNpz['pos'],sdfNpz['neg'],axis=0)

    pos = sdfSamples[sdfSamples[:,3]>0,:]
    neg = sdfSamples[sdfSamples[:,3]<=0,:]
    sdf = {}
    sdf['pos'] = pos.astype('float32')
    sdf['neg'] = neg.astype('float32')
    print("Pos samples: " + str(pos.shape))
    print("Neg samples: " + str(neg.shape))
    np.savez_compressed(outPath,dsdf=samples[:,:7].astype('float32'),sdf=sdf)

    return
def getSamplesName(path,general=False):
    expFolderPaths = os.path.split(path)
    expName="_"+os.path.basename(expFolderPaths[-2])+"_"+os.path.basename(expFolderPaths[-1])
    if general:
        return ""
    else:
        return expName
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Preprocess Meshes for DDF samples")
    arg_parser.add_argument(
        "--samples_directory",
        "-i",
        dest="sample_dir",
        required=True,
        help="Directory containing sdf samples",
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment",
        required=True,
        help="experiment folder",
    )

    arg_parser.add_argument(
        "--splits",
        "-s",
        dest="splits",
        required=True,
        help="split train/test",
    )
    arg_parser.add_argument(
        "--numSamples",
        "--nS",
        dest="numSamples",
        default="500000",
        help="Number of samples",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip already processed files",
    )
    arg_parser.add_argument(
        "--numDirs",
        "--nD",
        dest="numDirs",
        default="1",
        help="Number of directions",
    )
    arg_parser.add_argument(
        "--samplingMode",
        "--sMode",
        dest="samplingMode",
        default=0,
        help="0 sphere samples. 1 missing directions",
    )
    args = arg_parser.parse_args()
    numSamples = int(args.numSamples)
    numDirs = int(args.numDirs)
    samplingMode = int(args.samplingMode)
    skip = bool(args.skip)

    # load the json file to get the input files for the current experiment
    with open(args.experiment+"/specs.json",'r') as specsFile:
        contents = json.load(specsFile)
        try:
            generalPrePro=contents["generalpreprocessing"]
        except:
            generalPrePro=False
        shapeNetMeshSource=contents["ShapeNetMeshSource"]
        try:
            shapeNetImgSource=contents["ShapeNetImgSource"]
        except:
            shapeNetImgSource=''
        try:
            shapeNetSDFSource=contents["ShapeNetSDF"]
        except:
            shapeNetSDFSource=''


        if args.splits == "train" or args.splits == "Train":
            splitsFile = contents["TrainSplit"]
        else:
            splitsFile = contents["TestSplit"]
    expName=getSamplesName(args.experiment,generalPrePro)
    print(splitsFile)
    # get samples from the input directory and the source meshes and textures
    with open(splitsFile,'r') as split:
        splitsFile = json.load(split)
    procs = Pool(8)
    files_list = []
    sdfCount = 0
    samplefileCount =0 
    for dataset in splitsFile:
        for folder in splitsFile[dataset]:
            outFolder = os.path.join(args.sample_dir,"SdfSamples",dataset,folder)
            if not os.path.isdir(outFolder):
                os.makedirs(outFolder)
            if not (args.splits == "train" or args.splits == "Train"):
                splitsFile[dataset][folder] = splitsFile[dataset][folder][:200]
                print("Only considering the first 200")
            for file in splitsFile[dataset][folder]:
                if samplingMode == 2:
                    write_samples_file = os.path.join(outFolder,file+expName+"_misses.npz")
                    test_samples_file = os.path.join(outFolder,file+expName+".npz")
                else:
                    write_samples_file = os.path.join(outFolder,file+expName+".npz")
                inFolder = os.path.join(shapeNetMeshSource,folder,file,"models")
                inFolderSdf = os.path.join(shapeNetSDFSource,folder,file+".npz")
                if (os.path.isfile(inFolderSdf+"____rejected.npz")):
                    if os.path.isfile(write_samples_file):
                        os.remove(write_samples_file)
                    continue
                if os.path.isfile(inFolderSdf):
                    print("found sdf file")
                    sdfCount += 1
                
                if os.path.isfile(write_samples_file):
                    print("found samples file")
                    samplefileCount += 1
                mesh_file = os.path.join(inFolder,"model_normalized.obj")
                if not os.path.isfile(mesh_file):
                    mesh_file = os.path.join(inFolder,"model_normalized.ply")
                
                if not os.path.isfile(mesh_file):
                    errFile = os.path.join(outFolder,"notfound.pkl")
                    if os.path.isfile(errFile):
                        errFilePtr = open(errFile, "rb")
                        errs = pickle.load(errFilePtr)
                        errFilePtr.close()
                    else:
                        errs = []
                    errs.append([file])
                    errFilePtr = open(errFile, "wb")
                    pickle.dump(errs, errFilePtr)
                    errFilePtr.close()
                    continue

                if (not os.path.isfile(write_samples_file)) or (not skip):
                    current_args = [write_samples_file,mesh_file,numSamples,numDirs,samplingMode,inFolderSdf]
                    files_list.append(current_args)
    files_list = tuple(files_list)
    print(len(files_list))
    procs.map(meshToColorSDF,files_list)
    print("Error Files: ")
    errFile = os.path.join(outFolder,"notfound.pkl")
    if os.path.isfile(errFile):
        errFilePtr = open(errFile, "rb")
        errs = pickle.load(errFilePtr)
        errFilePtr.close()
        print(errs)
    else:
        print("No errors")
    print("SDF found: "+str(sdfCount))
    print("Samples found: "+str(samplefileCount))