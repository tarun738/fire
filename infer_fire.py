# this code base has been adapted from deepSDF
import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import DDF
import DDF.workspace as ws
from image_utils import renderAndSaveImage, fire_model, renderAndSaveVideo, fitImagesLatent, reconstructAndSaveMesh, saveColorCodedGT
from preprocessData import getSamplesName
from reconstruct import reconstructLatent

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Fit to DDF model given preprocessed SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_name",
        default="Train",
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=1000,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--imgHeight",
        "--imH",
        dest="imgHeight",
        default=256,
        help="Rendered image height.",
    )
    arg_parser.add_argument(
        "--imgWidth",
        "--imW",
        dest="imgWidth",
        default=256,
        help="Rendered image height.",
    )
    arg_parser.add_argument(
        "--reconClass",
        "--rc",
        dest="reconClass",
        default="image",
        help="image, video, mesh, or point cloud",
    )
    arg_parser.add_argument(
        "--fitTo",
        "--ft",
        dest="fitTo",
        default="distances",
        help="distances, depths, or colors",
    )
    arg_parser.add_argument(
        "--thresh",
        "-t",
        dest="thresh",
        default=0.999,
        help="Rendered image height.",
    )
    arg_parser.add_argument(
        "--fitDebug",
        "--fd",
        dest="fitDebug",
        action="store_true",
        help="Stores in fit_<mode>_debug folder",
    )
    arg_parser.add_argument(
        "--fitDIST",
        dest="fitDIST",
        action="store_true",
        help="Fits to 512x512 choy images and stores in fitDIST folder",
    )
    arg_parser.add_argument(
        "--evaluateMetricsOnly",
        dest="evaluateMetricsOnly",
        action="store_true",
        help="Evaluate only metrics for fits",
    )
    arg_parser.add_argument(
        "--meshThresh",
        dest="meshThresh",
        default=1e-3,
        help="SDF threshold for extracting mesh",
    )
    arg_parser.add_argument(
        "--ablations",
        dest="ablations",
        action="store_true",
        help="Ablation experiments",
    )
    arg_parser.add_argument(
        "--rendersPaper",
        dest="rendersPaper",
        action="store_true",
        help="Render for images",
    )
    arg_parser.add_argument(
        "--feat3Drec",
        dest="feat3Drec",
        action="store_true",
        help="Also learn features for 3D reconstruction",
    )
    arg_parser.add_argument(
        "--reconOnlySDF",
        dest="reconOnlySDF",
        action="store_true",
        help="SDF 3D reconstruction",
    )
    arg_parser.add_argument(
        "--abmode",
        dest="abmode",
        default=None,
        help="ablation mode",
    )
    arg_parser.add_argument(
        "--DDFSDFWt",
        dest="DDFSDFWt",
        default=1e0,
        help="ddf sdf consistency weight",
    )
    arg_parser.add_argument(
        "--DDFSDFPow",
        dest="DDFSDFPow",
        default=1e0,
        help="ddf sdf consistency loss power",
    )
    arg_parser.add_argument(
        "--scaletype",
        dest="scaletype",
        default="",
        help="prif for unitsphere scale",
    )
    arg_parser.add_argument(
        "--surfacesamplespath",
        dest="surfacesamplespath",
        default='',
        help='path to surface samples for evaluation'
    )
    rot = 0

    DDF.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    DDF.configure_logging(args)
    thresh = float(args.thresh)
    imgHeight = int(args.imgHeight)
    imgWidth = int(args.imgWidth)
    iterations = int(args.iterations)
    reconClass = args.reconClass
    fitDIST = args.fitDIST
    fitDebug = args.fitDebug
    evaluateMetricsOnly = args.evaluateMetricsOnly
    fitTo = args.fitTo
    meshThresh = float(args.meshThresh)
    ablations = bool(args.ablations)
    rendersPaper = bool(args.rendersPaper)
    feat3Drec = bool(args.feat3Drec)
    reconOnlySDF = bool(args.reconOnlySDF)
    abmode = args.abmode
    DDFSDFWt = float(args.DDFSDFWt)
    DDFSDFPow = float(args.DDFSDFPow)
    surfacesamplespath = args.surfacesamplespath

    if not (( fitTo == "distances") or (fitTo == "depths") or (fitTo == "colors") or (fitTo == "silhouette")):
        raise Exception(
            'Invalid mode to fit'
        )
        


    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var
    idListTest=[]
    exp_directory = args.experiment_directory

    specs_filename = os.path.join(exp_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))


    if specs["NetworkArchDDF"] == "models":
        arch_ddf = __import__("lfn." + specs["NetworkArchDDF"], fromlist=["LightFieldModel"])
    else:
        arch_ddf = __import__("networks." + specs["NetworkArchDDF"], fromlist=["DDFNet"])

    if not specs["NetworkArchSDF"] is None:
        arch_sdf = __import__("networks." + specs["NetworkArchSDF"], fromlist=["SDFNet"])
        no_net_sdf = False
    else:
        arch_sdf = None
        no_net_sdf = True

    latent_size = specs["CodeLength"]

    genProc = specs["generalpreprocessing"] if "generalpreprocessing" in specs else False
    missingDirs = specs["missingDirs"] if "missingDirs" in specs else False
    lat_vecs_seperate_ddf = specs["lat_vecs_seperate_ddf"] if "lat_vecs_seperate_ddf" in specs else False
    if specs["NetworkArchDDF"] == "models":
        decoder_ddf = arch_ddf.LightFieldModel(latent_size, **specs["NetworkSpecsDDF"])
    else:
        decoder_ddf = arch_ddf.DDFNet(latent_size, **specs["NetworkSpecsDDF"])
    decoder_ddf = torch.nn.DataParallel(decoder_ddf)
    saved_model_state_ddf = torch.load(
        os.path.join(
            exp_directory, ws.model_params_subdir, args.checkpoint + "_ddf.pth"
        ), map_location='cuda:0'
    )
    decoder_ddf.load_state_dict(saved_model_state_ddf["model_state_dict"])
    saved_model_epoch = saved_model_state_ddf["epoch"]
    decoder_ddf = decoder_ddf.module.cuda()
    
    
    if no_net_sdf:
        decoder_sdf = decoder_ddf
    else:
        decoder_sdf = arch_sdf.SDFNet(latent_size, **specs["NetworkSpecsSDF"])
        decoder_sdf = torch.nn.DataParallel(decoder_sdf)

        saved_model_state_sdf = torch.load(
        os.path.join(
        exp_directory, ws.model_params_subdir, args.checkpoint + "_sdf.pth"
        ), map_location='cuda:0'
        )
        decoder_sdf.load_state_dict(saved_model_state_sdf["model_state_dict"])
        saved_model_epoch = saved_model_state_sdf["epoch"]
        decoder_sdf = decoder_sdf.module.cuda()





    if args.split_name.endswith("json"):
        with open(args.split_name, "r") as f:
            split = json.load(f)
        useLearnedLatents=False
    else:
        if args.split_name == "Train":
            with open(specs["TrainSplit"], "r") as f:
                split = json.load(f)
            useLearnedLatents=True
        else:
            with open(specs["TestSplit"], "r") as f:
                split = json.load(f)
            useLearnedLatents=False

    npz_filenames = DDF.data.get_instance_filenames(args.data_source, split, appendName=getSamplesName(exp_directory,genProc), type=reconClass)
    print("Number of npzfiles: "+str(len(npz_filenames)))
    if fitDIST:
        if fitDebug:
            NumTestMeshes=1
        elif ablations:
            NumTestMeshes=64
        else:
            NumTestMeshes=200
            
        npz_filenames = npz_filenames[:NumTestMeshes]
    logging.info("Number of npzfiles: "+str(len(npz_filenames)))


    random.shuffle(npz_filenames)

    logging.debug(decoder_ddf)
    logging.debug(decoder_sdf)


    appendMeshName = "_feats" if feat3Drec else "" 
    appendMeshName += "_sdf" if reconOnlySDF else "" 
    
    reconstruction_dir = os.path.join(
        exp_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    if reconClass == 'image' or reconClass == 'video':
        reconstruction_meshes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_images_subdir
        )
    elif reconClass == 'fit':
        folderNameForFit = "fitDIST" if fitDIST else fitTo
        if abmode is not None:
            if abmode.startswith('SDFDDF'):
                folderNameForFit = folderNameForFit+"_"+abmode+"_"+str(DDFSDFWt)+"_"+str(DDFSDFPow)
            elif abmode.startswith('iters'):
                folderNameForFit = folderNameForFit+"_"+abmode+"_"+str(iterations)
            else:
                folderNameForFit = folderNameForFit+"_"+abmode
        
        
        if fitTo == "silhouette" and fitDIST:
            folderNameForFit = folderNameForFit+"_"+fitTo
        if fitDebug:
            reconstruction_meshes_dir = os.path.join(
                reconstruction_dir, ws.reconstruction_fit_subdir+"_"+folderNameForFit+"_debug"+"_m"+str(meshThresh)
            )
        else:
            reconstruction_meshes_dir = os.path.join(
                reconstruction_dir, ws.reconstruction_fit_subdir+"_"+folderNameForFit
            )
    else:
        reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir+appendMeshName
        )
    if rendersPaper:
        reconstruction_render_papers_dir = os.path.join(reconstruction_meshes_dir,"rendersPaper")

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir+appendMeshName
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstructedLatentGeom = torch.empty(1,int(latent_size))
    reconstructedLatentCol = torch.empty(1,int(latent_size))
    accuracyFile = os.path.join(reconstruction_meshes_dir,"accuracy.pkl")
    count_unavail = 0
    torch.manual_seed(0)
    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        logging.debug("loading {}".format(npz))
        npzFile = os.path.basename(npz)

        if not os.path.isfile(full_filename):
            if reconClass == "mesh":
                count_unavail += 1
                print("Skipping because file not found "+ str(count_unavail))
                continue
            data_dsdf = []
        else:
            if reconClass == "mesh":
                data_dsdf = DDF.data.read_sdf_samples_into_ram(full_filename,missingDirs=missingDirs)
            else:
                data_dsdf = []




        mesh_filename = os.path.join(reconstruction_meshes_dir, npzFile[:-4])
        if rendersPaper:
            renderPaperFile = os.path.join(reconstruction_render_papers_dir, npzFile[:-4])

        if reconClass =='mesh':
            latent_filename = os.path.join(
                reconstruction_codes_dir, npzFile[:-4] + ".pth" )
        else:
            latent_filename = os.path.join(
                reconstruction_dir, ws.reconstruction_fit_subdir+"_fitDIST", npzFile[:-4] + ".pth" )


        logging.info("Meshfile name {}".format(mesh_filename))

        logging.info("reconstructing {}".format(npz))

        if lat_vecs_seperate_ddf:
            latentCodes, latentCodes_ddf = ws.load_latent_vectors_geometry(args.experiment_directory,args.checkpoint)
            meanLatent = latentCodes.mean(dim=0).unsqueeze(0)
            meanLatent = torch.cat([meanLatent,latentCodes_ddf.mean(dim=0).unsqueeze(0)],dim=-1)
        else:
            latentCodes = ws.load_latent_vectors_geometry(args.experiment_directory,args.checkpoint)
            meanLatent = latentCodes.mean(dim=0).unsqueeze(0)
        logging.info("npz {}".format(npz))
        logging.info("npzFile {}".format(npzFile))
        if not useLearnedLatents:
            if (not (os.path.isfile(latent_filename) and args.skip)) and (not reconClass == 'fit'):
                start = time.time()
                err = 0;
                err, latentReconstructed = reconstructLatent(
                  decoder_ddf,
                  decoder_sdf,
                  int(1000),
                  latent_size,
                  data_dsdf,
                  0.01,  # [emp_mean,emp_var],
                  0.1,
                  num_samples=16384,
                  lr=5e-3,
                  l2reg=True,
                  learnFeats=feat3Drec,
                  reconDDF= (not reconOnlySDF)
                 )
                logging.info("reconstruct time: {}".format(time.time() - start))
            else:
                if os.path.isfile(latent_filename):
                    latentReconstructed = torch.load(latent_filename).squeeze()
                    latentReconstructed = [latentReconstructed]
                    if os.path.isfile(latent_filename[:-4]+"_feats.pth"):
                        latentReconstructed.append(torch.load(latent_filename[:-4]+"_feats.pth"))
                else:
                    latentReconstructed = torch.ones(1, latent_size).normal_(mean=0, std=0.01).cuda()
            latentcodeSet = latentReconstructed[0]
            if len(latentReconstructed)>1:
                addfeat = latentReconstructed[1]
            else:
                addfeat = []
        else:
            latentcodeSet = torch.cat([latentCodes[ii],latentCodes_ddf[ii]],dim=-1) #

        decoder_ddf.eval()
        decoder_sdf.eval()
        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))


        npzPathSplit = npz.split("/")
        npzStrippedName = npzPathSplit[2][:-4].split("_")[0]

        gtImagePath = os.path.join(specs["ShapeNetImgSource"],npzPathSplit[1],npzStrippedName,"img_choy2016","000.jpg")
        if fitDIST:
            gtPCEvalPath = os.path.join(surfacesamplespath,npzPathSplit[1],npzStrippedName+".ply")

        gtMeshPath = os.path.join(specs["ShapeNetMeshSource"],npzPathSplit[1],npzStrippedName,"models","model_normalized.obj")
        
        if not os.path.isfile(gtMeshPath):
            print("Not found: ",gtMeshPath)
            continue

        logging.info("GT Mesh not available. Total: {}".format(count_unavail))
        print(ii,gtMeshPath)
        start = time.time()
        fireinstance = fire_model(decoder_ddf, decoder_sdf, latentcodeSet, max_batch=2**18, dsdf_gt=None, seperate_lat=lat_vecs_seperate_ddf, no_net_sdf=no_net_sdf)
        meshRecon = False
        if reconClass == 'image':
            renderAndSaveImage(mesh_filename,imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh)
        elif reconClass == 'video':
            saveColorCodedGT(mesh_filename, gtMeshPath, gtImagePath)
            renderAndSaveVideo(mesh_filename,imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh,frames=25)
        elif reconClass == 'fit':
            if rendersPaper == False:
                renderPaperFile = None
            fireinstance.ddfthresh = 0.80
            fitImagesLatent(mesh_filename,accuracyFile,imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh, inImagePath=gtImagePath, latentInit=meanLatent, mode=fitTo, gt_pc=gtPCEvalPath, fitTo512Choy=False, evaluateOnly=evaluateMetricsOnly,skip=args.skip, meshThresh=meshThresh, rendersPaper=rendersPaper, renderPaperFile=renderPaperFile, abmode=abmode, DDFSDFWt=DDFSDFWt, DDFSDFPow=DDFSDFPow, iterations=iterations)
        elif reconClass == 'mesh':
            if not (os.path.isfile(mesh_filename+".ply") and args.skip):
                reconstructAndSaveMesh(mesh_filename,accuracyFile,gtMeshPath,fireinstance, thresh=meshThresh, gt_pc_path=gtPCEvalPath, gtData=data_dsdf, evalDDF=True,scaletype=args.scaletype)
                if not os.path.isfile(latent_filename):
                    torch.save(latentcodeSet.unsqueeze(0), latent_filename)
                if not os.path.isfile(latent_filename[:-4]+"_feats.pth") and (not addfeat is None):
                    torch.save(addfeat, latent_filename[:-4]+"_feats.pth")
            if not args.skip:
                renderAndSaveImage(mesh_filename,imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh)
                renderAndSaveVideo(mesh_filename,imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh, frames=25)
                if rendersPaper:
                    nframes = 6
                    for frame in range(0,nframes):
                        th = torch.tensor([-np.pi/6, 2*frame*np.pi/nframes])
                        renderAndSaveImage(renderPaperFile+"_recon_"+str(frame),imgWidth,imgHeight,fireinstance,gtMeshPath,thresh=thresh,inImagePath=None,saveGt=True, th=th)
            del addfeat[0]
            del addfeat[0]
            fireinstance.addDDFFeat = None
            fireinstance.addSDFFeat = None
            torch.cuda.empty_cache()
        logging.info("total time: {}".format(time.time() - start))
