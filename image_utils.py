#image rendering utilities

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import DDF
import cv2
import trimesh as tm
from renderutils import setupCameraAndGetRaysG, getSphereOrgs, shade
from mesh_utils import loadAndNormalizeMesh, writefile
from chamfer import computeChamfers
import skimage.measure
import plyfile
import pickle
import imageio


class StepLearningRateSchedule():
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

class ExpLearningRateSchedule():
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor
    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch / self.interval))


class fire_model():
    def __init__(self, decoder_ddf, decoder_sdf, latent_geom, max_batch=2**17, dsdf_gt = None, seperate_lat=False, ddfthresh=0.8, no_net_sdf=False):
        self.decoder_ddf = decoder_ddf
        decoder_ddf.eval()
        if hasattr(self.decoder_ddf, 'phi'):
            self.type = 'lfn'
        else:
            self.type = 'none'
        self.no_net_sdf = no_net_sdf
        if self.no_net_sdf:
            self.decoder_sdf = None
        else:
            self.decoder_sdf = decoder_sdf
            decoder_sdf.eval()
        self.max_batch = max_batch
        self.eps_n = 2e-3
        self.perfAnalysisWarmUp = False
        self.startGPU = torch.cuda.Event(enable_timing=True)
        self.endGPU = torch.cuda.Event(enable_timing=True)
        self.fit = False
        self.dsdf_gt = dsdf_gt
        self.dsdfFactor = 1.0
        self.ddfthresh = ddfthresh
        try:
            self.dsdf_out_dim = decoder_ddf.output_dim
        except:
            self.dsdf_out_dim = 1
        self.seperate_lat = seperate_lat
        if self.seperate_lat:
            if latent_geom.dim() == 1:
                latent_geom = latent_geom.unsqueeze(0)
            self.latent_vec_geom = latent_geom[:,:latent_geom.shape[-1]//2]
            self.latent_vec_ddf = latent_geom[:,latent_geom.shape[-1]//2:]
        else:
            self.latent_vec_geom = latent_geom
            self.latent_vec_ddf = latent_geom
    def evaluate(self, inputData, clampSDF=True, onlySDF=False, latVec=None, latVecDDF=None, retTime=False):
        if self.fit:
            return self.evalBackProp(inputData, clampSDF=clampSDF, onlySDF=onlySDF, latVec=latVec, latVecDDF=latVecDDF)
        else:
            return self.evaluateNoGrad(inputData, clampSDF=clampSDF, onlySDF=onlySDF, latVec=latVec, latVecDDF=latVecDDF, retTime=retTime)

    def evaluateNoGrad(self, inputData, clampSDF=True, onlySDF=False, latVec=None, latVecDDF=None, retTime=False):
        if latVec is None:
            latVec = self.latent_vec_geom
        if latVecDDF is None:
            latVecDDF = self.latent_vec_ddf
        maxBatch = self.max_batch
        dsdf = torch.zeros((inputData.shape[0],self.dsdf_out_dim))
        sdf = torch.zeros((inputData.shape[0],1))
        iter = 0
        while iter < inputData.shape[0]:
            if iter+maxBatch > inputData.shape[0]:
                upperLim = inputData.shape[0]
            else:
                upperLim = iter+maxBatch

            currData = inputData[iter:upperLim,:].cuda()
            totalStartTime = time.time()
            gpu_time_dsdf = 0
            if not onlySDF:
                currData = torch.cat([latVecDDF.repeat(currData.shape[0],1).float().cuda(),currData],dim=1)
                if self.perfAnalysisWarmUp == False:
                    print("Warming up for performance analysis")
                    with torch.no_grad():
                        _ = self.decoder_ddf(currData, inputDataDims=6)
                    self.perfAnalysisWarmUp = True

                self.startGPU.record()
                with torch.no_grad():
                    curr_dsdf = self.decoder_ddf(currData,inputDataDims=6)
                    if self.dsdf_out_dim>1:
                        curr_dsdf_seg = curr_dsdf[1].squeeze()
                        curr_dsdf = curr_dsdf[0]*self.dsdfFactor
                self.endGPU.record()
                torch.cuda.synchronize()
                gpu_time_dsdf = self.startGPU.elapsed_time(self.endGPU)
                print("DSDF: GPU Time in ms: "+str(gpu_time_dsdf))

                dsdf[iter:upperLim,0] = curr_dsdf
                if self.dsdf_out_dim>1:
                    dsdf[iter:upperLim,1] = curr_dsdf_seg
                dataPts = torch.cat([currData[:,-6:-3]+currData[:,-3:]*curr_dsdf.unsqueeze(-1)],dim=1)
            if not onlySDF:
                data = torch.cat([latVec.repeat(currData.shape[0],1).float().cuda(),dataPts],dim=1)
            else:
                data = torch.cat([latVec.repeat(currData.shape[0],1).float().cuda(),currData],dim=1)
            with torch.no_grad():
                self.startGPU.record()
                if not self.no_net_sdf:
                    curr_sdf = self.decoder_sdf(data)
                    curr_sdf = curr_sdf.squeeze()
                else:
                    curr_sdf = torch.tensor(0)
                self.endGPU.record()
                torch.cuda.synchronize()
                gpu_time_sdf = self.startGPU.elapsed_time(self.endGPU)
                # print("SDF: GPU Time in ms: " + str(gpu_time_sdf))
            sdf[iter:upperLim,:] =  curr_sdf.unsqueeze(-1)
            iter = upperLim
            # print("Total time in ms: "+str(gpu_time_sdf+gpu_time_dsdf))
            gpu_time_dsdf = gpu_time_dsdf+gpu_time_sdf

        if clampSDF:
            sdf = torch.clamp(sdf, -0.1, 0.1)
        if retTime:
            return dsdf, sdf, gpu_time_dsdf
        else:
            return dsdf,sdf

    def evalBackProp(self, inputData, clampSDF=True, onlySDF=False, latVec = None, latVecDDF = None):
        if latVec is None:
            latVec = self.latent_vec_geom
        if latVecDDF is None:
            latVecDDF = self.latent_vec_ddf
        if onlySDF:
            dsdf = 0
            sdfData = torch.cat([latVec.repeat(inputData.shape[0],1).float().cuda(),inputData],dim=1)
        else:
            dsdfdata = torch.cat([latVecDDF.repeat(inputData.shape[0],1).float().cuda(),inputData],dim=1)
            dsdf = self.decoder_ddf(dsdfdata,inputDataDims=6)
            if self.dsdf_out_dim>1:
                dsdf[0] = dsdf[0]*self.dsdfFactor
                dsdf = torch.cat([dsdf[0].unsqueeze(-1),dsdf[1]],dim=-1)
            else:
                dsdf = dsdf*self.dsdfFactor
            if self.haveFeatPl:
                if self.featPlanes is not None:
                    dsdfAdd = self.featPlanes(inputData.cuda())
                    dsdf += dsdfAdd
            sdfdata = torch.cat([latVec.repeat(inputData.shape[0],1).float().cuda(),inputData[:,-6:-3]+inputData[:,-3:]*dsdf.unsqueeze(-1)],dim=1)
        sdf = self.decoder_sdf(sdfdata)
        if clampSDF:
            sdf = torch.clamp(sdf, -0.1, 0.1)
        return dsdf, sdf

    def computeNormals(self,points,grad=False,sdf=None, latVec = None, retTime=False):
        if latVec is None:
            latVec = self.latent_vec_geom
        if grad:
            start = time.time()
            self.startGPU.record()
            a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([self.eps_n,0,0]).cuda()],dim=1)
            b = sdf
            norm_x = self.decoder_sdf(a)-self.decoder_sdf(b)
            a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([0,self.eps_n,0]).cuda()],dim=1)
            norm_y = self.decoder_sdf(a)-self.decoder_sdf(b)
            a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([0,0,self.eps_n]).cuda()],dim=1)
            norm_z = self.decoder_sdf(a)-self.decoder_sdf(b)

            self.endGPU.record()
            torch.cuda.synchronize()
            gpu_time = self.startGPU.elapsed_time(self.endGPU)
            print("Normals: GPU Time in ms: "+str(gpu_time))
            normals = torch.cat([norm_x.unsqueeze(-1),norm_y.unsqueeze(-1),norm_z.unsqueeze(-1)],dim=1)
            normals = normals*(1/torch.norm(normals,dim=1).unsqueeze(-1))
        else:
            with torch.no_grad():
                start = time.time()
                self.startGPU.record()
                a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([self.eps_n,0,0]).cuda()],dim=1)
                b = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points-torch.tensor([self.eps_n,0,0]).cuda()],dim=1)
                norm_x = self.decoder_sdf(a).squeeze()-self.decoder_sdf(b).squeeze()
                a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([0,self.eps_n,0]).cuda()],dim=1)
                b = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points-torch.tensor([0,self.eps_n,0]).cuda()],dim=1)
                norm_y = self.decoder_sdf(a).squeeze()-self.decoder_sdf(b).squeeze()
                a = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points+torch.tensor([0,0,self.eps_n]).cuda()],dim=1)
                b = torch.cat([latVec.repeat(points.shape[0],1).float().cuda(),points-torch.tensor([0,0,self.eps_n]).cuda()],dim=1)
                norm_z = self.decoder_sdf(a).squeeze()-self.decoder_sdf(b).squeeze()
                self.endGPU.record()
                torch.cuda.synchronize()
                gpu_time = self.startGPU.elapsed_time(self.endGPU)
                print("Normals: GPU Time in ms: "+str(gpu_time))
                normals = torch.cat([norm_x.unsqueeze(-1),norm_y.unsqueeze(-1),norm_z.unsqueeze(-1)],dim=1)
                normals = normals*(1/torch.norm(normals,dim=1).unsqueeze(-1))
        if retTime:
            return normals, gpu_time
        else:
            return normals

def colorCodeDepth(inImg):
    givenImg = inImg[:,:,0]
    depthValues = givenImg[(givenImg==255)==False].astype('float')
    depthValues = (depthValues-depthValues.min())
    depthValues = 128.0*depthValues/depthValues.max()
    depthValues += 1
    givenImg[(givenImg==255)==False] = depthValues.astype('uint8')
    
    givenImg = (givenImg).astype('uint8')
    colorCodedImage =  cv2.applyColorMap(givenImg, cv2.COLORMAP_PLASMA)
    colorCodedImage[inImg[:,:,0]==255,:] = 255
    return colorCodedImage

def saveGroundTruthImages(tmmesh,outFile,origins,rays,imgShape, write_images = False, projMat=None, retDepths=False):
    origins = origins.detach().cpu().numpy()
    rays = rays.detach().cpu().numpy()
    intersector = tm.ray.ray_pyembree.RayMeshIntersector(tmmesh, scale_to_box=False)

    iloc, iIdxs , fIdxs = intersector.intersects_location(origins,
                       rays, multiple_hits=False)
    hitMiss = np.zeros((rays.shape[0],1))
    hitMiss[iIdxs] = 1
    depths = np.zeros((rays.shape[0],1))
    depthsdb = np.zeros((rays.shape[0],1))
    if projMat is not None:
        depths[hitMiss==1] = np.dot(projMat,np.append(iloc.T,np.ones((1,iloc.shape[0])),axis=0))[2]
    else:
        depths[hitMiss==1] = np.absolute(origins[iIdxs,2]-iloc[:,2])
    depthsdb[hitMiss==1] = np.linalg.norm(origins[iIdxs]-iloc,axis=1)

    normals = np.zeros((rays.shape[0],3))
    normals[iIdxs] = tmmesh.face_normals[fIdxs].squeeze()
    normals[iIdxs] = normals[iIdxs]*np.repeat(np.expand_dims(1/np.linalg.norm(normals[iIdxs],axis=1),axis=1),3,axis=1)

    normals = np.reshape(normals.flatten(),(imgShape[0],imgShape[1],3))
    hitMiss = np.reshape(hitMiss.flatten(),imgShape[0:2])
    depths = np.reshape(depths.flatten(),imgShape[0:2])

    scale = np.max(np.max(depths))

    if write_images:
        t1 = getFrame(depths,hitMiss,depthScale=scale)
        cv2.imwrite(outFile+"_depth_gt.png",colorCodeDepth(t1))
        t = cv2.equalizeHist((depths*255.0).astype('uint8'))
        t = np.stack((t,t,t,(hitMiss*255.0).astype('uint8')),axis=2)
        cv2.imwrite(outFile+"_depth_eq_gt.png",t)
        cv2.imwrite(outFile+"_occupancy_gt.png",(hitMiss*255.0).astype('uint8'))
        pc = origins+np.repeat(depthsdb,3,axis=1)*rays
        img = getShadedImage(imgShape[0:2],hitMiss,normals,pc,origins[0,:].squeeze())
        cv2.imwrite(outFile+"_shaded_gt.png",img)
    if retDepths:
        return scale,hitMiss,np.reshape(depthsdb.flatten(),imgShape[0:2]), normals, depths
    else:
        return scale,hitMiss,np.reshape(depthsdb.flatten(),imgShape[0:2]), normals

def torSkew(th):
    zero = torch.tensor(0).unsqueeze(-1).to(th.device)
    th = th.unsqueeze(-1)
    return torch.cat([zero,-th[2],th[1],th[2],zero,-th[0],-th[1],th[0],zero]).reshape(3,3)

def rot(th,axis=None):
    if not axis is None:
        a = torch.cos(th).unsqueeze(-1)
        b = torch.sin(th).unsqueeze(-1)
        one = torch.tensor(1.0).unsqueeze(-1).to(th.device)
        zero = torch.tensor(0.0).unsqueeze(-1).to(th.device)
        if axis=='x':
            return torch.cat([one,zero,zero,zero,a,-b,zero,b,a]).reshape(3,3)
        elif axis=='y':
            return torch.cat([a,zero,b,zero,one,zero,-b,zero,a]).reshape(3,3)
        elif axis=='z':
            return torch.cat([a,-b,zero,b,a,zero,zero,zero,one]).reshape(3,3)
        else:
            return torch.cat([one,zero,zero,zero,one,zero,zero,zero,one]).reshape(3,3)
    else:
        return torch.matrix_exp(torSkew(th))

def sphereTracerKernel(data, thresh, numIters, ptFlags, counts, fireInstance, distance_fin, hitMiss_fin, clampSDF=True, mult=1.0):
    gpuComputeTime = 0
    iterations = 0
    while torch.sum(ptFlags) > 0:

        pc = data[ptFlags,0:3]+data[ptFlags,3:6]*distance_fin[ptFlags,:]
        _,sdf, gpuTime = fireInstance.evaluate(pc,clampSDF=clampSDF, onlySDF=True, retTime=True)
        gpuComputeTime += gpuTime
        distance_fin[ptFlags] += mult*sdf
        hitMiss_fin[ptFlags] = hitMiss_fin[ptFlags] | (torch.abs(sdf)<=thresh)

        temp = (torch.abs(sdf)>thresh).squeeze() & (torch.norm(pc,dim=1).squeeze()<=1.0)
        # import pdb; pdb.set_trace()
        ptFlags[ptFlags==True] = temp & ptFlags[ptFlags==True]
        counts[ptFlags] += 1
        print("Remaining: "+ str(torch.sum(ptFlags)))
        iterations += 1
        if iterations == numIters:
            break
    return gpuComputeTime, distance_fin, hitMiss_fin, iterations, counts

def sphereTrace(fireInstance, inFileName, numIters,  data, raySphereOrg, rays, t_sol, flags, thresh, depthScale, type='dsdf',\
                mult=1.0, gtHit=None, gtDepth=None, writeFiles=True, cOrg = None, localOptimisation=False, debug=False, R=None):
    fileName = os.path.join(os.path.dirname(inFileName),type)
    if not os.path.isdir(fileName):
        os.makedirs(fileName)
    fileName = os.path.join(fileName,os.path.basename(inFileName))


    distance_fin = torch.zeros((data.shape[0],1))
    normals_fin = torch.zeros((data.shape[0],3))
    hitMiss_fin = torch.zeros((data.shape[0],1)).to(torch.bool)

    if debug:
        flagsTest = torch.zeros(t_sol.shape)

    ptFlags = flags.view(-1,1).squeeze()==True
    counts = flags.view(-1,1).squeeze()*0
    totalTime = 0
    start=time.time()

    counts[ptFlags] += 1
    clampSDF = True
    # if type == "tsdf":
        # clampSDF = True
    # else:
        # clampSDF = False

    if debug:
        importantIdx = torch.where(flagsTest[ptFlags]==1)
        print(importantIdx)

    gpuComputeTime = 0
    iterations = 0
    if type == "dsdf" or type == "dsdfLoc":
        dsdf,sdf,gpuTime = fireInstance.evaluate(data[ptFlags,:],clampSDF,retTime=True)
        gpuComputeTime += gpuTime
        if dsdf.shape[1]==1:
            if not fireInstance.no_net_sdf:
                hitMiss_fin[ptFlags] = hitMiss_fin[ptFlags] | (torch.abs(sdf)<=thresh)
            else:
                hitMiss_fin[ptFlags] = hitMiss_fin[ptFlags]
        else:
            if not fireInstance.no_net_sdf:
                hitMiss_fin[ptFlags] = (hitMiss_fin[ptFlags] | (torch.abs(sdf)<=thresh)) | (dsdf[:,1] > fireInstance.ddfthresh).unsqueeze(-1)
            else:
                hitMiss_fin[ptFlags] = (hitMiss_fin[ptFlags]) | (dsdf[:,1] > fireInstance.ddfthresh).unsqueeze(-1)

            dsdf = dsdf[:,0].unsqueeze(-1)

        distance_fin[ptFlags] += mult*dsdf

        iterations += 1
        if localOptimisation:

            gpuTime, distance_fin, hitMiss_fin, iters_ret, counts = sphereTracerKernel(data, thresh, numIters, ptFlags, counts, fireInstance, distance_fin, hitMiss_fin, clampSDF=clampSDF)
            gpuComputeTime += gpuTime
            iterations +=iters_ret

    else:
        distance_fin = distance_fin+0.1
        gpuTime, distance_fin, hitMiss_fin, iters_ret, counts = sphereTracerKernel(data, thresh, numIters, ptFlags, counts, fireInstance, distance_fin, hitMiss_fin, clampSDF=clampSDF, mult=mult)
        iterations +=iters_ret
        gpuComputeTime += gpuTime

    tempIdx = (hitMiss_fin==True).squeeze()
    pc = data[tempIdx,0:3]+data[tempIdx,3:6]*distance_fin[tempIdx,:]
    depth = pc.clone()
    if R is not None:
        depth = torch.matmul(R , depth.T).T
        K = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float()
        K[0,0] = 0.5463
        K[1,1] = 0.5463
        depth = torch.matmul(K , depth.T).T
    depth = torch.abs(depth[:,2])
    # import pdb; pdb.set_trace()
    dpt = torch.zeros(t_sol.shape)
    dpt = dpt.view(-1)
    dpt[tempIdx] = depth
    depth = dpt.view(t_sol.shape)
    t = t_sol + distance_fin.reshape(t_sol.shape)
    if fireInstance.no_net_sdf:
        normals = torch.ones([t.shape[0],t.shape[1],3])
        normals[:-2,:,0] = -(t[2:,:] - t[:-2,:])/2
        normals[:,:-2,1] = -(t[:,2:] - t[:,:-2])/2
        normals = normals/torch.norm(normals,dim=-1).unsqueeze(-1)
        normals = normals.view(-1,normals.shape[-1])
        normals = normals[tempIdx,:]
        gpuTime += 0
    else:
        normals, gpuTime = fireInstance.computeNormals(pc.cuda(), retTime=True)
    gpuComputeTime+=gpuTime

    normals_fin[tempIdx,:] = normals.cpu()


    data = raySphereOrg+data[:,3:6]*distance_fin
    end=time.time()
    totalTime += end-start
    hitMiss = (hitMiss_fin).squeeze() & (flags.view(-1,1).squeeze()==True) & (torch.norm(data,dim=1).squeeze()<=1.0)
    hitMiss = hitMiss.reshape(t_sol.shape)
    normals_fin = normals_fin.reshape((t_sol.shape[0],t_sol.shape[1],3))
    if writeFiles == True:
        gtHit = gtHit.astype('bool')
        writeImages(fileName, data, t, hitMiss, gtDepth, gtHit, depthScale, list(t_sol.shape), normals_fin, cOrg,depth=depth.numpy())
        # writePC(fileName+"_pc.obj",data)


    print("Time to infer: "+str(totalTime))

    return gpuComputeTime, iterations, counts[counts>0].detach().cpu().numpy(), t, hitMiss, normals_fin

def writePC(fileName,points):
    points = points.cpu().numpy()
    f= open(fileName,"w+")
    if points.shape[1] == 3:
      for i in range(points.shape[0]):
         f.write( "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + "\n")
    f.close()

def writeToBuffer(buffer, t, predHit):
    return

def colorCodeDepth(inImg):
    givenImg = inImg[:,:,0]
    depthValues = givenImg[(givenImg==255)==False].astype('float')
    depthValues = (depthValues-depthValues.min())
    depthValues = 250.0*depthValues/depthValues.max()
    depthValues += 1
    givenImg[(givenImg==255)==False] = depthValues.astype('uint8')
    givenImg = (givenImg).astype('uint8')
    colorCodedImage =  cv2.applyColorMap(givenImg, cv2.COLORMAP_PLASMA)
    colorCodedImage[inImg[:,:,0]==255,:] = 255
    return colorCodedImage

def getFrame(t,predHit,depthScale=1.0):
    t = t*(predHit).astype('float')
    if depthScale == 1.0:
        t = t/np.max(np.max(t))
    else:
        t = t/depthScale
    t = 1-t
    t1 = (np.stack((t,t,t,predHit),axis=2)*255.0).astype('uint8')
    

    return t1

def getShadedImage(imageSize,predHit,normals,pc,cOrg):
    img1 = np.ones((imageSize[0]*imageSize[1],3))
    idxs = np.reshape(predHit,(-1,1)).squeeze()==1
    normals = np.reshape(normals,(-1,3)).squeeze()
    img1[idxs,:] = shade(cOrg, normals[idxs,:], pc[idxs,:], cOrg, lightPower=2.5)
    img1 = (255.0*np.reshape(img1,(imageSize[0],imageSize[1],3))).astype('uint8')
    return img1

def errorImage(gtDepth,t,gtHit,imageSize):
    disterr = np.abs(gtDepth-t)

    # import pdb; pdb.set_trace()
    #
    # disterrClone = np.clone(disterr)
    disterr[disterr<3e-4] = 0

    disterr[disterr>0.03] = np.max(np.max(disterr))



    disterr = disterr/np.max(np.max(disterr))
    disterr = np.repeat(np.expand_dims(disterr,axis=2),3,axis=2)
    #red image
    img1 = 0*np.ones((imageSize[0],imageSize[1],3))
    img1[:,:,2] = 255.0

    # green image
    img2 = 0*np.ones((imageSize[0],imageSize[1],3))
    img2[:,:,1] = 255.0

    # image based on error
    img = (img2*(1-disterr)+img1*disterr).astype('uint8')
    img[gtHit==False,:] = 0
    return img

def writeImages(fileName, pc, t, predHit, gtDepth, gtHit, depthScale, imageSize, normals=None, cOrg=None, depth=None):
    # debug error
    pc = pc.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    
    np.save(fileName+"_distances.npy",t)
    
    predHit = predHit.detach().cpu().numpy()
    disterr = np.abs(gtDepth-t)
    disterr[gtHit==False] = 0
    print("Max error:"+str(np.max(np.max(disterr[gtHit==True]))))
    print("Median error:"+str(np.median(disterr[gtHit==True])))
    print("Mean error:"+str(np.mean(disterr[gtHit==True])))
    print("99th Percentile error:"+str(np.percentile(disterr[gtHit==True].squeeze(),99)))
    print("95th Percentile error:"+str(np.percentile(disterr[gtHit==True].squeeze(),95)))
    print("90th Percentile error:"+str(np.percentile(disterr[gtHit==True].squeeze(),90)))

    with open(fileName+"_errors.pkl","wb") as f:
        renderTimes = {}
        renderTimes["max"] = np.max(np.max(disterr[gtHit==True]))
        renderTimes["mean"] = np.mean(disterr[gtHit==True])
        renderTimes["median"] = np.median(disterr[gtHit==True])
        pickle.dump(renderTimes, f)
        f.close()

    #################writing error image
    img = errorImage(gtDepth,t,gtHit,imageSize)
    cv2.imwrite(fileName+"_errorInDSDF.png",img)
    ######################

    t1 = getFrame(t,predHit,depthScale=depthScale)
    cv2.imwrite(fileName+"_depth.png",t1)
    t1 = getFrame(depth,predHit,depthScale=1.0)
    cv2.imwrite(fileName+"_truedepth.png",t1)



    t = cv2.equalizeHist((t*255.0).astype('uint8'))
    t = np.stack((t,t,t,((predHit)*255.0).astype('uint8')),axis=2)
    cv2.imwrite(fileName+"_eq_depth.png",t)


    cv2.imwrite(fileName+"_occupancy.png",((predHit)*255.0).astype('uint8'))


    if not normals is None:
        normals = normals.detach().cpu().numpy()
        startGPU = torch.cuda.Event(enable_timing=True)
        endGPU = torch.cuda.Event(enable_timing=True)
        startGPU.record()
        img1 = getShadedImage(imageSize,predHit,normals,pc,cOrg)
        endGPU.record()
        torch.cuda.synchronize()
        gpu_time = startGPU.elapsed_time(endGPU)
        print("Time to shade: in ms "+ str(gpu_time))
        cv2.imwrite(fileName+"_shaded.png",img1)
    return

def renderAndSaveImage(fileName, f_width, f_height, fireInstance, gtMeshPath, skip=True, thresh=1e-3,inImagePath=None, saveGt=True, th=None, modes=["dsdf","sdf","dsdfLoc"],rotate=0):
    # if os.path.isfile(os.path.join(fileName,"_depth_gt.png")) and skip:
        # return
    print(f_width)
    print(f_height)
    decoder = fireInstance
    #setup rotation matrix for camera Rotation
    if th is None:
        th = np.pi*torch.ones(3,)
        R = np.matmul(rot(torch.tensor(np.pi/4),'x'),rot(torch.tensor(np.pi),'y'))
    else:
        R = torch.matmul(rot(th[1],'y'),rot(th[0],'x'))

    # print(R)
    start = time.time()
    if inImagePath is None:
        org, rays, raySphereOrg, flags, t_sol = setupCameraAndGetRaysG(f_width,f_height,2.5,R,onSphere=True)
    else:
        logging.info("Using benchmark baseline parameters")
        org, rays, raySphereOrg, flags, t_sol, f_height, f_width = setupEvalAndGetRays(fileName, gtMeshPath, inImagePath)

    print("Time for camera setup: "+str(time.time()-start))
    data = torch.cat([raySphereOrg,rays],dim=-1).reshape(-1,6)
    orgsRep = torch.zeros(rays.shape) + org

    if gtMeshPath is not None:
        tmmesh = loadAndNormalizeMesh(gtMeshPath,rotth=rotate)
        depthScale, hitmiss_gt, distancesGT, _ = saveGroundTruthImages(tmmesh,fileName,orgsRep.flatten().view(-1,3),rays.flatten().view(-1,3),list(raySphereOrg.shape),write_images=saveGt)
    else:
        depthScale, hitmiss_gt, distancesGT = 1.0, np.ones((f_height,f_width)), np.ones((f_height,f_width))

    raySphereOrg = np.reshape(raySphereOrg.flatten(),(-1,3))

    #DSDF rendering
    renderTimes = {}
    if "dsdf" in modes:
        numIters = 200
        dsdfTime, dsdfIters, dsdfcounts, distancesPred, hitmissPred, _ = sphereTrace(fireInstance, fileName, numIters,  data.clone(), raySphereOrg, rays, t_sol, flags, thresh, depthScale,type="dsdf",mult=1.0,gtHit=hitmiss_gt,gtDepth=distancesGT, cOrg = org, R=R)
        renderTimes["dsdf"] = dsdfTime

    if "sdf" in modes:
        numIters = 200
        sdfTime, sdfIters, sdfcounts, distancesPredSDF, hitmissPredSDF, _ = sphereTrace(fireInstance, fileName, numIters,  data.clone(), raySphereOrg, rays, t_sol, flags, thresh, depthScale,type="sdf",mult=1.0,gtHit=hitmiss_gt,gtDepth=distancesGT, cOrg = org, R=R)
        renderTimes["sdf"] = sdfTime

    if "dsdfLoc" in modes:
        numIters = 200
        dsdfLocTime, dsdfLocIters, dsdfLoccounts, distancesPreddsdfLoc, hitmissPreddsdfLoc, _= sphereTrace(fireInstance, fileName, numIters,  data.clone(), raySphereOrg, rays, t_sol, flags, thresh, depthScale,type="dsdfLoc",mult=1.0,gtHit=hitmiss_gt,gtDepth=distancesGT, cOrg = org,localOptimisation=True, R=R)
        renderTimes["dsdf_loc"] = dsdfLocTime

    with open(fileName+"_renderTimes.pkl","wb") as f:
        pickle.dump(renderTimes, f)
        f.close()

    if "dsdf" in modes:
        print("DDF: Time:" + str(dsdfTime) + "s Iters: " + str(dsdfIters) )
        print("DDF Counts max:" + str(np.max(dsdfcounts)) + " min: " + str(np.min(dsdfcounts)) + " mean: " + str(np.mean(dsdfcounts))+ " median: " + str(np.median(dsdfcounts)) )
    if "sdf" in modes:
        print("SDF: Time:" + str(sdfTime) + "s Iters: " + str(sdfIters) )
        print("SDF Counts max:" + str(np.max(sdfcounts)) + " min: " + str(np.min(sdfcounts)) + " mean: " + str(np.mean(sdfcounts))+ " median: " + str(np.median(sdfcounts)) )
    if "dsdfLoc" in modes:
        print("DDF Loc: Time:" + str(dsdfLocTime) + "s Iters: " + str(dsdfLocIters) )
        print("DDF Loc Counts max:" + str(np.max(dsdfLoccounts)) + " min: " + str(np.min(dsdfLoccounts)) + " mean: " + str(np.mean(dsdfLoccounts))+ " median: " + str(np.median(dsdfLoccounts)) )
    return

def transform_camera(intrinsic, extrinsic, scale, offset):
    '''
    transform the extrinsic parameters to align with the sdf volume.
    '''
    transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    transform_matrix2 = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    K = intrinsic
    R, T = extrinsic[:,:3], extrinsic[:,3]
    T_new = (np.dot(np.dot(R, transform_matrix2), -offset) + T) * scale
    R_new = np.dot(R,transform_matrix2)
    RT_new = np.concatenate([R_new, T_new[:,None]], 1)
    return K, RT_new

def setupEvalAndGetRays(fileName, gtMeshPath, inImagePath, retGTImg=False, fitTo512Choy=False, returnCameraMats=False, scale=None):
    if not os.path.isfile(inImagePath):
        logging.info("Can't find the gt file" + inImagePath)
        return
    _,scale,cent = loadAndNormalizeMesh(gtMeshPath,normalize=1.0/1.03, returnScale=True)
    gtImg = cv2.imread(inImagePath)
    cv2.imwrite(fileName+"choy.jpg",gtImg)
    if not fitTo512Choy:
        f_width = gtImg.shape[0]
        f_height = gtImg.shape[1]
    else:
        f_width = 512
        f_height = 512


    # gtImg = np.flip(gtImg,2)
    gtImg = gtImg.copy()

    camerasFile=os.path.join(os.path.dirname(inImagePath),"cameras.npz")
    cameras = np.load(camerasFile)
    camIntrinsic = cameras["camera_mat_0"]
    if fitTo512Choy:
        camIntrinsic = camIntrinsic*f_height*0.5/camIntrinsic[0,2]
        camIntrinsic[2,2] = 1.0
    worldMat = cameras["world_mat_0"]
    # camIntrinsic[0,0] *= scale
    # camIntrinsic[1,1] *= scale
    camIntrinsic,worldMat = transform_camera(camIntrinsic,worldMat, 1/scale, -cent)

    cols = np.linspace(0,f_width-1,f_width)
    rows = np.linspace(0,f_height-1,f_height)
    pixels = np.meshgrid(rows,cols)
    pix = np.ones((f_height,f_width,3))
    pix[:,:,0] = pixels[1]
    pix[:,:,1] = pixels[0]

    pix2 = np.transpose(pix,[2,1,0])
    kinv=np.linalg.inv(camIntrinsic)
    pix3=torch.matmul(torch.from_numpy(kinv),torch.from_numpy(pix2).reshape(3,-1))

    pix4=torch.cat([pix3,torch.ones([1,pix3.shape[1]]).double()],dim=0)
    gInv=np.linalg.inv(np.append(worldMat,np.array([[0,0,0,1]]),axis=0))
    pix3d = torch.matmul(torch.from_numpy(gInv),pix4.view(4,-1))

    dirs = pix3d.view(4,f_width,f_height).numpy()
    dirs = np.transpose(dirs,[1,2,0])
    rays = torch.from_numpy(dirs[:,:,:-1])
    cameraOrg = np.matmul(gInv,np.array([[0,0,0,1]]).T)
    org=torch.from_numpy(cameraOrg.squeeze()[:-1])
    orgsRep = torch.zeros(rays.shape).to(org.device).double() + org

    rays = rays - orgsRep
    rays = rays*(1/torch.norm(rays,dim=-1)).unsqueeze(-1)
    raySphereOrg, t_sol, flags  = getSphereOrgs(org, rays)

    if retGTImg and returnCameraMats:
        return org.float(), rays.float(), raySphereOrg.float(), flags.to(torch.bool), t_sol.float(), f_height, f_width, torch.from_numpy(gtImg), camIntrinsic, worldMat, scale, cent
    elif retGTImg:
        return org.float(), rays.float(), raySphereOrg.float(), flags.to(torch.bool), t_sol.float(), f_height, f_width, torch.from_numpy(gtImg)
    else:
        return org.float(), rays.float(), raySphereOrg.float(), flags.to(torch.bool), t_sol.float(), f_height, f_width


def fitImagesLatent(fileName, accuracyFile, f_width, f_height, fireInstance, gtMeshPath, skip=True, thresh=1e-3, inImagePath=None,latentInit=None, mode="distances",gt_pc=None, fitTo512Choy=False,evaluateOnly = False, meshThresh=1e-3, rendersPaper=False, renderPaperFile=None, abmode=None, DDFSDFWt=1e0, DDFSDFPow = 1.0, iterations=1000):
    if os.path.isfile(fileName+"_recon.ply") and skip and (not evaluateOnly):
        return
    decoder = fireInstance
    evaluateBaseline = True
    debugVid = False
    render_width = f_width
    render_height = f_height

    renderThresh = 0.009
    if not evaluateOnly:
        start = time.time()
        org, rays, raySphereOrg, _, _,  f_height, f_width, colorsGT, camIntrinsic, worldMat, scale, cent = setupEvalAndGetRays(fileName, gtMeshPath, inImagePath,fitTo512Choy=fitTo512Choy, retGTImg=True, returnCameraMats=True)

        if debugVid:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(fileName+".avi", fourcc, 10, (f_width*3, f_height),True)
        data = torch.cat([raySphereOrg,rays],dim=-1).reshape(-1,6)
        orgsRep = torch.zeros(rays.shape).to(org.device) + org
        tmmesh = loadAndNormalizeMesh(gtMeshPath, normalize=1.0)
        _, hitmiss_gt, distancesGT, normalsGT, depthsGT = saveGroundTruthImages(tmmesh,fileName,orgsRep.flatten().view(-1,3),rays.flatten().view(-1,3),list(raySphereOrg.shape),write_images=True, retDepths=True, projMat= np.dot(camIntrinsic,worldMat))
        print("Time for GT setup: "+str(time.time()-start))
        normalsGT = torch.from_numpy(normalsGT).cpu().reshape(-1,3).float()
        distancesGT = torch.from_numpy(distancesGT).cpu().reshape(-1).float()
        depthsGT = torch.from_numpy(depthsGT).cpu().reshape(-1).float()
        colorsGT = (colorsGT.view(-1,3).float())/255.0
        orgsRep = orgsRep.flatten().view(-1,3)
        hitmiss_gt = torch.from_numpy(hitmiss_gt==False).cpu().reshape(-1).float()

        optParamList = []
        if latentInit is not None:
            latent = latentInit[:,:decoder.latent_vec_geom.shape[-1]].clone().cuda() #+  torch.ones(1, decoder.latent_vec_geom.shape[-1]).normal_(mean=0, std=0.1).cuda()
        else:
            latent = torch.ones(1, decoder.latent_vec_geom.shape[-1]).normal_(mean=0, std=0.1).cuda()
        optParamList.append({ "params": latent})
        latent.requires_grad = True

        if fireInstance.seperate_lat:
            if latentInit is not None:
                latentDDF = latentInit[:,decoder.latent_vec_ddf.shape[-1]:].clone().cuda()
            else:
                latentDDF = torch.ones(1, decoder.latent_vec_ddf.shape[-1]).normal_(mean=0, std=0.01).cuda()
            optParamList.append({ "params": latentDDF})
            latentDDF.requires_grad = True
        initLR = 0.001
        numIters = iterations
        optimizer = torch.optim.Adam(optParamList, lr=initLR)

        learningRateScheduler = StepLearningRateSchedule(initLR, numIters, 2)
        def adjust_learning_rate( optimizer, iteration ):
            lr = learningRateScheduler.get_learning_rate(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


        decoder.fit = True
        lossBCE = torch.nn.BCELoss()
        breakIter = numIters

        batch_size = 16384 if f_width*f_height>16384 else f_width*f_height
        GT1Idx = torch.where(distancesGT>0)[0]
        GT0Idx = torch.where(distancesGT==0)[0]
        if GT1Idx.shape[0] <int(3*batch_size/4):
            batch_size = int(GT1Idx.shape[0]*4/3)

        org, rays, raySphereOrg, flags, t_sol, f_height, f_width, _, camIntrinsic, worldMat, scale, cent = setupEvalAndGetRays(fileName, gtMeshPath, inImagePath, fitTo512Choy=fitTo512Choy, retGTImg=True, returnCameraMats=True)
        worldMat = torch.from_numpy(worldMat).float().cuda()
        camIntrinsic = torch.from_numpy(camIntrinsic).float().cuda()
        data = torch.cat([raySphereOrg,rays],dim=-1).reshape(-1,6).cuda()
        t_sol = t_sol.flatten().view(-1).cuda()
        flags = flags.flatten().view(-1)

        minLoss = np.inf
        start = time.time()
        totalTime = 0
        for iteration in range(0,numIters):

            iterationStart = time.time()
            adjust_learning_rate(optimizer,iteration)
            optimizer.zero_grad()

            batchIdx1 =torch.randint(0,GT1Idx.shape[0],(int(3*batch_size/4),))
            batchIdx2 =torch.randint(0,GT0Idx.shape[0],(int(batch_size/4),))
            batchIdx = torch.cat([GT1Idx[batchIdx1],GT0Idx[batchIdx2]])
            batchIdxBool = torch.cat([torch.ones_like(GT1Idx[batchIdx1]),torch.zeros_like(GT0Idx[batchIdx2])]).bool()
            if not fireInstance.seperate_lat:
                latentDDF = latent

            currBatchSize = data[batchIdx,:].shape[0]
            dsdfDataIn = data[batchIdx,:].cuda()
            dsdfDataIn = torch.cat([latentDDF.repeat(dsdfDataIn.shape[0],1),dsdfDataIn],dim=1)
            
            if abmode == "oneSDFStep":
                dsdf = decoder.decoder_sdf(torch.cat([latentDDF.repeat(data[batchIdx,0:3].shape[0],1),data[batchIdx,0:3].cuda()],dim=1),inputDataDims=3)
                dsdf_hitMiss = None
            else:
                if not abmode == "noDDF":
                    dsdf = decoder.decoder_ddf(dsdfDataIn,inputDataDims=6)
                    # import pdb; pdb.set_trace()
                    if decoder.dsdf_out_dim>1:
                        dsdf_hitMiss = dsdf[1].squeeze()
                        dsdf = dsdf[0]
                    else:
                        dsdf_hitMiss = None
                    dsdf = dsdf[:currBatchSize]
                else:
                    dsdf = (distancesGT[batchIdx].cuda()-t_sol[batchIdx].cuda())
                    dsdf_hitMiss = None
            
            pc = data[batchIdx,0:3]+data[batchIdx,3:]*dsdf.unsqueeze(-1)
            if abmode == "SDFGTSupDepth":
                pcSDF = data[batchIdx,0:3]+data[batchIdx,3:]*dsdf.unsqueeze(-1)
                pcSDF[batchIdxBool,:] = data[batchIdx,0:3][batchIdxBool,:]+data[batchIdx,3:][batchIdxBool,:]*(distancesGT[batchIdx].cuda()-t_sol[batchIdx].cuda())[batchIdxBool].unsqueeze(-1)
            else:
                pcSDF = pc
            sdfDataIn = pcSDF

            if not decoder.no_net_sdf:
                sdf = decoder.decoder_sdf(torch.cat([latent.repeat(sdfDataIn.shape[0],1),sdfDataIn.cuda()],dim=1),inputDataDims=3)
                sdf = sdf[:currBatchSize]
                


            lossIter = 0
            lossIdx = GT1Idx[batchIdx1]

            if mode == "depths":
                if not abmode == "noDDF":
                    depthPred = pc.cuda()
                    depthPreds = torch.cat([depthPred.transpose(0,1),torch.ones_like(depthPred.transpose(0,1)[-1,:].unsqueeze(0))],dim=0)
                    depthPreds = torch.matmul(worldMat,depthPreds)
                    depthPreds = torch.matmul(camIntrinsic,depthPreds)
                    depthPred = torch.abs(depthPreds[-1,:])
                    depthPred[(flags[batchIdx]==False) | (torch.norm(pc,dim=1)>1.0).cpu().detach()] = 0
                    dsdfLoss = 1.0*torch.abs(depthPred[batchIdxBool] - depthsGT[lossIdx].cuda()).mean()
                else:
                    dsdfLoss = 0

                w_latentNorm = (1e-4)
            elif mode == "silhouette":
                dsdfLoss = 0.0
                w_latentNorm = (5e-2)
            checknosil = True if abmode is None else (not abmode.startswith("noSil"))
            if checknosil:
                if not decoder.no_net_sdf:
                    sdfSurfLoss = torch.abs(sdf[batchIdxBool]).mean()
                    sdfNonSurfLoss = torch.abs(torch.abs(torch.clamp(sdf[batchIdxBool==False],-0.1,0.1))-0.1).mean()
                else:
                    sdfSurfLoss = torch.tensor(0).cuda()
                    sdfNonSurfLoss = torch.tensor(0).cuda()
                    
                
                dsdf_hitMissLoss = 0
                if dsdf_hitMiss is not None:
                    dsdf_hitMissLoss += (1e-2)*lossBCE(dsdf_hitMiss[:currBatchSize].squeeze(),batchIdxBool.float().squeeze().cuda())
            else:
                if abmode.endswith("noSDF"):
                    sdfsilregularizerweight =0.0
                else:
                    sdfsilregularizerweight =1.0
                
                dsdf_hitMissLoss = torch.tensor(0)
                if not decoder.no_net_sdf:
                    sdfSurfLoss = sdfsilregularizerweight*torch.abs(sdf[dsdf_hitMiss[:currBatchSize]>decoder.ddfthresh]).mean()
                    sdfNonSurfLoss = sdfsilregularizerweight*torch.abs(torch.abs(torch.clamp(sdf[dsdf_hitMiss[:currBatchSize]<=decoder.ddfthresh],-0.1,0.1))-0.1).mean()
                else:
                    sdfSurfLoss = torch.tensor(0).cuda()
                    sdfNonSurfLoss = torch.tensor(0).cuda()
                    
            if abmode == "noDDF":
                lossIter = sdfSurfLoss 
            elif abmode == "noSDFSilhouette":
                lossIter = dsdfLoss + dsdf_hitMissLoss
            elif abmode == "noSDFNegSilhouette":
                lossIter = dsdfLoss + dsdf_hitMissLoss + sdfSurfLoss
            elif abmode == "noSDFPosSilhouette":
                lossIter = dsdfLoss + dsdf_hitMissLoss + sdfNonSurfLoss
            elif abmode == "noDDFSilhouette":
                lossIter = dsdfLoss + sdfSurfLoss + sdfNonSurfLoss 
            elif abmode == "onlyDDF":
                lossIter = dsdfLoss + dsdf_hitMissLoss
            elif abmode == "noDDFSDFCons":
                lossIter = dsdfLoss + sdfSurfLoss + sdfNonSurfLoss + dsdf_hitMissLoss
            else:
                lossIter = dsdfLoss + sdfSurfLoss + sdfNonSurfLoss + dsdf_hitMissLoss


            if lossIter < minLoss:
                minLoss = lossIter
                iterBest = iteration
            if not abmode == "noLatentNorm":
                lossIter += w_latentNorm*torch.norm(latent.squeeze())
                if fireInstance.seperate_lat:
                    lossIter += w_latentNorm*torch.norm(latentDDF.squeeze())
            
            lossIter.backward()

            optimizer.step()
            iterationEnd = time.time()
            totalTime += iterationEnd-iterationStart

            if (iteration%50 == 0):

                end = time.time()
                logging.info("Time for the last 50 iterations: {:.3f}".format(end-start))
                start = time.time()
                if not evaluateBaseline:
                    logging.info("Iter: {:.3f}, Losses: DSDF: {:.3f}, SDF: {:.3f}, SDF Non Surface: {:.3f}, Latent Norm: {:.3f}, dsdf_hitMissLoss: {:.3f}".format(iteration,dsdfLoss,sdfSurfLoss,sdfNonSurfLoss,torch.norm(latent.squeeze()).item(),dsdf_hitMissLoss.item()))
                else:
                    printstring = "Iter: {:.3f}, Losses:" + mode + ": {:.3f}, SDF: {:.3f}, SDF Non Surface: {:.3f}, Latent Norm: {:.3f}, dsdf_hitMissLoss: {:.3f}, Best Iter: {}"
                    logging.info(printstring.format(iteration,dsdfLoss,sdfSurfLoss,sdfNonSurfLoss,torch.norm(latent.squeeze()).item(),dsdf_hitMissLoss.item() if not(dsdf_hitMiss is None) else dsdf_hitMissLoss ,iterBest))



            if (iteration%50 == 0) and debugVid:
                decoder.fit = False
                dsdf,sdf,_ = decoder.evaluate(data.cuda(),latVec=latent.squeeze().detach().cuda())
                dsdf = dsdf.squeeze()
                sdf = sdf.squeeze()
                decoder.fit = True

                distPred = (dsdf.cpu()+t_sol.cpu()).detach().cpu()
                distPredDebug = distPred.clone()

                pc = data[:,0:3].cpu()+data[:,3:6].cpu()*dsdf.unsqueeze(-1).cpu()
                distPred[(flags==False).cpu() | (torch.abs(sdf)>thresh).cpu().detach() | (torch.norm(pc,dim=1)>1.0).cpu().detach()] = 0

                debug = torch.cat([distancesGT.view((f_height,f_width))/torch.max(torch.max(distancesGT)),(distPred.view((f_height,f_width))/torch.max(torch.max(distPred))).cpu(),(distPredDebug.view((f_height,f_width))/torch.max(torch.max(distPredDebug))).cpu()],dim=1).detach().numpy()
                frame = (debug*255).astype('uint8')
                frame = np.repeat(np.expand_dims(frame,axis=2),3,axis=-1)
                writer.write(frame)
                print(torch.norm(latent.detach().cpu()-decoder.latent_vec_geom.cpu()))
            if iteration == breakIter:
                break
        if debugVid:
            writer.release()
        optimizer.zero_grad()

        with open(fileName+"_averageTime.pkl",'wb') as f:
            pickle.dump(totalTime/numIters,f)
            f.close()
        print("Average Time per iteration: "+str(totalTime/numIters))
        decoder.fit = False
        decoder.latent_vec_geom = latent.detach().squeeze()
        if fireInstance.seperate_lat:
            decoder.latent_vec_ddf = latentDDF.detach().squeeze()
        else:
            decoder.latent_vec_ddf = latent.detach().squeeze()
        
        torch.save(latent.detach(), fileName+".pth")
        if decoder.seperate_lat:
            torch.save(latentDDF.detach(), fileName+"_ddf.pth")
        renderAndSaveVideo(fileName+"_recon",render_width,render_height,decoder,gtMeshPath,thresh=renderThresh,frames=25)
        renderAndSaveImage(fileName+"_recon",render_width,render_height,decoder,gtMeshPath,thresh=renderThresh,inImagePath=inImagePath,saveGt=False)
            
    if not os.path.isfile(fileName+".pth"):
        return
    decoder.latent_vec_geom = torch.load(fileName+".pth")
    decoder.latent_vec_ddf = decoder.latent_vec_geom
    if decoder.seperate_lat:
        decoder.latent_vec_ddf = torch.load(fileName+"_ddf.pth")
    if rendersPaper:
        if decoder.type == "lfn":
            decoder.max_batch = 2**10
        else:
            decoder.max_batch = 2**15
        nframes = 6
        for frame in range(0,nframes):
            th = torch.tensor([-np.pi/6, 2*frame*np.pi/nframes])
            renderAndSaveImage(renderPaperFile+"_recon_"+str(frame),render_width,render_height,decoder,gtMeshPath,thresh=9e-3,inImagePath=None,saveGt=True, th=th)
    try:
        if not skip:
            reconstructAndSaveMesh(fileName+"_recon",accuracyFile,gtMeshPath,decoder,thresh=meshThresh,skipRecon=evaluateOnly and skip,gt_pc_path=gt_pc,evalDDF=True,saveGT=False)
        
    except:
        print("Error Meshing")
    return

def saveColorCodedGT(fileName, gtMeshPath, inImagePath):
    org, rays, raySphereOrg, flags, t_sol, f_height, f_width, _,camIntrinsic, worldMat, _, _ = setupEvalAndGetRays(fileName, gtMeshPath, inImagePath,retGTImg=True, fitTo512Choy=True, returnCameraMats=True)
    data = torch.cat([raySphereOrg,rays],dim=-1).reshape(-1,6)
    orgsRep = torch.zeros(rays.shape) + org
    tmmesh = loadAndNormalizeMesh(gtMeshPath,rotth=0)
    depthScale, hitmiss_gt, distancesGT, _ = saveGroundTruthImages(tmmesh,fileName,orgsRep.flatten().view(-1,3),rays.flatten().view(-1,3),list(raySphereOrg.shape),write_images=True,projMat=camIntrinsic@worldMat)
    return

def renderAndSaveVideo(fileName, f_width, f_height, fireInstance, gtMeshPath, skip=True, thresh=1e-3,frames=100):
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    writer = cv2.VideoWriter(fileName+".mp4", fourcc, 10, (f_width*4, f_height),True)
    gifwriter_depth = imageio.get_writer(fileName+"_depth.gif", mode='I')
    gifwriter_shade = imageio.get_writer(fileName+"_shaded.gif", mode='I')
    gifwriter_gt_shade = imageio.get_writer(fileName+"_shaded_gt.gif", mode='I')
    gifwriter_sdf_shade = imageio.get_writer(fileName+"_shaded_sdf.gif", mode='I')
    gifwriter_error = imageio.get_writer(fileName+"_error.gif", mode='I')
    gifwriter_color = imageio.get_writer(fileName+"_color.gif", mode='I')

    decoder = fireInstance
    if decoder.type == "lfn":
        decoder.max_batch = 2**10
    #setup rotation matrix for camera Rotation
    nFrames = frames
    rots = torch.linspace(0,2,nFrames)
    tmmesh = loadAndNormalizeMesh(gtMeshPath)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topCenter = (int(0.1*f_width),int(0.1*f_height))
    fontScale              = 1
    fontColor              = (0,0,0)
    lineType               = 2

    for r in range(0,rots.shape[0]):
        print("Frame: "+str(r)+ " of " + str(rots.shape[0]))
        # R = rot(np.pi*rots[r],'y')
        R = np.matmul(rot(np.pi*rots[r],'y'),rot(torch.tensor(-np.pi/4).float(),'x'))
        start = time.time()
        org, rays, raySphereOrg, flags, t_sol = setupCameraAndGetRaysG(f_width,f_height,2.5,R,onSphere=True)
        print("Time for camera setup: "+str(time.time()-start))
        data = torch.cat([raySphereOrg,rays],dim=-1).flatten().view(-1,6)
        orgsRep = torch.zeros(rays.shape) + org

        depthScale = 1.0

        depthScale, hitmiss_gt, distancesGT, normalsGT = saveGroundTruthImages(tmmesh,fileName,orgsRep.flatten().view(-1,3),rays.flatten().view(-1,3),list(raySphereOrg.shape))

        t = getFrame(distancesGT,hitmiss_gt)
        orgsRep = orgsRep.flatten().view(-1,3)
        raySphereOrg = raySphereOrg.flatten().view(-1,3)
        pc = orgsRep.detach().cpu().numpy() \
         + np.repeat(np.reshape(distancesGT,(-1,1)),3,axis=1)*(rays.flatten().view(-1,3).detach().cpu().numpy())

        img1 = getShadedImage(list(t_sol.shape),hitmiss_gt,normalsGT,pc,org.detach().cpu().numpy())
        gifwriter_gt_shade.append_data(img1)
        cv2.putText(t,'GT Depth', topCenter, font, fontScale, fontColor, lineType)
        cv2.putText(img1,'GT Geom Shaded', topCenter, font, fontScale, fontColor, lineType)
        gtFrame = np.append(t[:,:,0:3],img1,axis=1)
        # gtFrame = np.append(gtFrame,img1*0,axis=1)


        numIters = 200
        _,_,_, distancesPred, hitmissPred, normals = sphereTrace(fireInstance, fileName, numIters,  \
        data.clone(), raySphereOrg, rays, t_sol, flags, thresh, depthScale,type="sdf",mult=1.0,writeFiles=False)
        distancesPred = distancesPred.detach().cpu().numpy()
        t = getFrame(distancesPred,hitmissPred.detach().cpu().numpy())
        pc = orgsRep.detach().cpu().numpy() \
         + np.repeat(np.reshape(distancesPred,(-1,1)),3,axis=1)*(rays.flatten().view(-1,3).detach().cpu().numpy())

        img1 = getShadedImage(list(t_sol.shape),hitmissPred.detach().cpu().numpy(),normals.detach().cpu().numpy(),pc,org.detach().cpu().numpy())
        gifwriter_sdf_shade.append_data(img1)

        numIters = 1
        #DSDF rendering
        start = time.time()
        _,_,_, distancesPred, hitmissPred, normals = sphereTrace(fireInstance, fileName, numIters,  \
        data.clone(), raySphereOrg, rays, t_sol, flags, thresh, depthScale,type="dsdf",mult=1.0,writeFiles=False)
        print("Time for DSDF Rendering: "+str(time.time()-start))
        distancesPred = distancesPred.detach().cpu().numpy()
        t = getFrame(distancesPred,hitmissPred.detach().cpu().numpy())
        pc = orgsRep.detach().cpu().numpy() \
         + np.repeat(np.reshape(distancesPred,(-1,1)),3,axis=1)*(rays.flatten().view(-1,3).detach().cpu().numpy())



        errImg = errorImage(distancesGT,distancesPred,hitmiss_gt,[f_width,f_height])

        img1 = getShadedImage(list(t_sol.shape),hitmissPred.detach().cpu().numpy(),normals.detach().cpu().numpy(),pc,org.detach().cpu().numpy())
        gifwriter_shade.append_data(img1)
        gifwriter_depth.append_data(t)
        gifwriter_error.append_data(np.flip(errImg,2))

        cv2.putText(t,'Predicted Depth', topCenter, font, fontScale, fontColor, lineType)
        cv2.putText(img1,'Predicted Geom Shaded', topCenter, font, fontScale, fontColor, lineType)

        cv2.putText(errImg,'Prediction Error', topCenter, font, fontScale, (255,255,255), lineType)


        predFrame = np.append(t[:,:,0:3],img1,axis=1)
        predFrame = np.append(predFrame,errImg,axis=1)
        predFrame = np.append(predFrame,gtFrame,axis=1)
        write_frame = predFrame
        writer.write(write_frame)
        

    writer.release()

# from DeepSDF
def deepsdf_convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    thresh=0.0,
    decoder=None,
    writeColors=False
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    verts, faces, normals, values = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=thresh, spacing=[voxel_size] * 3)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])


    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
    return mesh_points
def getGridSDFSamples(decoder, N=256):
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    decoder.max_batch=2**18
    _, tsdf = decoder.evaluate(samples[:,0:3].float(),onlySDF=True)
    sdfvals = tsdf.squeeze().reshape(N,N,N)
    return sdfvals, voxel_origin, voxel_size
def loadNormalizationParameters(gt_pc_path):
    normParamFile = gt_pc_path.split("SurfaceSamples/")
    normParamFile = os.path.join(normParamFile[0],"NormalizationParameters",normParamFile[1][:-4]+".npz")
    normParams = np.load(normParamFile)
    scale = normParams['scale']
    offset = normParams['offset']
    return scale, offset
def reconstructAndSaveMesh(mesh_filename,accuracyFile,gtMeshPath,decoder,N=256,saveGT=True, thresh=0, skipRecon=False,gt_pc_path=None,evalDDF=True, gtData =None, scaletype=""):
    if not decoder.no_net_sdf:
        if (not os.path.isfile(mesh_filename+".ply")) or (not skipRecon):
            sdfvals, voxel_origin, voxel_size = getGridSDFSamples(decoder,N=N)
            meshverts = deepsdf_convert_sdf_samples_to_ply(
                sdfvals.data.cpu(),
                voxel_origin,
                voxel_size,
                mesh_filename+".ply",
                offset=None,
                scale=None,
                thresh=thresh,
                decoder=decoder,
                writeColors=False,
            )
    if evalDDF:
        dirs = tm.sample.sample_surface_sphere(int(2e6))
        sphSamples = tm.sample.sample_surface_sphere(int(2e6))
        batch = decoder.max_batch
        if decoder.type == "lfn":
            decoder.max_batch = 2**10
        else:
            decoder.max_batch = 2**15

        sampleData = np.append(sphSamples,dirs,axis=1)
        sampleData = torch.from_numpy(sampleData).float()
        ddf, evalSDF = decoder.evaluate(sampleData)
        if ddf.shape[1] == 1:
            ddfPoints = sampleData[:,0:3] + ddf*sampleData[:,3:]
            _, tsdf = decoder.evaluate(ddfPoints,onlySDF=True)
            ddfPoints = ddfPoints[(tsdf<0.001).squeeze()]
            evalSDF = tsdf[(tsdf<0.001).squeeze()]
        else:
            ddfPoints = sampleData[:,0:3] + ddf[:,0].unsqueeze(-1)*sampleData[:,3:]
            ddfPoints = ddfPoints[ddf[:,1]>decoder.ddfthresh]
            evalSDF = evalSDF[ddf[:,1]>decoder.ddfthresh]
        decoder.max_batch = batch
        ddfPoints = ddfPoints.numpy()
        ddfPoints = ddfPoints[np.linalg.norm(ddfPoints,axis=-1)<=1.0]
        evalSDF = evalSDF.numpy()
        print(ddfPoints.shape)
        writefile(mesh_filename+"_ddf.obj",ddfPoints)
        
    tmmesh, scalev2, centv2 = loadAndNormalizeMesh(gtMeshPath, returnScale=True)

    if os.path.isfile(accuracyFile):
        accuracyFilePtr = open(accuracyFile, "rb")
        accuracyNumbers = pickle.load(accuracyFilePtr)
        accuracyFilePtr.close()
    else:
        accuracyNumbers = {}
    filetoken = os.path.basename(mesh_filename)

    if not os.path.isfile(gt_pc_path):
        return
    gtPc = tm.load(gt_pc_path,process=False,force="mesh")
    pc_gt = gtPc.vertices

    scale, offset = loadNormalizationParameters(gt_pc_path)

    if scaletype == "prif":
        pc_gt2 = (pc_gt + offset)*scale

        newoffset = (pc_gt2.max(axis=0)+pc_gt2.min(axis=0))/2
        pc_gt2 = pc_gt2 - newoffset
        newscale = np.linalg.norm(pc_gt2,axis=1).max()
        pc_gt2 = pc_gt2/newscale
        newoffset = newoffset/newscale
        pc_gt = np.copy(pc_gt2)
        scale = newscale
        offset = np.copy(newoffset)
    if not decoder.no_net_sdf:
        recon_mesh = tm.load(mesh_filename+".ply",process=False,force="mesh")
        recon_mesh.vertices = recon_mesh.vertices/scale - offset
    else:
        recon_mesh = None
    if not decoder.no_net_sdf:
        results=computeChamfers(pc_gt,recon_mesh,pc_gt.shape[0])
        logging.info("L1: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l1_chamfer"],results["l1_chamfer_gt_to_recon"],results["l1_chamfer_recon_to_gt"]))
        logging.info("L2: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l2_chamfer"],results["l2_chamfer_gt_to_recon"],results["l2_chamfer_recon_to_gt"]))
        accuracyNumbers[filetoken] = results

    if evalDDF:
        results=computeChamfers(pc_gt,ddfPoints/scale - offset,pc_gt.shape[0])

        logging.info("L1: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l1_chamfer"],results["l1_chamfer_gt_to_recon"],results["l1_chamfer_recon_to_gt"]))
        logging.info("L2: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l2_chamfer"],results["l2_chamfer_gt_to_recon"],results["l2_chamfer_recon_to_gt"]))
        accuracyNumbers[filetoken+"_ddf"] = results

        if gtData is not None:
            def computeGTDDFACC(pc_gt,gtData):
                dsdf_data = DDF.data.unpack_sdf_samples_from_ram(gtData, pc_gt.shape[0]).cuda()
                xyzdirs = dsdf_data[:, 0:6]
                dsdf_gt = dsdf_data[:, 6]
                gtDDFPC = xyzdirs[:,:3] +  dsdf_gt.unsqueeze(-1)*xyzdirs[:,3:]
                results=computeChamfers(pc_gt,gtDDFPC.cpu().detach().numpy()/scale - offset,pc_gt.shape[0])

                logging.info("L1: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l1_chamfer"],results["l1_chamfer_gt_to_recon"],results["l1_chamfer_recon_to_gt"]))
                logging.info("L2: Symm: {}, GT2Recon: {}, Recon2GT: {}".format(results["l2_chamfer"],results["l2_chamfer_gt_to_recon"],results["l2_chamfer_recon_to_gt"]))
                return results
            results = computeGTDDFACC(pc_gt,gtData)
            accuracyNumbers[filetoken+"_gt_ddf"] = results
    if not decoder.no_net_sdf:
        evalSDF = evalSDF[np.random.randint(0,high=evalSDF.shape[0],size=pc_gt.shape[0])]
        accuracyNumbers[filetoken+"_sdfddfcons"] = np.abs(evalSDF).mean()
    loadAccFile = open(accuracyFile, "wb")
    pickle.dump(accuracyNumbers, loadAccFile)
    loadAccFile.close()
    if saveGT:
        tmmesh.export(mesh_filename+"_gtMesh.ply",file_type='ply')
    return