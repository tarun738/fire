## changing opengl to egl to ensure that the code works with ssh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh as tm
import sys
import time

def sampleDDFsFromMeshOnSphere(mesh, numSamples=50000, numDirs=16, NmSamples=500000):
   # compute the number of points to sample near the surface and away from the surface
   N=int(numSamples)
   D=int(numDirs)
   # number of points that definitely hit the mesh surface
   hit = int(D)
      #densely sample points on mesh for directions
   meshSamples,_ = tm.sample.sample_surface(mesh,NmSamples)
   #randomly choose from sampled mesh points for directions
   dirIdxs = np.random.randint(0,NmSamples,N*D)
   hitDirs = meshSamples[dirIdxs,:]
   # reshape so that we have D*R directions for each of the N samples
   hitDirs = np.reshape(hitDirs,(N,hit,3))
   #sample N points on sphere from which we compute directional distances
   samples =tm.sample.sample_surface_sphere(N)
   # randomLengths = np.random.uniform(0,1,N)
   # samples = samples*np.repeat(np.expand_dims(randomLengths,axis=1),3,axis=1)
   #repeat each of N samples D times so we can do (sample, direction)
   tmp = np.expand_dims(samples,axis=1)
   tmp = np.repeat(tmp,hitDirs.shape[1],axis=1)
   # from sampled mesh points, finally compute directions and normalize them
   hitDirs = hitDirs-tmp
   hitDirsN = np.linalg.norm(hitDirs,axis=2)
   hitDirsN = np.repeat(np.expand_dims(hitDirsN,axis=2),3,axis=2)
   hitDirs = hitDirs/hitDirsN

   # finally append samples in front of directions to have (samples,dirs)
   samplesRep = np.repeat(np.expand_dims(samples,axis=1),D,axis=1)
   dsdfSamples = np.append(samplesRep,hitDirs,axis=2)
   # append two more fields (samples, dirs, distance, hits/misses 1/0)
   tempZeros = np.zeros((dsdfSamples.shape[0],dsdfSamples.shape[1],3))
   dsdfSamples = np.append(dsdfSamples,tempZeros,axis=2)
   dsdfSamples = np.reshape(dsdfSamples,(N*D,1,dsdfSamples.shape[2]))
   dsdfSamples = np.squeeze(dsdfSamples)
   startTime = time.time()
   # find ray mesh intersections
   intersector = tm.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
   iloc, iIdxs , intersectFaceIds = intersector.intersects_location(dsdfSamples[:,:3],
                       dsdfSamples[:,3:6], multiple_hits=False)
   if iIdxs.size > 0:
      dsdfSamples[iIdxs,6] = np.linalg.norm(dsdfSamples[iIdxs,:3]-iloc,axis=1)
      dsdfSamples[iIdxs,7] = 1
   dsdfSamples = dsdfSamples[iIdxs,:]
   normals = mesh.face_normals[intersectFaceIds.squeeze()]

   sdfSamples = []
   print("Time to finish = " + str(time.time()-startTime) + "s")

   return dsdfSamples


def sampleMissingDDFsFromMeshOnSphere(mesh, numSamples=50000, numDirs=16, NmSamples=500000):
   # compute the number of points to sample near the surface and away from the surface
   N=int(numSamples)*4
   D=int(numDirs)
   # number of points that definitely hit the mesh surface
   miss = int(D)
   
   #sample N points on sphere from which we compute directional distances
   samples =tm.sample.sample_surface_sphere(N)
   missDirs = tm.sample.sample_surface_sphere(N*D)
   # reshape so that we have D*R directions for each of the N samples
   missDirs = np.reshape(missDirs,(N,miss,3))

   # finally append samples in front of directions to have (samples,dirs)
   samplesRep = np.repeat(np.expand_dims(samples,axis=1),D,axis=1)
   dsdfSamples = np.append(samplesRep,missDirs,axis=2)
   # append two more fields (samples, dirs, distance, hits/misses 1/0)
   tempZeros = np.zeros((dsdfSamples.shape[0],dsdfSamples.shape[1],3))
   dsdfSamples = np.append(dsdfSamples,tempZeros,axis=2)
   dsdfSamples = np.reshape(dsdfSamples,(N*D,1,dsdfSamples.shape[2]))
   dsdfSamples = np.squeeze(dsdfSamples)
   startTime = time.time()
   # find ray mesh intersections
   intersector = tm.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
   iloc, iIdxs , intersectFaceIds = intersector.intersects_location(dsdfSamples[:,:3],
                       dsdfSamples[:,3:6], multiple_hits=False)
   if iIdxs.size > 0:
      dsdfSamples[iIdxs,6] = np.linalg.norm(dsdfSamples[iIdxs,:3]-iloc,axis=1)
      dsdfSamples[iIdxs,7] = 1
   dsdfSamples = dsdfSamples
   
   dsdfSamples = dsdfSamples[dsdfSamples[:,7]==0,:]
   idxs = np.random.randint(0,dsdfSamples.shape[0],int(numSamples))
   dsdfSamples = dsdfSamples[idxs,:]
   return dsdfSamples