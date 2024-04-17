import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh as tm
import torch
def normalizePC(pc, normalize=0.8):
    cent = (np.max(pc,axis=0)+np.min(pc,axis=0))/2
    pc = pc - cent
    norm = np.max(np.linalg.norm(pc,axis=1))
    pc = normalize*pc/norm
    return pc,  norm/normalize, cent
def writefile(filename, points):
    f= open(filename,"w+")
    if points.shape[1] == 3:
        for i in range(points.shape[0]):
            f.write( "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + "\n")
    else:
        for i in range(points.shape[0]):
            color = ((points[i,3:6])*255.0).astype('uint8')
            f.write( "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")
    f.close()
    return
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

def loadAndNormalizeMesh(mesh,normalize=1.0/1.03,rotth=0,rotax='y',returnScale=False):
    tmmesh = tm.load(mesh, force='mesh', process=True)
    if rotth>0:
        tmmesh.vertices = np.matmul(tmmesh.vertices,rot(torch.tensor([rotth*np.pi/180]).squeeze(),rotax).numpy())
    if not normalize == 0.0:
        tmmesh.vertices, scale, cent = normalizePC(tmmesh.vertices, normalize=normalize)
    if returnScale:
        return tmmesh, scale, cent
    else:
        return tmmesh
