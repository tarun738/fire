import torch
import numpy as np

def normalize(vec):
    return vec/torch.norm(vec,dim=-1).unsqueeze(-1)

def getSphereOrgs(org, rays, onSphere=True):
    # To get the ray depths on the unit sphere, find t that solves ||o+td||=1
    o_dot_d = rays[:,:,0]*org[0] + rays[:,:,1]*org[1] + rays[:,:,2]*org[2]
    rootTerm = o_dot_d**2 - (torch.norm(org)**2-1)
    flags = rootTerm>=0
    rootTermCorrected = rootTerm
    rootTermCorrected[flags==False] = 0
    t_sol_1 = -o_dot_d + torch.sqrt(rootTermCorrected)
    t_sol_1[flags==False] = -float("Inf")
    t_sol_2 = -o_dot_d - torch.sqrt(rootTermCorrected)
    t_sol_2[flags==False] = -float("Inf")
    t_sol = torch.min(t_sol_1,t_sol_2)
    t_sol[t_sol==-float("Inf")] = -1
    flags = flags & (t_sol != -1)
    if onSphere==False:
        t_sol[t_sol!=-1] = t_sol[t_sol!=-1]+0.1
    # print("Sanity :"+ str(np.sum(np.sum(t_sol[flags==1]))))
    t_sol_copy = t_sol.clone()
    t_sol_copy[flags==0]=0

    raySphereOrg = rays*t_sol_copy.unsqueeze(-1) + org
    return raySphereOrg, t_sol, flags


class Camera:
    def __init__(self, o, at, up, fov, res, transform=None, invZ=True):

        self.aspect = float(res[1])/res[0]
        self.tanFOV = torch.tan(fov/2).to(o.device)
        self.origin = o
        self.res = res
        if transform is not None:
            self.transform = transform
        else:
            v = at.float()-o
            if invZ==True:
                z = -v/torch.norm(-v)
            else:
                z = normalize(v)
            x = normalize(torch.cross(up,z))
            y = normalize(torch.cross(z,x))
            self.transform = torch.stack([x,y,z],dim=1).transpose(0,1)

    def generateRays(self,xy):
        d = torch.stack([xy[:,:,1]*self.tanFOV, -xy[:,:,0]*self.aspect*self.tanFOV, -torch.ones(xy.shape[0],xy.shape[1]).to(xy.device)],dim=-1)
        d = torch.matmul(d,self.transform)
        d = normalize(d)
        return d
        
def setupCameraAndGetRaysG(width,height,dist,R,onSphere=True,at=torch.tensor([0,0,0]),cam=None):
    
    if cam is None:
        #camera origin, default is on the z-axis
        zero = torch.tensor([0.0]).to(R.device)
        one = torch.tensor([1.0]).to(R.device)
        if not torch.is_tensor(dist):
            dist = torch.tensor([dist]).to(R.device)
        else:
            dist = dist.to(R.device)
        cOrg = torch.matmul(R,torch.cat([zero, zero, dist]).unsqueeze(-1)).transpose(0,1).squeeze()

        # up direction
        up = torch.matmul(R,torch.cat([zero, one, zero]).unsqueeze(-1)).transpose(0,1).squeeze()
        #camera field of view
        fov = torch.tensor([1.0]).to(R.device)
        #get the camera matrix
        cam = Camera(cOrg,at.to(R.device),up,fov,[width,height])
    R = cam.transform
    #setup image pixel locations and get pixel coordinates
    rows = (torch.linspace(1,height,height)-1).to(R.device)
    cols = (torch.linspace(1,width,width)-1).to(R.device)
    x = ((rows+0.5)*2)/height - 1
    y = ((cols+0.5)*2)/width - 1
    xa, yb = torch.meshgrid(x,y)
    xy = torch.cat([xa.unsqueeze(-1),yb.unsqueeze(-1)],dim=-1)
    rays = cam.generateRays(xy)
    org = cam.origin
    # import pdb; pdb.set_trace()
    ############################################################################
    raySphereOrg, t_sol, flags = getSphereOrgs(org, rays, onSphere=onSphere)
    return org, rays, raySphereOrg, flags, t_sol



def shade(lightPos, normal, surfacePoint, org, lightPower=4.0):
    lightPos = np.repeat(np.expand_dims(lightPos,axis=0),normal.shape[0],axis=0)
    org = np.repeat(np.expand_dims(org,axis=0),normal.shape[0],axis=0)
    vertPos = surfacePoint

    lightColor = np.array([[1.0, 1.0, 1.0]]);
    lightColor = np.repeat(lightColor,normal.shape[0],axis=0)
    ambientColor = np.array([[0.01, 0.01, 0.01]]);
    ambientColor = np.repeat(ambientColor,normal.shape[0],axis=0)
    diffuseColor = np.array([[0.75, 0.75, 0.75]]);
    diffuseColor = np.repeat(diffuseColor,normal.shape[0],axis=0)
    # specColor = np.array([1.0, 1.0, 1.0]);
    shininess = 200.0;
    screenGamma = 2.2; # Assume the monitor is calibrated to the sRGB color space
    mode = 1

    lightDir = lightPos - vertPos;
    distance = np.linalg.norm(lightDir,axis=1);
    distance = np.expand_dims(distance,axis=1)
    oneOverDistance = np.repeat(1/distance,3,axis=1)
    lightDir = lightDir*oneOverDistance;
    distance = distance**2
    normal[np.sum(lightDir*normal,axis=1)<0,] = -1*normal[np.sum(lightDir*normal,axis=1)<0,]

    lambertian = np.sum(lightDir*normal,axis=1)
    lambertian[lambertian<0] = 0
    lambertian[lambertian<1e-2] = 1.0
    specular = 0.0;

    # if lambertian > 0.0:

    viewDir = org-vertPos
    viewDir = viewDir*np.repeat(np.expand_dims(1/np.linalg.norm(viewDir,axis=1),axis=1),3,axis=1)

    halfDir = lightDir + viewDir;
    halfDir = halfDir*np.repeat(np.expand_dims(1/np.linalg.norm(halfDir,axis=1),axis=1),3,axis=1)
    specAngle = np.sum(halfDir*normal,axis=1)
    specAngle[specAngle<0] = 0
    specular = np.power(specAngle, shininess);

    # this is phong (for comparison)
    if mode == 2:
      reflectDir = -lightDir - 2*np.dot(-lightDir,normal)*normal;
      specAngle = np.max([np.dot(reflectDir, viewDir), 0.0]);
      # note that the exponent is different here
      specular = np.power(specAngle, shininess/4.0)

    colorLinear = ambientColor + (diffuseColor * np.repeat(np.expand_dims(lambertian,axis=1),3,axis=1) * lightColor * lightPower *oneOverDistance) #+ (specColor * specular * lightColor * lightPower / distance)
    # apply gamma correction (assume ambientColor, diffuseColor and specColor
    # have been linearized, i.e. have no gamma correction in them)
    colorLinear[colorLinear>1.0] = 0.9
    colorGammaCorrected = np.power(colorLinear, (1.0 / screenGamma));
    # use the gamma corrected color in the fragment
    return colorGammaCorrected