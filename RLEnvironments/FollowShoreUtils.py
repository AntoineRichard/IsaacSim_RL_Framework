from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np
import pickle
import cv2
import os

def load(path:str) -> (dict, list, np.array):
    with open(os.path.join(path, 'contours.pkl'),'rb') as f:
        islands = pickle.load(f)
    with open(os.path.join(path, 'objects.pkl'), 'rb') as f:
        objects = pickle.load(f)
    mask = np.load(os.path.join(path, 'mask.npz'))['data']
    return islands, objects, mask

def generateInflatedMap(islands:dict,
                    objects:dict,
                    mask:np.array,
                    d:float=11.0,
                    res:float=0.1,
                    maxgap:float=2.0) -> np.array:
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    r = int(d/res)
    for i in objects:
        cv2.circle(new_mask, list(i['position'].astype(np.int32)), r, 1, -1)
    for i in islands.values():
        for j in i: 
            for k in j:
                cv2.circle(new_mask, k, r, 1, -1)
    gap = int(maxgap/res)
    gap += gap%2    
    kernel = np.zeros([gap,gap], np.uint8)
    kernel = cv2.circle(kernel,(int(gap/2),int(gap/2)),int(gap/2),1,-1)
    new_mask = cv2.dilate(new_mask,kernel) 
    new_mask = cv2.erode(new_mask,kernel) 
    return new_mask

def getContours(mask:np.array,
                islands:dict) -> list:
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = {k:cv2.contourArea(c) for k,c in enumerate(contours[0])}
    A = [(k,v) for k, v in sorted(areas.items(), key=lambda item: item[1])]

    tmp1 = np.zeros_like(mask, dtype=np.int32)
    tmp1 = cv2.drawContours(tmp1, [contours[0][A[-3][0]]], -1, 1, -1)
    err = False
    for key in islands.keys():
        if key == "main":
            continue
        for cnt in islands[key]:
            cnt = np.expand_dims(np.array(cnt),1)
            tmp2 = np.zeros_like(mask, dtype=np.int32)
            tmp2 = cv2.drawContours(tmp2, [cnt], -1, 1, -1)
            diff = (tmp1 - tmp2)
            if np.sum(diff > 0) == 0:
                err = True
                break
    if err:
        contours = [contours[0][A[-2][0]]]
    else:
        contours = [contours[0][A[-2][0]], contours[0][A[-3][0]]]
    return contours

def plot(mask:np.array,
        contours:list,
        island:dict={}) -> None:
    plt.imshow(mask)
    for i in contours:
        plt.plot(i[:,0,0], i[:,0,1],'r')
    for i in island.values():
        for j in i:
            j = np.array(j)
            plt.plot(j[:,0], j[:,1],'k')

def interpolateLine(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.arange(0.002, 0.998, 0.001)
    out = interpolate.splev(unew, tck)
    t = np.arange(unew.shape[0])
    fx = UnivariateSpline(t, out[0], k=4)
    fy = UnivariateSpline(t, out[1], k=4)
    t2 = np.arange(unew.shape[0]*4)/4
    x = fx(t2)
    y = fy(t2)
    x1 = fx.derivative(1)(t2)
    y1 = fy.derivative(1)(t2)
    rz = np.arctan2(y1, x1)
    return x, y, rz, x1, y1

def generateSpawnPositions(path:str,
                           debug:bool=False,
                           d:float=11.0,
                           res:float=0.1,
                           isaac_res:float=100.,
                           maxgap:float=2.0) -> np.array:
    islands, objects, mask = load(path)
    inflated_map = generateInflatedMap(islands, objects, mask, d=d, res=res, maxgap=maxgap)
    c = getContours(inflated_map, islands)
    if debug:
        plt.figure
        plot(inflated_map, c, island=islands)
        #plt.show()
    px = []
    py = []
    rz = []
    for i in c:
        x, y, z, x1, x2 = interpolateLine(c[0])
        px.append(x*res*isaac_res)
        py.append(y*res*isaac_res)
        rz.append(z)
    px = np.concatenate(px,axis=0)
    py = np.concatenate(py,axis=0)
    rz = np.concatenate(rz,axis=0)
    pos = np.stack([px,py,rz],axis=-1)
    return pos

class FollowingSampler:
    def __init__(self, path_to_data:str,
                       ideal_dist:float=10.5,
                       min_dist:float=7.0,
                       max_dist:float=13.0,
                       warmup:int=2.5e5,
                       target_step:int=1e6,
                       alpha:float=1.0,
                       isaac_res:float=100.0,
                       map_res:float=0.1,
                       **kwargs):

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.ideal_dist = ideal_dist
        self.warmup = warmup
        self.target_step = target_step
        self.alpha = alpha
        self.running_reward = None
        self.ideal_poses = generateSpawnPositions(path_to_data,
                                                  d=self.ideal_dist,
                                                  res=map_res,
                                                  isaac_res=isaac_res)
    @staticmethod
    def rsign():
        return np.sign(np.random.rand() - 0.5)
    
    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def sample(self, step:int,
                     reward:float,
                     training:bool,
                     mode:str) -> (list, float, float):
        if mode == 'none':
            x, y, rz = self.sampleRandomIdealPosition()
            t = np.zeros(10)
            p = np.ones(t.shape[0])
            return [x,y,rz], t, p/np.sum(p)
        elif mode == 'random':
            x, y, rz, t, p = self.samplePositionFixedCurriculum(step=self.target_step+1)
            return [x,y,rz], t, p
        else:
            x, y, rz, t, p = self.samplePositionFixedCurriculum(step=step, mode=mode)
            return [x,y,rz], t, p

    def sampleRandomIdealPosition(self) -> (float, float, float):
        idx = np.random.choice(np.arange(self.ideal_poses.shape[0]))
        return self.ideal_poses[idx,0], self.ideal_poses[idx,1], self.ideal_poses[idx,2]

    def samplePositionFixedCurriculum(self, step:int,
                                            mode:str='power',
                                            max_ang_noise:float=np.pi/2) -> (float,
                                                                             float,
                                                                             float,
                                                                             np.array,
                                                                             np.array):
        px, py, rz = self.sampleRandomIdealPosition()
        if step > self.target_step:
            coeff = 1.0
        elif step > self.warmup:
            if mode=='sigmoid':
                t = (step-self.warmup)/(self.target_step-self.warmup)*6 - 3
                coeff = np.tanh(t)/2 + 0.5
            elif mode=='power':
                max_v = 5**self.alpha
                t = (step-self.warmup)/(self.target_step-self.warmup)*5
                coeff = t**self.alpha/max_v
            else:
                raise ValueError('Unknown mode: '+mode)
        else:
            t = np.zeros(10)
            p = np.ones(t.shape[0])
            return px, py, rz, t, p/np.sum(p)
        # get distance distribution
        dist, dist_dist = self.genDualGaussianDist(coeff)
        # apply distance to usv
        dist = np.random.choice(dist,1,p=dist_dist)
        px = px + np.cos(rz+np.pi/2)*dist
        py = py + np.sin(rz+np.pi/2)*dist
        # compensate for the angular error created
        rz = rz + np.arctan2(-dist,5)
        # add some noise on the ideal yaw
        t, p = self.genGaussian(coeff)
        yaw_c = np.random.choice(t,1,p=p)
        rz = rz + yaw_c*max_ang_noise*self.rsign()
        return px[0], py[0], rz[0], t, p

    def genDualGaussianDist(self, coeff:float,
                                sig1:float = 1.,
                                sig2:float = 0.5,
                                eps:float = 0.025) -> (np.array, np.array):
        bmax = self.max_dist - self.ideal_dist
        bmin = self.min_dist - self.ideal_dist
        t = np.arange(bmin, bmax, 0.01)
        mu1 = coeff * bmin
        mu2 = coeff * bmax
        pbmin = int((mu1 - bmin)/0.01)
        pbmax = int((mu2 - bmin)/0.01)
        g1 = self.gaussian(t, mu1, sig1)
        g2 = self.gaussian(t, mu2, sig2)
        w = g1 + g2
        w[pbmin:pbmax] = (w[pbmin:pbmax] < eps)*eps + (w[pbmin:pbmax] > eps)*w[pbmin:pbmax]
        p = w/np.sum(w)
        return t, p

    def genGaussian(self, coeff:float,
                        sig:float = 0.25) -> (np.array, np.array):
        t = np.arange(0,1,0.01)
        g = self.gaussian(t, coeff, sig)
        return t, g/np.sum(g)