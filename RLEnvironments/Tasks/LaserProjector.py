import numpy as np
import cv2

class LaserProjector:
    def __init__(self, ideal_dist=10, theta_min=-np.pi, theta_max=np.pi, num_points=360):
        # Initialize grid size
        self.ideal_dist_ = ideal_dist
        self.safe_to_shore_ = max(self.ideal_dist_*0.4,2.0)
        self.local_grid_front_size_ = min(self.ideal_dist_*1.2, 10.0)
        self.local_grid_rear_size_ = min(self.ideal_dist_/5.0, 1.0)
        self.local_grid_side_size_ = self.ideal_dist_*1.2
        self.local_grid_res_ = 1.0/self.ideal_dist_

        #image size
        hsize = int((self.local_grid_front_size_+self.local_grid_rear_size_)/self.local_grid_res_)
        wsize = int((self.local_grid_side_size_)*2/self.local_grid_res_)
        self.size_ = max(hsize, wsize)
        x_offset_ = self.size_/2#int(self.local_grid_side_size_ / self.local_grid_res_)
        y_offset_ = self.size_/2#int(self.local_grid_rear_size_ / self.local_grid_res_)
        self.offset_ = np.array([x_offset_, y_offset_], dtype=np.int32)
        self.kernel_size_ = int(self.ideal_dist_/self.local_grid_res_)
        self.safe_kernel_size_ = int(self.safe_to_shore_/self.local_grid_res_)

        # crop the ranges for the costmap
        self.minx = max(0,int(self.size_/2 - self.local_grid_rear_size_/self.local_grid_res_))
        self.maxx = min(int(self.size_/2 + self.local_grid_front_size_/self.local_grid_res_), self.size_)
        self.miny = max(0,int(self.size_/2 - self.local_grid_side_size_/self.local_grid_res_))
        self.maxy = min(int(self.size_/2 + self.local_grid_side_size_/self.local_grid_res_), self.size_)

        # Initialiaze theta array
        theta = np.linspace(theta_min, theta_max, num_points)
        c = np.cos(theta)
        s = np.sin(theta)
        self.cs = np.stack([c,s]).T
        self.initializeMaps()

    def initializeMaps(self):
        if (self.kernel_size_ & 1) == 0:
                self.kernel_size_ += 1
        if (self.safe_kernel_size_ & 1) == 0:
                self.safe_kernel_size_ += 1
        self.dreamers_map_ = np.zeros([self.size_, self.size_,3], np.uint8)
        self.blue_map_ = np.zeros([self.size_, self.size_,3], np.uint8)
        self.blue_map_[:,:,0] = 255
        self.map_ = np.zeros([self.size_, self.size_], np.uint8)
        self.zero_map_ = np.zeros([self.size_, self.size_], np.uint8)

    def genContourMap(self, points):
        for point in points:
            cv2.circle(self.map_, (point[1],point[0]), self.kernel_size_, 255, -1)
            cv2.circle(self.dreamers_map_, (point[1],point[0]), self.safe_kernel_size_, (0,0,255), -1)
        contours, _ = cv2.findContours(self.map_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.dreamers_map_, contours, -1, (0,0,0), 7)
        return cv2.resize(self.dreamers_map_[self.minx:self.maxx,self.miny:self.maxy], (64,64), cv2.INTER_NEAREST)

    def refreshMaps(self):
        self.map_ = self.zero_map_.copy()
        self.dreamers_map_ = self.blue_map_.copy()

    def projectLaser(self, ranges):
        ranges[ranges>=20] = 100
        # Takes between 1.5 and 3.0ms for 360 points
        points = np.einsum('ij,i->ij', self.cs, ranges/self.local_grid_res_)
        points = self.offset_ + points.astype(np.int32)
        self.refreshMaps()
        return self.genContourMap(points)

