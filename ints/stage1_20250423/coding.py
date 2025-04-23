# Problem: code KNN, write two functions train (construct the database) and query (label new point given the database)
import numpy as np
import heapq

class KNN:
    def __init__(self, points: List[List[float]], labels: List[int], K: int):
        # TODO: some checks on input and corner cases
        if len(points) != len(labels):
            print("size mismatch!")
            return
        self.n_sample = len(points)
        self.n_features = len(points[0])
        self.points = points
        self.labels = labels
        self.K = K

    #def train(self, ):
        # TODO
    
    def dist(self, point1: List[float], point2: List[float]) -> float:
        # input check
        if len(point1) != len(point2):
            print("size mismatch")
            return float('-inf')
        distance = 0
        # dist = sum{(xi-yi)^2}
        for i in range(len(point1)):
            distance += (point1[i] - point2[i])**2
        return np.sqrt(distance) # syntax check
        
    
    def query(self, point: List[float]) -> int:
        # TODO
        # check dimension
        if len(point) != self.n_features:
            print("size mismatch")
            return -1
        # find k closest neighbors and store their label in some datastructure
        # develop a min heap to find k nearest neighbors
        # Create distances
        dists_heap = []
        for i in range(self.n_sample):
            dist_to_i = - self.dist(point, self.points[i])
            heapq.heappush(dists_heap, (dist_to_i, i))
            if len(dists_heap > self.K):
                heapq.heappop(dists_heap)
            
        # take consensus
        # TODO