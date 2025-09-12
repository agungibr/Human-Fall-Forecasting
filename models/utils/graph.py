import torch

# Source: https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
class Graph:
    def __init__(self, layout='openpose'):
        # For now we just hardcode 17 joints (OpenPose COCO)
        if layout != 'openpose':
            raise ValueError("Only 'openpose' layout is supported for now")
        self.num_node = 17
        self.A = self._get_adjacency_matrix()

    def _get_adjacency_matrix(self):
        # Build adjacency: self-loops + neighbor connections
        self_link = [(i, i) for i in range(self.num_node)]
        inward = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),
                  (8,9),(9,10),(10,11),(8,12),(12,13),(13,14),(0,15),(0,16)]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        edge = self_link + neighbor

        A = torch.zeros((1, self.num_node, self.num_node))
        for i, j in edge:
            A[0, i, j] = 1
        return A
