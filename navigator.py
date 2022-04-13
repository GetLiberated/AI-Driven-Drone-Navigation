"""
Navigator Class

Provides the safest positional direction to aid drone's navigation.
"""
from numpy import argmax, exp

# Constants
CAM_WIDTH = 960
CAM_HEIGHT = 720

# calculate the softmax of a vector
def softmax(vector):
    e = exp(vector)
    return e / e.sum()


class Navigator:
    def __init__(self):
        # self.direction = [0, 0, 0, 0]  # Forward, Left, Right, Stop
        self.direction = [0, 0, 0, 0, 0]  # Forward, Left, Right, Up, Down
        self.near_obj = None
        self.near_obj_name = None

    def get_direction(self, objects, depth):
        self.direction = [1, 0, 0, 0, 0]
        self.near_obj = None
        self.near_obj_name = None

        if len(objects) > 0:
            for i, result in objects.iterrows():
                area = (result['xmax'] - result['xmin']) * (result['ymax'] - result['ymin'])
                if area > ((CAM_WIDTH * CAM_HEIGHT) * 0.4):
                    self.near_obj = result
                    self.near_obj_name = result['name']

            if self.near_obj is not None:
                self.direction[0] = 1 - self.near_obj['confidence']
                xratio = (self.near_obj['xmax'] - self.near_obj['xmin']) / CAM_WIDTH
                yratio = (self.near_obj['ymax'] - self.near_obj['ymin']) / CAM_HEIGHT

                if xratio >= 0.9:
                    if self.near_obj['ymax'] - CAM_HEIGHT / 2 > CAM_HEIGHT / 2 - self.near_obj['ymin']:
                        # If object is mostly on the bottom side, go up
                        self.direction[3] = self.near_obj['confidence']
                    else:
                        # If object is mostly on the top side, go down
                        self.direction[4] = self.near_obj['confidence']
                else:
                    if self.near_obj['xmax'] - CAM_WIDTH / 2 > CAM_WIDTH / 2 - self.near_obj['xmin']:
                        # If object is mostly on the right side, go left
                        self.direction[1] = self.near_obj['confidence']
                    else:
                        # If object is mostly on the left side, go right
                        self.direction[2] = self.near_obj['confidence']

        depth_idx = argmax(depth)
        depth_prob = depth[0][depth_idx]
        if depth_prob > self.direction[depth_idx]:
            self.direction[depth_idx] = depth_prob
            if self.near_obj is None and depth_idx > 0:
                self.direction[0] = 1 - depth_prob

        direction_idx = argmax(self.direction)

        if depth_prob == self.direction[direction_idx]:
            self.near_obj_name = 'Wall'

        return direction_idx, self.near_obj_name, str(softmax(self.direction))
