"""
Depth Dataset Tool

Tool to create, edit, and visualise the dataset used to
train depth direction decision model.
"""
import cv2
import numpy as np


class DepthDatasetTool:
    def __init__(self):
        self.depth_images = None
        self.depth_labels = None

    def init(self):
        try:
            self.depth_images = np.load("depth_images.npy")
            self.depth_labels = np.load("depth_labels.npy")
        except:
            pass

    def collect(self, new_depth_images, new_depth_labels):
        if self.depth_images is None:
            self.init()
        if self.depth_images is None:
            np.save("depth_images.npy", np.asarray(new_depth_images))
            np.save("depth_labels.npy", np.asarray(new_depth_labels))
        else:
            self.depth_images = np.append(self.depth_images, new_depth_images, axis=0)
            self.depth_labels = np.append(self.depth_labels, new_depth_labels)
            np.save("depth_images.npy", self.depth_images)
            np.save("depth_labels.npy", self.depth_labels)

    def verify(self, show=True):
        if self.depth_images is None:
            self.init()
        if show:
            for i in self.depth_images:
                depth_norm = cv2.normalize(i, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
                cv2.putText(depth_norm, "Direction: {}".format(self.depth_labels[i]), (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow('Tello Depth Camera', depth_norm)
                key = cv2.waitKey(100) & 0xff
                if key == 27:  # ESC
                    break
            cv2.destroyAllWindows()
        if len(self.depth_images) == len(self.depth_labels):
            print('All data points matched!')
        else:
            print('Warning: there are some missing data points!')

    def editor(self):
        if self.depth_images is None:
            self.init()
        i = len(self.depth_images) - 1
        deleted = []
        while 1:
            depth_norm = cv2.normalize(self.depth_images[i], None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            cv2.putText(depth_norm, "Direction: {}".format(self.depth_labels[i]), (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            if i in deleted:
                cv2.putText(depth_norm, "Deleted", (5, 720 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
            cv2.imshow('Tello Depth Camera', depth_norm)
            key = cv2.waitKey(0) & 0xff
            if key == 27:  # ESC
                break
            if key == ord('d') and i != len(self.depth_images) - 1:
                i += 1
            if key == ord('a') and i != 0:
                i -= 1
            if key == ord('q'):
                if i not in deleted:
                    deleted.append(i)
                i -= 1
            if key == ord('z'):
                if i in deleted:
                    deleted.pop(deleted.index(i))
        cv2.destroyAllWindows()
        self.depth_images = np.delete(self.depth_images, deleted, 0)
        self.depth_labels = np.delete(self.depth_labels, deleted, 0)
        np.save("depth_images.npy", self.depth_images)
        np.save("depth_labels.npy", self.depth_labels)

    def size(self):
        if self.depth_images is None:
            self.init()
        direction_count = [0, 0, 0]
        for label in self.depth_labels:
            direction_count[label] += 1
        print('Data labelled forward: ', direction_count[0])
        print('Data labelled left: ', direction_count[1])
        print('Data labelled right: ', direction_count[2])
        print('Dataset size: ', len(self.depth_images))

    def label_count(self):
        if self.depth_images is None:
            self.init()
        unique, counts = np.unique(self.depth_labels, return_counts=True)
        print(dict(zip(unique, counts)))

    def test_resource_usage(self):
        np.save("test.npy", np.zeros((10000,960,1280), dtype=np.float32))
