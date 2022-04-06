"""
Floorplan coordinates and labels 
"""
import re
from matplotlib import pyplot as plt, image as mpimg
import os
import numpy as np
import pandas as pd


class Floorplan:
    """
    path names as expected in db files
    """

    # FIXME:
    # if in floorplan
    if "Floorplan" == os.getcwd().split("/")[-1]:
        IMG_DIR = "images"
    else: # in in seth
        IMG_DIR = "Mapping/images"
    # NOTE: what if outside seth?
    
    # names of paths
    BASEMENT = "engr0"
    OFFICE = "engr1"
    PATHS = ["engr0", "engr1"]

    # Metrics that a database for a path should pass
    # define metrics for each path
    # see check_metrics()
    METRIC_ENGR0 = {
        "num_rp": 61,
        "shape": (366, -1),
        "min_rp": 6
    }
    METRIC_ENGR1 = {
        "num_rp": 48,
        "shape": (288, -1),
        "min_rp": 6
    }


    # Scale of each floorplan based on image
    # in px
    SCALE_BASEMENT = 14.2
    SCALE_OFFICE = 17 


    def get_scale(self, floorplan):
        if floorplan == self.BASEMENT:
            return self.SCALE_BASEMENT
        elif floorplan == self.OFFICE:
            return self.SCALE_OFFICE
        else:
            print("invalid input")
            raise KeyError

    def get_metric(self, name):
        if name == self.BASEMENT:
            return self.METRIC_ENGR0
        elif name == self.OFFICE:
            return self.METRIC_ENGR1
        else:
            print("invalid name")
            raise KeyError

    
    def get_coords(self, path_name):
        """
        get a numpy array with 3 columns 
        first two cols are x y
        last columns is label
        """
        plans = {
            self.BASEMENT: self.basement_coords,
            self.OFFICE: self.office_coords
        }

        return plans[path_name]

    
    def show_floorplan(self, path_name):

        plans = {
            self.BASEMENT: self.show_basement_map,
            self.OFFICE: self.show_office_map
        }

        # call the function in dict
        # will raise key error if fails
        plans[path_name]()
        
        
    @property
    def basement_coords(self):
        """
        get coordinates for the basement as dataframe
        """
        # start pos label 0
        rps = [[545, 260, 0]]

        # move down by 38 units
        for i in range(38):
            x = rps[-1][0]
            y = rps[-1][1] + self.SCALE_BASEMENT 
            label = rps[-1][2] + 1
            rps.append([x, y, label])

        # move left by 19 units
        for i in range(19):
            x = rps[-1][0] - self.SCALE_BASEMENT
            y = rps[-1][1] 
            label = rps[-1][2] + 1
            rps.append([x, y, label])

        # move up by 3 units
        for i in range(3):
            x = rps[-1][0] 
            y = rps[-1][1] - self.SCALE_BASEMENT
            label = rps[-1][2] + 1
            rps.append([x, y, label])


        rps = pd.DataFrame(rps, columns=["x", "y", "label"])

        rps["label"] = rps["label"].astype(int)
        
        return rps

        
    @property
    def office_coords(self):

        start_pos = [350, 260, 0]
        rps = [start_pos,]

        # move up 8 units
        for i in range(8):
            x = rps[-1][0] 
            y = rps[-1][1] - self.SCALE_OFFICE
            label = rps[-1][2] + 1
            rps.append([x, y, label])
        
        # move right 19 units
        for i in range(19):
            x = rps[-1][0] + self.SCALE_OFFICE
            y = rps[-1][1]
            label = rps[-1][2] + 1
            rps.append([x, y, label])

        # move down 3 units
        for i in range(3):
            x = rps[-1][0] 
            y = rps[-1][1] + self.SCALE_OFFICE
            label = rps[-1][2] + 1
            rps.append([x, y, label])

        # move right 17 units
        for i in range(17):
            x = rps[-1][0] + self.SCALE_OFFICE 
            y = rps[-1][1]
            label = rps[-1][2] + 1
            rps.append([x, y, label])
            
        # convert to df and fix label column format
        rps = pd.DataFrame(rps, columns=["x", "y", "label"])
        rps["label"] = rps["label"].astype(int)
            
        return rps

            
    def __plot_floorplan__(self, img_path, rps):

        # plot background
        img = mpimg.imread(img_path)
        plt.imshow(img)

        # plot rps
        plt.scatter(rps[:, 0], rps[:, 1], marker="s", c="C1", s=9)

        # reurn the plot object
        return plt

    
    def show_basement_map(self, save=None):
        img_path = os.path.join(self.IMG_DIR, self.BASEMENT + ".png")

        rps = self.basement_coords.values

        plt = self.__plot_floorplan__(img_path, rps)

        if save is not None:
            plt.savefig(save, dpi=640)

        plt.show()

        
    def show_office_map(self, save=None):
        img_path = os.path.join(self.IMG_DIR, self.OFFICE + ".png")

        rps = self.office_coords.values

        plt = self.__plot_floorplan__(img_path, rps)

        if save is not None:
            plt.savefig(save, dpi=640)

        plt.show()



if __name__ == "__main__":
    # Floorplan().show_basement_map(save="Mapping/path/basement.png")
    # Floorplan().show_office_map(save="Mapping/path/office.png")

    print(Floorplan().office_coords.shape)
    print(Floorplan().basement_coords.shape)
