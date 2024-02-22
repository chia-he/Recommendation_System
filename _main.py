from config import *
from py_utils import visualization
from _fetch import fetchData
import _recommender
import os
import json

if __name__ == '__main__':

    '''
        Only execute fetchData(), outputData(), _plot.eval() when 
        output folder is empty.
    '''

    filepath = "./data/output/output_tar.json"

    if os.path.isfile(filepath):

        json.loads(open(filepath, "r").read())

    else:
        # inputfile in './data/input/', accept : video
        inputName = 'input2.avi'
        
        # output to './data/output/'
        outputName = 'Test_High'
        
        # what you want to focus on，accept objects : COCO_INSTANCE_CATEGORY_NAMES
        # reference : config.py
        rec_objects = ['bottle', 'bowl', 'cup']

        # To record and present which target (person) focus on which object (commodity) 
        record = visualization.Visual(rec_objects, COCO_INSTANCE_CATEGORY_NAMES)

        # (__,__,__, stride,__) : to skip (s-1)/s frames, make it faster.
        # (__,__,__,__,mode) : High or Low Resolution 
        rd = fetchData(inputName, outputName, record, stride=None, mode='High')
        # Dump json file for data and performance statistic. 
        rd.outputData(outputName, columns=["Frame", "FPS", "Time Cost"])
        # Plot statistic data
        rd.plotDataStats()
    
    # Show the website
    #_recommender.startWeb()
