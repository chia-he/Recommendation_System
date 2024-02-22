import numpy as np
import cv2
import json
import pandas as pd
from os import path 
import xlsxwriter
import matplotlib.pyplot as plt

class Visual:

    def __init__(self, rec_objects, all_objects):
        
        self.rec_efficiency = []
        self.rec_objects = {obj:[] for obj in rec_objects} # recording objects coordinate
        self.rec_targets = [] # recording target coordinate
        self.all_objects = all_objects
        self.result = { obj:0 for obj in rec_objects}

    def object2color(self, shape=3,randseed=None):
        
        if not (randseed is None):
            np.random.seed(randseed)
        colors = [tuple(255 * np.random.rand(shape)) for _ in range(len(self.all_objects))]
        return {obj: color for obj, color in zip(self.all_objects, colors)}

    def drawObjects(self, image, target, boxes,  pred_objects):

        colors = self.object2color(shape=3, randseed=1)
        
        for i in range(len(boxes)):
            if pred_objects[i] in self.rec_objects.keys():
                (x1, y1), (x2, y2) = boxes[i]
                label = pred_objects[i]
                color = colors[label]
                caption = '{}'.format(label)
                # Only Point What Focus on
                if  x1 <= target[0]  <= x2 and y1 <= target[1]  <= y2:
                    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    image = cv2.putText(
                        image, caption, (int(x1), int(y1*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                # Still record everything
                self.rec_objects[pred_objects[i]].append((int(x1), int(y1), int(x2), int(y2)))
        
        return image

    def drawTarget(self, image, target, boxes, probs, landmarks):

        # Run through every face detected
        for box, prob, ld in zip(boxes, probs, landmarks):
            
            # Draw rectangle on image
            cv2.rectangle(image, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(100, 210, 100),thickness=2)

            # Show probability score
            # cv2.putText(image, str(prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            for i in range(len(ld)):
                cv2.circle(image, (round(ld[i][0]), round(ld[i][1])), 3, (50, 0, 200), -1)

            # Draw attention target
            if target[0] != 0:
                start_point = (int((ld[0][0]+ld[1][0])/2), int((ld[0][1]+ld[1][1])/2))
                cv2.line(image, start_point, target, (0, 255, 255), 3)
                cv2.circle(image, target, 10, (0, 255, 255), -1)
                self.rec_targets.append(target)

        return image

    def outputData(self, sheet_name, columns,  datapath='./data/output/'):

        # with open(f"{datapath}output_obj.csv', 'a', newline='') as file:
        #     fileWriter = csv.writer(file)
        #     for key, value in record_objects.items():
        #         fileWriter.writerow([len(value)])
        #         for row in value:
        #             fileWriter.writerow(row)

        # with open(f"{datapath}output_tar.csv', 'a', newline='') as file:
        #     fileWriter = csv.writer(file)
        #     for row in record_target:
        #         fileWriter.writerow(row)

        with open(f"{datapath}output_obj.json", "w") as jsonfile:
            json.dump(self.rec_objects, jsonfile, indent=3)

        with open(f"{datapath}output_tar.json", "w") as jsonfile:
            json.dump(self.rec_targets, jsonfile, indent=3)

        if not path.isfile(f"{datapath}output_eff.xlsx"):
            wb = xlsxwriter.Workbook(f"{datapath}output_eff.xlsx")
            wb.close()

        df = pd.DataFrame(self.rec_efficiency, columns=columns)
        with pd.ExcelWriter(f"{datapath}output_eff.xlsx", mode="a", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name)

    def count(self):

         for key, value in self.rec_objects.items():

            for i in range(len(self.rec_targets)):
                # 1. value[i][0]: x1, value[i][1]: y1, value[i][2]: x2, value[i][3]: y2
                # 2. target[i][0]: x, target[i][1]: y
                if  value[i][0] <= self.rec_targets[i][0]  <= value[i][2] and value[i][1] <= self.rec_targets[i][1]  <= value[i][3]:
                    self.result[key] += 1

    def plotDataStats(self, datapath='./data/output/'):
        
        self.count()
        max_key = max(self.result, key=self.result.get)

        plt.ylabel("Total frames", fontsize=12)
        plt.bar(range(len(self.result)), list(self.result.values()), align='center')
        plt.xticks(range(len(self.result)), list(self.result.keys()))
        plt.savefig(f"{datapath}result.png")
        plt.show()

        with open(f"{datapath}plot_eval.json", "w") as jsonfile:
            json.dump(max_key, jsonfile, indent=2)