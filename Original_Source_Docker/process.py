import SimpleITK
from pathlib import Path


from pandas import DataFrame
import torch
import torchvision
from util.nms_WSI import nms
from PIL import Image
import numpy as np
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

import evalutils

import json

from detection import MyMitosisDetection
# TODO: Adapt to MIDOG 2022 reference algos

# TODO: We have this parameter to adapt the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
execute_in_docker = True

class Mitosisdetection(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/images/histopathology-roi-cropout/") if execute_in_docker else Path("./test/"),
            output_file = Path("/output/mitotic-figures.json") if execute_in_docker else Path("./output/mitotic-figures.json")
        )
        # TODO: This path should lead to your model weights
        if execute_in_docker:
           path_model = "/opt/algorithm/checkpoints/RetinaNetDA.pth"
        else:
            path_model = "./model_weights/RetinaNetDA.pth"

        self.size = 1280
        self.batchsize = 1
        self.detect_thresh = 0.5
        self.nms_thresh = 0.4
        self.level = 0
        # TODO: You may adapt this to your model/algorithm here.
        #####################################################################################
        # Note: As of MIDOG 2022, the format has changed to enable calculation of the mAP. ##
        #####################################################################################

        # Use NMS threshold as detection threshold for now so we can forward sub-threshold detections to the calculations of the mAP

        self.md = MyMitosisDetection(path_model, self.size, self.batchsize, detect_threshold=self.nms_thresh, nms_threshold=self.nms_thresh)
        load_success = self.md.load_model()
        print("Successfully loaded model.")

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)

        print(dict(type="Multiple points", points=scored_candidates, version={ "major": 1, "minor": 0 }))

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple points", points=scored_candidates, version={ "major": 1, "minor": 0 })

    def predict(self, *, input_image: SimpleITK.Image, input_image_file_path) -> DataFrame:
        print('In predict load image and convert to numpy')
        try:
            img = Image.open(input_image_file_path)
            resize_facx = float(112244/img.info['dpi'][0])
            resize_facy = float(112244/img.info['dpi'][1])
            new_shape = (img.size[0]*resize_facx, img.size[1]*resize_facy)
            img = img.resize((int(new_shape[0]+0.5),int(new_shape[1]+0.5)))
            image_data = np.array(img)
            del img
        except:
            print('image_data could not be loaded with PIL.Image and could not be resized.')
            image_data = SimpleITK.GetArrayFromImage(input_image)
            resize_facx, resize_facy = 1, 1

        # TODO: This is the part that you want to adapt to your submission.
        with torch.no_grad():
            result_boxes = self.md.process_image(image_data)

            # TODO perform nms per image:
            #print("All computations done, nms as a last step")
            #result_boxes = nms(result_boxes, self.nms_thresh)

        candidates = list()

        classnames = ['non-mitotic figure', 'mitotic figure']

        if len(result_boxes) == 0:
            print('No predictions found, add a dummy:')
            result_boxes = np.array([[1,1,0.1]])
        else:
            print('Found ', len(result_boxes), 'predictions.')

        for i, detection in enumerate(result_boxes):
            # our prediction returns x_1, y_1, x_2, y_2, prediction, score -> transform to center coordinates
            x, y, pred = detection
            #coord = tuple(((x_1 + x_2) / 2, (y_1 + y_2) / 2))

            # For the test set, we expect the coordinates in millimeters - this transformation ensures that the pixel
            # coordinates are transformed to mm - if resolution information is available in the .tiff image. If not,
            # pixel coordinates are returned.
            world_coords = input_image.TransformContinuousIndexToPhysicalPoint(
                [x/resize_facx,y/resize_facy]
            )


            # Expected syntax from evaluation container is:
            # x-coordinate(centroid),y-coordinate(centroid),0, detection, score
            # where detection should be 1 if score is above threshold and 0 else
            candidates.append([*tuple(world_coords),0,int(pred>self.detect_thresh), pred])

        result = [{"point": c[0:3], "probability": c[4]/1.5, "name": classnames[c[3]] } for c in candidates]
        return result


if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    Mitosisdetection().process()
