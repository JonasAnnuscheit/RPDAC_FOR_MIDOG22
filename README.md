# Contribution to the MIDOG22 challenge by using the RP-DAC
This is the repository for the paper "Radial Prediction Domain Adaption Classifier for the MIDOG 2022 challenge".

## Abstract
This paper describes our contribution to the MIDOG 2022 challenge for detecting mitotic cells. One of the major problems to be addressed in the MIDOG 2022 challenge is the robustness under the natural variance that appears for real-life data in the histopathology field.  To address the problem, we use an adapted YOLOv5s model for object detection in conjunction with a new Domain Adaption Classifier (DAC) variant, the Radial-Prediction-DAC, to achieve robustness under domain shifts. In addition, we increase the variability of the available training data using stain augmentation in HED color space.

## Keywords
- MIDOG 22 contribution
- mitosis detection
- Domain Adaption Classifier
- Stain Augmentation
- Test Time Augmentation


# Neural Network Architecture
The neural network architecture can be found [here](https://github.com/JonasAnnuscheit/RPDAC_FOR_MIDOG22/blob/main/Original_Source/models/MYyolov5s0.1concatT.yaml).

# Trainable Concat-Layer
We have replaced the Concat layer of the YOLOv5s model with a trainable version. The trainable version of the Concat layer allows us to have a hyperparamter for each channel. This hyperparamter goes into a sigmoid function such that the value is between 0 and 1 and is multiplied by the channel. This allows the information of the residual connection to be reduced. The source code of this layer can be found [here](https://github.com/JonasAnnuscheit/RPDAC_FOR_MIDOG22/blob/main/Original_Source/models/common.py#L430-L444).

# Radial Prediction Domain Adaption Classifier-Layer (RP-DAC)
The implementation of the RP-DAC-Layer can be found [here](https://github.com/JonasAnnuscheit/RPDAC_FOR_MIDOG22/blob/main/Original_Source/models/common.py#L42-L79) and the Loss can be found [here](https://github.com/JonasAnnuscheit/RPDAC_FOR_MIDOG22/blob/main/Original_Source/utils/loss.py#L13-L44).


# Augmentation in the HED-Space for the MIDOG-DS
The source code for the DA can be found [here](https://github.com/JonasAnnuscheit/RPDAC_FOR_MIDOG22/blob/main/Original_Source/utils/augmentations.py#L57-L95).

# Contact
If you have any questions, contact me at jonas.annuscheit@htw-berlin.de.

# Acknowledgements
The authors acknowledge the financial support by the Federal Ministry of Education and Research of
Germany (BMBF) in the project deep.Health (project number 13FH770IX6).
