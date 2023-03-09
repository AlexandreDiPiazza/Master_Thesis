# Introduction

This is the repository of my Master Thesis carried out in the Visual Computing Group at Harvard University.
Please refer to the pdf of the paper on this repository for more details. 


Axon segmentation is a fundamental task in neuroimaging analysis, enabling the inves-
tigation of neuronal morphology, connectivity, and function. It is challenging due to
the thin, densely packed, and often overlapping nature of axons, as well as the high
anisotropy and artifact defects commonly found in neuroimaging data. As a result, ex-
isting segmentation methods often suffer from many errors. In this paper, we perform a
thorough analysis on split errors corrections in axon segmentation, comparing different
Deep Learning models. The solution is a post-processing technique, that takes as input
an already computed segmentation. The first step involves utilizing high-level representa-
tions of objects with skeletons to identify potential pairs of axons that should be merged.
These pairs are then subjected to classification by a Deep Learning model to determine
if any corrections are required. 


### Installation
All the code is provided in the folder "SplitErrorCorrectionInConnectomics".
Install the librairies with  "requirements.txt"

### Organization of the code
The main file is "SplitErrorCorrectionInConnectomics". The different subfolers are descibed as follows: 
  -  "BinaryClassification": Contains the models usels, the evaluation and the training files.
  -  "correctedSegmentation": Contains the metrics to evaluate the final segmentation, as well as the corrected segmentation data of each model.
  - "dataPreprocessing": Contains the files to take only the axons of the SNEMI3D dataset.
  - "Endpoints": Contains the files to compute all the ground truth split errors and the skeletons to find the endpoints. 
  - "datasets": contains the dataset for the SNEMI3D dataset.
  
Note that the dataset for AxonEM-M is not present, as well as the checkpoints of the models, due to their memory requirements. Please contact me at: "alexandredipiazza98@gmail.com" if you need them.
