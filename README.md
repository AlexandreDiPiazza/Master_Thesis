# Project Title

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


### Prerequisites

install the librairies with  "requirements.txt"

### 