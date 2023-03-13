import numpy as np 


"""
We found lots of examples on the GT for the Mouse Cortex where the data was not correctly annotated.
Some pairs of axons in the GT have different labels when they belong to the same instance, eg (labelA, labelB) should actually onlt have labelA, because labelA 
and labelB belong to the same instance.
We thus correct these errors here
"""

gt = np.load('../datasets/MouseCortex/axons_only/gT_axons_ori.npy')
print(len(np.unique(gt)))


GT_pairs_errors = [
    (3735, 4421),(1474, 5081),(6405, 4132),(1647, 7685),(7421, 2273),(5725,408),(4936,5611),
    (7085, 1105),(3362, 6902),(3332, 7448),(4835, 653),(5095, 7303),(6060, 87),(5743, 6994),
    (1220, 7463),(147, 4695), (3521, 6474),(5817, 997),(4868, 462),(3317, 6611),(543, 5930),
    (2990, 7305), (2074, 6742), (4536, 5562), (1873, 5978), (7555, 1207), (7029, 5499),
    (3917, 3716), (1944, 6755), (4110, 5881), (6764, 1944), (7077, 6032), (4715, 1676),
    (5002, 3088), (3795, 6403), (7324, 4327), (6301 ,3909), (7560, 6841), (4805, 4979),
    (7738, 3334), (7582, 5937), (2438, 5965),(5520, 6423),(4570, 5276),(3121,7058),(2009, 7314),
    (3846, 6848)
    ]
    
for elem in GT_pairs_errors:
    elem1 = elem[0]
    elem2 = elem[1]
    np.place(gt, gt == elem2, elem1) # put same label for the same pair

np.save('../datasets/MouseCortex/axons_only/gT_axons_corrected.npy', gt)