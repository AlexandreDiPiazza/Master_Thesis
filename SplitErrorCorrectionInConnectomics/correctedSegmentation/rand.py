import numpy as np
import scipy.sparse as sparse
import h5py
from scipy import ndimage

## evaluation code from Zudi -Lin: https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/evaluate.py#L11
__all__ = [
    'get_binary_jaccard',
]


def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index 
    (excluding the zero component of the original labels). Adapted 
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n, int)

    p_ij = sparse.csr_matrix(
        (ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are

# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
if __name__ == '__main__':
    seg_path ='./corrections/SNEMI3D/perfect_correction.npy'

    #seg = np.load(seg_path).astype(np.int64)
    data_type = 'MouseCortex'
    if data_type == 'MouseCortex':
        D1 = '../datasets/MouseCortex/axons_only/'  # image
        gt = np.load(D1 +  'gT_axons_corrected.npy')[375:,:,:] # first part is for train
        #seg = np.load(D1 + 'seg_axons.npy')[375:,:,:]
        path = './corrections/MouseCortex/perfect_correction.npy'
        seg = np.load(path).astype(np.int64)
    elif data_type == 'SNEMI3D':
        data = 'test'
        D1 = '../datasets/SNEMI3D/axons_only/'+data+'/'  # image
        #seg = np.load(D1 + 'seg1_axons.npy').astype(np.int64)
        gt = np.load(D1 + 'gT_axons.npy') 
        path = './corrections/SNEMI3D/LSTM/corrected_seg.npy'
        path = './corrections/SNEMI3D/perfect_correction.npy'
        seg = np.load(path).astype(np.int64)
    are = adapted_rand(seg, gt, all_stats=False)
    print(are)
    
