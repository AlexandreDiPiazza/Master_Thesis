import numpy as np
from skimage.morphology import skeletonize
import time


def segLabelsforAxonv3(output: np.ndarray, gt: np.ndarray, Z, Y, X) -> dict:
    """
    The idea is we go through the matrices onece, and compute all the statistics that can be useful for later.
    We create two  dictionnaries:
        Dict1: We detect the split errors,
            for each gT axon we say all the IDs in the seg that correspond.
            they keys are the gt axons, the values are the seg ids
            eg: Dict[42] = [1,2,3], for the gt axon 42 we predicted labels 1,2,3 in segmentation
        Dict2: for each output ID we have: the nbr of such IDs N, z_max_ z_min, y_min, y_max,
                                       x_min, x_max, and a dict saying for this ID all the
                                       gt axons in the gt that correspdong, with the number of pixels corresponding
            eg: Dict[3] = [128, 2, 4, 0,10, 5, 15, other_dict, list]. seg_ID 3 has 128 pixels, starting at z = 2 and finishing z = 4
                          start y: 4, finish y : 10
                          start x: 5, finish x: 15
                          other_dict[48] = 88 # 88 of the pixels of this seg_id correspond to axon 48 in GT
                          other_dict[52] = 5  # 5 of the pixels of this seg_id correspond to axon 52 in GT
                          the list contains all the neighbors of this seg_ID
    params:
        Output: Segmentation matrix
        gt: GroundTruth
        Z,H,W : dimmensions of the matrix
    output:
        the two dictionarries described above.
    """
    split_errors = {} #first dict
    output_ids_stats = {} # second dict
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                gt_label = gt[z,y,x]
                output_label = output[z,y,x]
                if output_label == 0:
                    continue # we don't care bout it
                if output_label not in output_ids_stats: # initiate dict value
                    output_ids_stats[output_label] = [0,-np.inf,np.inf, -np.inf, np.inf,
                                                      -np.inf, np.inf,{}, []]  # initialized with high and low for min and max

                output_ids_stats[output_label][0] +=1 # We add 1 to count the nbr of pixels for this seg_id
                output_ids_stats[output_label][1] = max(z, output_ids_stats[output_label][1]) # z max of this ID
                output_ids_stats[output_label][2] = min(z, output_ids_stats[output_label][2]) # z min of this ID
                output_ids_stats[output_label][3] = max(y, output_ids_stats[output_label][3])  # y max of this ID
                output_ids_stats[output_label][4] = min(y, output_ids_stats[output_label][4])  # y min of this ID
                output_ids_stats[output_label][5] = max(x, output_ids_stats[output_label][5])  # x max of this ID
                output_ids_stats[output_label][6] = min(x, output_ids_stats[output_label][6])  # x min of this ID

                neighbors = checkNeighborHood(output, Z, Y, X, z, y, x, output_label)
                l1 = output_ids_stats[output_label][8]
                l1 = list(set(l1).union(set(neighbors))) # add to l1 all the elements of neighbors that are not already there
                output_ids_stats[output_label][8] = l1
                if gt_label == 0:
                    continue  # We don't care about the 0s for the split errors

                if gt_label not in output_ids_stats[output_label][7]: #initliaze the dict
                    output_ids_stats[output_label][7][gt_label] = 0

                output_ids_stats[output_label][7][gt_label] += 1

                if gt_label not in split_errors: # We create the value in the dict for this label
                    split_errors[gt_label] = []
                if output_label not in split_errors[gt_label]: # We have not seen it yet, we add this output_id to the id
                    split_errors[gt_label].append(output_label)
    return split_errors, output_ids_stats


def findEndPointsBis(skel: np.ndarray, z_min, y_min, x_min) -> list:
    """
    find all the endPoints
    :param seg: skeleton matrix
    :return:list of endpoints, each endpoint is (x,y,z)
    """
    indices = np.where(skel == 1)
    endpoints = []
    for i in range(len(indices[0])):
        z = indices[0][i];
        y = indices[1][i];
        x = indices[2][i]
        if check_neighboorhood(skel, z, y, x) == True:
            # print((z,y,x))
            endpoints.append((z + z_min, y + y_min, x + x_min))
    return endpoints

def coordsToMatrix(coords):
    """
    Give a list of coordinates, we fill all those coordinaters with 1. Reconstructing the binary matrix from sparse
    :param coords:
    :return:
    """
    Z = 100;
    Y = 1024;
    X = 1024;
    array = np.zeros(shape=(Z, Y, X))
    N = len(coords[0])
    for i in range(N):
        z = coords[0][i]
        y = coords[1][i]
        x = coords[2][i]
        array[z, y, x] = 1
    return array.astype('uint8')


def findEndPoints(skel: np.ndarray) -> list:
    """
    find all the endPoints
    :param seg: skeleton matrix
    :return:list of endpoints, each endpoint is (x,y,z)
    """
    indices = np.where(skel == 1)
    endpoints = []
    for i in range(len(indices[0])):
        z = indices[0][i];
        y = indices[1][i];
        x = indices[2][i]
        if check_neighboorhood(skel, z, y, x) == True:
            # print((z,y,x))
            endpoints.append((z, y, x))
    return endpoints


def euclDist(p1, p2, treshold):
    """
    Compute the Eucl Dist
    Factor of 5 because of the anisotropy
    """
    points = []
    #Beta = 5 for Snemi, Beta = 2.6 for AxonEM
    beta = 0.2
    for X1 in p1:
        for X2 in p2:
            l2 = np.sqrt((beta * (X1[0] - X2[0])) ** 2 + (X1[1] - X2[1]) ** 2 + (X1[2] - X2[2]) ** 2)
            #print('L2,', l2)
            # if l2 < 70:
            #    print(l2, X1, X2)
            if l2 < treshold:
                # print(l2, X1, X2)
                points.append((X1, X2))
    return points

def checkNeighborHood(arr: np.ndarray, Z, H, W, z_i, y_i, x_i,  axon_ID) -> list:
    """
    Compute the neighborhood in 3D of a coordinate, compute the nbr of uniques IDs that touch this voxel ( ecept backgroun 0)
    :param arr: segmentation 3D array to check
    :param Z,H, W: dimmensions of the array
    :param z_i, y_i, x_i: coordinates of the voxel to check
    :param axon_ID: ID of the voxel
    :return: unique seg ID's that are neighbor with the vocel ( except 0 and the ID itsel)
    """
    neighbors = []
    if y_i < H - 1:
        target = arr[z_i, y_i + 1, x_i]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            # We update the Dict with the corresponding affinity:
            if target not in neighbors:
                neighbors.append(target)
    if y_i > 0:
        target = arr[z_i, y_i - 1, x_i]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            if target not in neighbors:
                neighbors.append(target)
    if x_i > 0:
        target = arr[z_i, y_i, x_i - 1]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            if target not in neighbors:
                neighbors.append(target)

    if x_i < W - 1:
        target = arr[z_i, y_i, x_i + 1]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            if target not in neighbors:
                neighbors.append(target)

    if z_i > 0:
        target = arr[z_i - 1, y_i, x_i]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            if target not in neighbors:
                neighbors.append(target)

    if z_i < Z - 1:
        target = arr[z_i + 1, y_i, x_i]
        if target != 0 and target != axon_ID and target > 0:
            "we found a neighbor"
            if target not in neighbors:
                neighbors.append(target)
    return list(np.unique(neighbors))

def findNeighborsv2(arr: np.ndarray, axon_ID: int) -> list:
    """
    Returns the list of IDS neighbors to axon_ID
    :param arr: np.ndarray, the segmentation
    :param axon_ID: int , target ID
    :return: list[int], list of neighbors
    """
    seg_unique = ((arr == axon_ID) * axon_ID).astype('uint64')
    inner = (seg2aff_v2(seg_unique).min(axis=0) * axon_ID).astype('uint64')
    # Outer bounday of the object
    outter = seg_unique - inner
    indices = np.where(outter == axon_ID)
    Z, H, W = np.shape(arr)  # Z: z ; H: y; W: x
    nbr_elem = len(indices[0])
    intersection1 = []
    intersection2 = []
    neighbors = []
    for i in range(nbr_elem):
        z_i = indices[0][i]
        y_i = indices[1][i]
        x_i = indices[2][i]
        if y_i < H - 1:
            target = arr[z_i, y_i + 1, x_i]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                # We update the Dict with the corresponding affinity:
                if target not in neighbors:
                    neighbors.append(target)

        if y_i > 0:
            target = arr[z_i, y_i - 1, x_i]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                if target not in neighbors:
                    neighbors.append(target)

        if x_i > 0:
            target = arr[z_i, y_i, x_i - 1]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                if target not in neighbors:
                    neighbors.append(target)

        if x_i < W - 1:
            target = arr[z_i, y_i, x_i + 1]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                if target not in neighbors:
                    neighbors.append(target)

        if z_i > 0:
            target = arr[z_i - 1, y_i, x_i]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                if target not in neighbors:
                    neighbors.append(target)

        if z_i < Z - 1:
            target = arr[z_i + 1, y_i, x_i]
            if target != 0 and target != axon_ID and target > 0:
                "we found a neighbor"
                if target not in neighbors:
                    neighbors.append(target)
    return neighbors


def findNeighborsLongRange(arr: np.ndarray, axon_ID: int, long_range_z: int, long_range_xy) -> list:
    """
    Returns the list of IDS neighbors to axon_ID
    :param arr: np.ndarray, the segmentation
    :param axon_ID: int , target ID
    :param long_range_z: range to check the neighbors in z direction
    :param long_range_xy: range to check the neighbor in xy direction
    :return: list[int], list of neighbors
    """
    seg_unique = ((arr == axon_ID) * axon_ID).astype('uint64')
    inner = (seg2aff_v2(seg_unique).min(axis=0) * axon_ID).astype('uint64')
    # Outer bounday of the object
    outter = seg_unique - inner
    indices = np.where(outter == axon_ID)
    Z, H, W = np.shape(arr)  # Z: z ; H: y; W: x
    nbr_elem = len(indices[0])
    neighbors = []
    for i in range(nbr_elem):
        z_i = indices[0][i]
        y_i = indices[1][i]
        x_i = indices[2][i]
        for z_range in range(1, long_range_z + 1):
            for xy_range in range(1, long_range_xy + 1):
                if y_i < H - xy_range:
                    target = arr[z_i, y_i + xy_range, x_i]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        # We update the Dict with the corresponding affinity:
                        if target not in neighbors:
                            neighbors.append(target)

                if y_i > xy_range - 1:
                    target = arr[z_i, y_i - xy_range, x_i]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        if target not in neighbors:
                            neighbors.append(target)
                if x_i > xy_range - 1:
                    target = arr[z_i, y_i, x_i - xy_range]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        if target not in neighbors:
                            neighbors.append(target)

                if x_i < W - xy_range:
                    target = arr[z_i, y_i, x_i + xy_range]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        if target not in neighbors:
                            neighbors.append(target)

                if z_i > z_range - 1:
                    target = arr[z_i - z_range, y_i, x_i]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        if target not in neighbors:
                            neighbors.append(target)

                if z_i < Z - z_range:
                    target = arr[z_i + z_range, y_i, x_i]
                    if target != 0 and target != axon_ID and target > 0:
                        "we found a neighbor"
                        if target not in neighbors:
                            neighbors.append(target)
    return neighbors


def seg2aff_v2(seg: np.ndarray,
               dz: int = 1,
               dy: int = 1,
               dx: int = 1,
               padding: str = 'edge') -> np.array:
    # Calaulate long range affinity. Output: (affs, z, y, x)

    shape = seg.shape
    z_1 = slice(dz, -dz)
    y_1 = slice(dy, -dy)
    x_1 = slice(dx, -dx)

    z_2 = slice(None, dz)
    y_2 = slice(None, dy)
    x_2 = slice(None, dx)

    z_3 = slice(None, -2 * dz)
    y_3 = slice(None, -2 * dy)
    x_3 = slice(None, -2 * dx)

    z_4 = slice(2 * dz, None)
    y_4 = slice(2 * dy, None)
    x_4 = slice(2 * dx, None)

    z_5 = slice(-dz, None)
    y_5 = slice(-dy, None)
    x_5 = slice(-dx, None)

    if seg.ndim == 3:
        aff = np.zeros((3,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dz, dz), (dy, dy), (dx, dx)), 'edge')
            # print(seg_pad.shape)
            aff[2] = (seg_pad[z_1, y_1, x_3] == seg_pad[z_1, y_1, x_4]) * \
                     (seg_pad[z_1, y_1, x_3] != 0) * (seg_pad[z_1, y_1, x_4] != 0)
            aff[1] = (seg_pad[z_1, y_3, x_1] == seg_pad[z_1, y_4, x_1]) * \
                     (seg_pad[z_1, y_3, x_1] != 0) * (seg_pad[z_1, y_4, x_1] != 0)
            aff[0] = (seg_pad[z_3, y_1, x_1] == seg_pad[z_4, y_1, x_1]) * \
                     (seg_pad[z_3, y_1, x_1] != 0) * (seg_pad[z_4, y_1, x_1] != 0)

        else:
            aff[2, :, :, x_1] = (seg[:, :, x_3] == seg[:, :, x_4]) * \
                                (seg[:, :, x_3] != 0) * (seg[:, :, x_4] != 0)
            aff[1, :, y_1, :] = (seg[:, y_3, :] == seg[:, y_4, :]) * \
                                (seg[:, y_3, :] != 0) * (seg[:, y_4, :] != 0)
            aff[0, z_1, :, :] = (seg[z_3, :, :] == seg[z_4, :, :]) * \
                                (seg[z_3, :, :] != 0) * (seg[z_4, :, :] != 0)
            if padding == 'replicate':
                aff[2, :, :, x_2] = (seg[:, :, x_2] != 0).astype(aff.dtype)
                aff[1, :, y_2, :] = (seg[:, y_2, :] != 0).astype(aff.dtype)
                aff[0, z_2, :, :] = (seg[z_2, :, :] != 0).astype(aff.dtype)
                aff[2, :, :, x_5] = (seg[:, :, x_5] != 0).astype(aff.dtype)
                aff[1, :, y_5, :] = (seg[:, y_5, :] != 0).astype(aff.dtype)
                aff[0, z_5, :, :] = (seg[z_5, :, :] != 0).astype(aff.dtype)

    elif seg.ndim == 2:
        aff = np.zeros((2,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dy, dy), (dx, dx)), 'edge')
            # print(seg_pad.shape)
            aff[1] = (seg_pad[y_1, x_3] == seg_pad[y_1, x_4]) * \
                     (seg_pad[y_1, x_3] != 0) * (seg_pad[y_1, x_4] != 0)
            aff[0] = (seg_pad[y_3, x_1] == seg_pad[y_4, x_1]) * \
                     (seg_pad[y_3, x_1] != 0) * (seg_pad[y_4, x_1] != 0)

        else:
            aff[1, :, x_1] = (seg[:, x_3] == seg[:, x_4]) * \
                             (seg[:, x_3] != 0) * (seg[:, x_4] != 0)
            aff[0, y_1, :] = (seg[y_3, :] == seg[y_4, :]) * \
                             (seg[y_3, :] != 0) * (seg[y_4, :] != 0)
            if padding == 'replicate':
                aff[1, :, x_2] = (seg[:, x_2] != 0).astype(aff.dtype)
                aff[0, y_2, :] = (seg[y_2, :] != 0).astype(aff.dtype)
                aff[1, :, x_5] = (seg[:, x_5] != 0).astype(aff.dtype)
                aff[0, y_5, :] = (seg[y_5, :] != 0).astype(aff.dtype)

    return aff


def check_neighboorhood(skel: np.ndarray, z: int, y: int, x: int) -> bool:
    """
    Check if a point has 1 neighboor or more
    If it has 1 neighboor retuns True, else False
    :param skel: Skeleton matrix ( Z,Y,X, containning 0 or 1
    :param z:  z coorodinate
    :param y:  y coordinate
    :param x:  x cooredinate
    :return: True if <= 1 neighbor, False if > 1 neighbor
    """
    count = 0
    Z, H, W = np.shape(skel)
    if y < H - 1 and skel[z, y + 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if y < H - 1 and x > 0 and skel[z, y + 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False

    if y < H - 1 and x < W - 1 and skel[z, y + 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if x > 0 and skel[z, y, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if x < W - 1 and skel[z, y, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if y > 0 and skel[z, y - 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if y > 0 and x > 0 and skel[z, y - 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if y > 0 and x < W - 1 and skel[z, y - 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False

    if z > 0 and y < H - 1 and skel[z - 1, y + 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and y < H - 1 and x > 0 and skel[z - 1, y + 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and y < H - 1 and x < W - 1 and skel[z - 1, y + 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and skel[z - 1, y, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and x > 0 and skel[z - 1, y, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and x < W - 1 and skel[z - 1, y, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and y > 0 and skel[z - 1, y - 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and y > 0 and x > 0 and skel[z - 1, y - 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z > 0 and y > 0 and x < W - 1 and skel[z - 1, y - 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False

    if z < Z - 1 and y < H - 1 and skel[z + 1, y + 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and y < H - 1 and x > 0 and skel[z + 1, y + 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and y < H - 1 and x < W - 1 and skel[z + 1, y + 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and skel[z + 1, y, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and x > 0 and skel[z + 1, y, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and x < W - 1 and skel[z + 1, y, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and y > 0 and skel[z + 1, y - 1, x] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and y > 0 and x > 0 and skel[z + 1, y - 1, x - 1] == 1:
        count += 1
        if count >= 2:
            return False
    if z < Z - 1 and y > 0 and x < W - 1 and skel[z + 1, y - 1, x + 1] == 1:
        count += 1
        if count >= 2:
            return False
    return True