import trimesh as tm
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh as tm

def computeChamfers(gtMeshSamples, reconMesh, num_mesh_samples, scale=None, cent=None):
    # gtSamples = tm.sample.sample_surface(gtMesh, num_mesh_samples)[0]
    np.random.shuffle(gtMeshSamples)
    gtSamples = gtMeshSamples[:num_mesh_samples, :]
    if type(reconMesh).__module__ == np.__name__:
        reconSamples = reconMesh[np.random.randint(0,high=reconMesh.shape[0],size=num_mesh_samples)]
    else:
        reconSamples = tm.sample.sample_surface(reconMesh, num_mesh_samples)[0]
    
    if scale is not None:
        gtSamples = gtSamples*scale
        reconSamples = reconSamples*scale
    if cent is not None:
        gtSamples = gtSamples + cent
        reconSamples = reconSamples + cent
        
    
    # one direction
    gt_kd_tree = KDTree(gtSamples)
    one_distances, one_vertex_ids = gt_kd_tree.query(reconSamples)
    one_distances_L1 = np.mean(one_distances)
    recon_to_gt_L2 = np.mean(one_distances**2)

    # other direction
    recon_kd_tree = KDTree(reconSamples)
    two_distances, two_vertex_ids = recon_kd_tree.query(gtSamples)
    two_distances_L1 = np.mean(two_distances)
    gt_to_recon_L2 = np.mean(two_distances**2)

    # max_side_length = np.max(bb_max - bb_min)
    f_score_threshold = 0.01 # deep structured implicit functions sets tau = 0.01
    # L2 chamfer
    l2_chamfer = (gt_to_recon_L2 + recon_to_gt_L2)
    # F-score
    f_completeness = np.mean(one_distances <= f_score_threshold)
    f_accuracy = np.mean(two_distances <= f_score_threshold)
    f_score = 100*2 * f_completeness * f_accuracy / (f_completeness + f_accuracy) # harmonic mean
    accuracyDict = {}
    accuracyDict["f_score"] = f_score
    accuracyDict["l2_chamfer"] = l2_chamfer*1000
    accuracyDict["l2_chamfer_gt_to_recon"] = gt_to_recon_L2*1000
    accuracyDict["l2_chamfer_recon_to_gt"] = recon_to_gt_L2*1000
    accuracyDict["l1_chamfer"] = (two_distances_L1 + one_distances_L1)/2
    accuracyDict["l1_chamfer_gt_to_recon"] = two_distances_L1
    accuracyDict["l1_chamfer_recon_to_gt"] = one_distances_L1
    return accuracyDict
    