"""Local PnP observability, conditional on fixed map points (not true accuracy)."""
import cv2
import numpy as np


def pose_geometry_diagnostics(camera_points, camera_matrix, distortion, rms_px):
    points=np.asarray(camera_points,float).reshape(-1,3)
    result={'model':'camera-local-projection-hessian', 'conditional_on_fixed_map_points':True,
            'used_for_acceptance':False,'valid':False}
    try:
        if len(points)<4 or not np.isfinite(points).all() or np.any(points[:,2]<=0):return result
        _,J=cv2.projectPoints(points,np.zeros(3),np.zeros(3),camera_matrix,distortion)
        J=J[:,:6];depth=float(np.median(points[:,2]))
        units=np.array([1.,1.,1.,depth,depth,depth]);normalized=J*units
        _,s,v=np.linalg.svd(normalized,full_matrices=False)
        rank=int(np.sum(s>s[0]*1e-8));result['rank']=rank
        if rank<6:return result
        sigma=max(.5,float(rms_px)/np.sqrt(2.))
        covariance=np.einsum('ij,j,jk->ik',v.T,(sigma/s)**2,v)
        covariance*=units[:,None]*units[None,:]
        result.update(valid=True,normalized_condition_number=float((s[0]/s[-1])**2),
            position_std_m=float(np.sqrt(np.trace(covariance[3:,3:]))),
            rotation_std_deg=float(np.degrees(np.sqrt(np.trace(covariance[:3,:3])))),pixel_sigma_px=sigma)
    except (ValueError,np.linalg.LinAlgError,cv2.error):pass
    return result
