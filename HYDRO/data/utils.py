import numpy as np
import cv2
from skimage import transform as trans


# Left eye, right eye, nose, left mouth, right mouth
arcface_src = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)
unpg_src = np.expand_dims(unpg_src, axis=0)


def estimate_norm(lmk, image_size=112, shrink_factor=1.0, offset_y=0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112

    arcface_src_l = arcface_src.copy()
    arcface_src_l[:, :, 1] += offset_y
    src = arcface_src_l * shrink_factor + (1 - shrink_factor) * 56
    src = src * src_factor

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def inverse_estimate_norm(lmk, t_lmk, image_size=112, shrink_factor=1.0, offset_y=0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    arcface_src_l = arcface_src.copy()
    arcface_src_l[:, :, 1] += offset_y
    src = arcface_src_l * shrink_factor + (1 - shrink_factor) * 56
    src = src * src_factor
  
    for i in np.arange(src.shape[0]):
        tform.estimate(t_lmk, lmk)
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=256, shrink_factor=1.0, offset_y=0):
    """
    Align and crop the image based of the facial landmarks in the image. The alignment is done with
    a similarity transformation based of source coordinates.
    :param img: Image to transform.
    :param landmark: Five landmark coordinates in the image.
    :param image_size: Desired output size after transformation.
    :param shrink_factor: Shrink factor that shrinks the source landmark coordinates. This will include more border
    information around the face. Useful when you want to include more background information when performing face swaps.
    The lower the shrink factor the more of the face is included. Default value 1.0 will align the image to be ready
    for the Arcface recognition model, but usually omits part of the chin. Value of 0.0 would transform all source points
    to the middle of the image, probably rendering the alignment procedure useless.

    If you process the image with a shrink factor of 0.85 and then want to extract the identity embedding with arcface,
    you simply do a central crop of factor 0.85 to yield same cropped result as using shrink factor 1.0. This will
    reduce the resolution, the recommendation is to processed images to output resolutions higher than 112 is using
    Arcface. This will make sure no information is lost by resampling the image after central crop.
    :return: Returns the transformed image.
    """
    M, pose_index = estimate_norm(landmark, image_size, shrink_factor=shrink_factor, offset_y=offset_y)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def transform_landmark_points(M, points):
    lmk_tran = np.insert(points, 2, values=np.ones(5), axis=1)
    transformed_lmk = np.dot(M, lmk_tran.T)
    transformed_lmk = transformed_lmk.T

    return transformed_lmk





