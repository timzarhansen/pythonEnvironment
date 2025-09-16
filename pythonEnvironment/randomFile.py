import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_transform(matrix):
  """Converts a 4x4 transformation matrix to x, y, z, roll, pitch, yaw."""
  translation = matrix[:3, 3]
  rotation = R.from_matrix(matrix[:3,:3])
  euler = rotation.as_euler('xyz', degrees=False)
  return np.asarray([translation[0], translation[1], translation[2], euler[0], euler[1], euler[2]])


a = np.array([[ 9.99876632e-01,  1.57073540e-02,  0.00000000e+00,  2.37038395e+01],
              [-1.57073540e-02,  9.99876632e-01,  0.00000000e+00,  4.21836882e-01],
              [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

b = np.array([[ 0.99935171, -0.03600222,  0.        , 34.30259565],
              [ 0.03600222,  0.99935171,  0.        ,  1.95724961],
              [ 0.        ,  0.        ,  1.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])

result = np.dot(a, np.linalg.inv(b))
print(matrix_to_transform(result))



