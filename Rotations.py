import transforms3d as tf3
import math
import numpy as np

# Declaring different variables representing different orientation notations

quaternion = [math.pi/8, math.pi/8, 0, 0]

quaternion2 = [math.pi/4, math.pi/4, 0, 0]

quaternion2_conjugate = [quaternion2[0], -quaternion2[1], -quaternion2[2], -quaternion2[3]]

euler = tf3.euler.quat2euler(quaternion)

euler2 = [0.0, -1.7, 0.0]

euler3 = [-math.pi/2, 0.0, 0.0]

x = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]])
y = np.array([[1, 0, 0], [2, 0, 0], [1, 2, 3]])

# Transformations from Euler to other notations

axangle = tf3.euler.euler2axangle(euler2[0], euler2[1], euler2[2])

quaternion3 = tf3.euler.euler2quat(euler3[0], euler3[1], euler3[2])

rotmat_ref = tf3.euler.euler2mat(euler3[0], euler3[1], euler3[2])

# print(rotmat_ref)

# print("Euler is {}".format(euler3))

# print("Axangle is {}".format(axangle))

# Quaternion to Rotation Matrix

matrix = tf3.quaternions.quat2mat(quaternion3)

# Rotation Matrix to Axis-Angle

axangle2 = tf3.axangles.mat2axangle(matrix)

# print("Quaternion is {}".format(quaternion))

# print("Matrix is {}, Axangle is {}".format(matrix), axangle2)

# print(matrix)

# print(axangle2)

# Multiplying Matrices using Numpy

z = np.ndarray.__matmul__(x,y)

w = np.ndarray.__matmul__(matrix,matrix.transpose())

# print (z.transpose())


# Checking norms

norm = np.linalg.norm(z)
norm2 = np.linalg.norm(w)

e = tf3.euler.mat2euler(w)

norm3 = np.linalg.norm(e)

print (w)
print (e)
print (norm3)
