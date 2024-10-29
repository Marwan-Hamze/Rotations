import transforms3d as tf3
import math
import numpy as np

# Declaring different variables representing different orientation notations

quaternion = [1, 0, 0, 0]

quaternion2 = [1, 0, 0, 0]

quaternion2_conjugate = [quaternion2[0], -quaternion2[1], -quaternion2[2], -quaternion2[3]]

quaternion4 = [0.5556245380733142, -0.5370365782417992, 0.38832455314425374, 0.5020728311176293]

euler = tf3.euler.quat2euler(quaternion)

euler2 = [0.0, -1.7, 0.0]

euler3 = [math.pi/2, math.pi/2, math.pi/2]

x = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]])
y = np.array([[1, 0, 0], [2, 0, 0], [1, 2, 3]])

# Transformations from Euler to other notations

axangle = tf3.euler.euler2axangle(euler2[0], euler2[1], euler2[2])

quaternion3 = tf3.euler.euler2quat(euler3[0], euler3[1], euler3[2])

rotmat_ref = tf3.euler.euler2mat(euler3[0], euler3[1], euler3[2])

# print(quaternion3)

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

# print (w)
# print (e)
# print (norm3)

# Difference in Quaternions just like in the RL project

Diff = 1 * (1 - np.inner(quaternion, quaternion4) ** 2)

print("This is the error in quaternions used in RL: {}".format(Diff))

print("This is the reward of quaternions difference: {}".format(0.50 * np.exp(-Diff)))
