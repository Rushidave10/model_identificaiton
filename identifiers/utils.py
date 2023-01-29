import numpy as np
import math

# Clark's transformation ABC to alphabeta
t_32 = 2 / 3 * np.array([[1, -0.5, -0.5],
                         [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)],
                         ])

# Inverse Clark's transformation alphabeta to ABC
t_23 = 2 / 3 * np.array([[1, 0],
                         [-0.5, 0.5 * np.sqrt(3)],
                         [-0.5, -0.5 * np.sqrt(3)]
                         ])


def T(abc):
    """
    ABC to alphabeta
    """
    return np.matmul(t_32, abc)


def T_inv(alphabeta):
    """
    alphabeta to ABC
    """
    return np.matmul(t_23, alphabeta)


def q(U, angle):
    """
    Park's transformation: dq to alphabeta
    """
    cos = math.cos(angle)
    sin = math.sin(angle)
    return np.array([[cos * U[0] - sin * U[1]], [sin * U[0] + cos * U[1]]])


def q_inv(U, angle):
    """
    Inverse Park's transformation: alphabeta to dq
    """
    cos = math.cos(angle)
    sin = math.sin(angle)
    return -sin * U[0] - cos * U[1], cos * U[0] - sin * U[1]


def abc_to_dq(abc, angle):
    return q_inv(T(abc), angle)


def dq_to_abc(dq, angle):

    return T_inv(q(dq, angle))
