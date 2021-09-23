import numpy as np

#   *******************************************************************************************************
#   *   |link_length(a_{i-1}) |  link_twist (\alpha_{i-1}) | link_offset (d_i) | joint_angle (\theta_i)    *
#   * 1 |link_length_0        |  link_twist_0              | link_offset_1     | joint_angle_1             *
#   * 2 |link_length_1        |  link_twist_1              | link_offset_2     | joint_angle_2             *
#   * 3 |link_length_2        |  link_twist_2              | link_offset_3     | joint_angle_3             *
#   *******************************************************************************************************


def get_conecting_rod_pos(point, joint_angle_1, joint_angle_2, joint_angle_3):
    link_length_0 = 0
    link_twist_0 = 0
    # link_offset_1 = 10.95       # change due to real situation
    link_length_1 = 0       # change due to real situation
    link_twist_1 = np.pi/2        # change due to real situation
    link_offset_2 = 5.2       # change due to real situation
    link_length_2 = 20.8       # change due to real situation
    link_twist_2 = 0        # change due to real situation
    link_offset_3 = 0       # change due to real situation

    p_3 = np.ones([4, 1])
    p_3[0] = point[0]
    p_3[1] = point[1]
    p_3[2] = point[2]

    t_3_to_2 = get_dh_t_matrix(link_length_2, link_twist_2, link_offset_3, joint_angle_3)
    t_2_to_1 = get_dh_t_matrix(link_length_1, link_twist_1, link_offset_2, joint_angle_2)
    t_1_to_0 = get_dh_t_matrix(link_length_0, link_twist_0, link_offset_1, joint_angle_1)

    p_2 = np.dot(t_3_to_2, p_3)
    t_2_to_0 = np.dot(t_1_to_0, t_2_to_1)

    p_0 = np.dot(t_2_to_0, p_2)
    new_point = np.array([p_0[0][0], p_0[1][0], p_0[2][0]])

    return new_point



def get_dh_t_matrix(link_length, link_twist, link_offset, joint_angle):
    result_matrix=np.zeros([4, 4])
    result_matrix[0, 0] = np.cos(joint_angle)
    result_matrix[0, 1] = -np.sin(joint_angle)
    result_matrix[0, 3] = link_length
    result_matrix[1, 0] = np.sin(joint_angle) * np.cos(link_twist)
    result_matrix[1, 1] = np.cos(joint_angle) * np.cos(link_twist)
    result_matrix[1, 2] = -np.sin(link_twist)
    result_matrix[1, 3] = -np.sin(link_twist) * link_offset
    result_matrix[2, 0] = np.sin(joint_angle) * np.sin(link_twist)
    result_matrix[2, 1] = np.cos(joint_angle) * np.sin(link_twist)
    result_matrix[2, 2] = np.cos(link_twist)
    result_matrix[2, 3] = np.cos(link_twist) * link_offset
    result_matrix[3, 3] = 1
    return result_matrix

