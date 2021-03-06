import threading
from threading import Lock
import time
import numpy as np
import cv2
import try_ra

import serial

# 存探针数据
class Probe:
    def __init__(self):

        # 球铰位置：
        self.world_x = 0.
        self.world_y = 0.
        self.world_z = 0.

        # 探针陀螺仪关于x, y, z 轴旋转角度
        self.world_rot_x = 0.
        self.world_rot_y = 0.
        self.world_rot_z = 0.

        # 探针远端到球铰的相对坐标
        self.end_probe_pos=np.array([0., 0., 0.])

        # 棋盘格原点到球铰相对坐标
        self.grid_probe_pos=np.array([1.2, 2.2, 4.])

        # 探针p3坐标到棋盘格坐标的相对旋转量
        self.world_2_grid_rot_x = 0
        self.world_2_grid_rot_y = 0.

        self.world_2_grid_rot_z = 0.

    def get_point_world_pos(self, p):
        # 返回探针上任意一点p当前的世界坐标
        # p为球铰到点的相对坐标
        R = get_revex(self.world_rot_x, self.world_rot_y, self.world_rot_z)
        relative_pos = -np.reshape(p, [3, 1])
        return np.reshape(np.array([self.world_x, self.world_y, self.world_z]), [3, 1]) - np.dot(R, relative_pos)

    def get_end_world_pos(self):
        # 返回当前探针远端的世界坐标
        return self.get_point_world_pos(self.end_probe_pos)

    def get_grid_world_pos(self):
        # 返回当前探针上棋盘格原点的世界坐标
        return self.get_point_world_pos(self.grid_probe_pos)


# 新建一个探针对象
probe = Probe()


class DH (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        while(1):
            ser = serial.Serial("COM6", 9600, timeout=0.5)
            while (ser):
                x = ser.readline()
                # print(x)
                if (len(x) > 5):
                    y = x.decode()
                    # print(y)
                    arr = y.split(" ")
                    arr1 = list(map(float, arr))
                    joint_angle_1 = (arr1[0] - 683) / 180 * np.pi / 1.433
                    joint_angle_2 = (arr1[1] - 586 + 90 * 1.394) / 180 * np.pi / 1.394
                    joint_angle_3 = (arr1[2] - 862) / 180 * np.pi / 1.411
                    new_pos = try_ra.get_conecting_rod_pos(np.array([1,0,-3.6]), joint_angle_1, joint_angle_2, joint_angle_3)
                    probe.world_x = new_pos[0]
                    probe.world_y = new_pos[1]
                    probe.world_z = new_pos[2]



class Wit_Sensor (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while(1):
            ser2 = serial.Serial("com4", 9600, timeout=0.5)
            print(ser2.is_open)
            print(ser2.name)
            while (1):
                datahex = ser2.read(33)
                # print(datahex)
                DueData(datahex)
                probe.world_rot_x = Angle[0] / 180 * np.pi
                probe.world_rot_y = Angle[1] / 180 * np.pi
                probe.world_rot_z = Angle[2] / 180 * np.pi



ACCData = [0.0] * 8
GYROData = [0.0] * 8
AngleData = [0.0] * 8
FrameState = 0  # 通过0x后面的值判断属于哪一种情况
Bytenum = 0  # 读取到这一段的第几位
CheckSum = 0  # 求和校验位

acc = [0.0] * 3
w = [0.0] * 3
Angle = [0.0] * 3
# 每个点的数据类型
class point:
    def __init__(self, x, y, z, info=0):
        # 点的世界坐标
        self.world_x = x
        self.world_y = y
        self.world_z = z

        # 这个点的肌瘤信息
        self.info = info

    def print(self):
        print(self.world_x,self.world_y,self.world_z)


def get_revex(rot_x,rot_y,rot_z):
    # 知道相对x, y, z 轴的旋转量，获取旋转矩阵
    x=np.zeros([3,3])
    x[0, 0] = 1.
    x[1, 1] = np.cos(rot_x)
    x[1, 2] = np.sin(rot_x)
    x[2, 1] = -np.sin(rot_x)
    x[2, 2] = np.cos(rot_x)
    # print(x)
    y=np.zeros(([3,3]))
    y[1, 1] = 1.
    y[0, 0] = np.cos(rot_y)
    y[2, 2] = np.cos(rot_y)
    y[0, 2] = -np.sin(rot_y)
    y[2, 0] = np.sin(rot_y)
    # print(y)
    z = np.zeros(([3, 3]))
    z[2, 2] = 1.
    z[0, 0] = np.cos(rot_z)
    z[1, 1] = np.cos(rot_z)
    z[1, 0] = -np.sin(rot_z)
    z[0, 1] = np.sin(rot_z)
    # print(z)
    ans = np.dot(x, y)
    ans = np.dot(ans, z)
    return ans

def get_inv_revex(rot_x,rot_y,rot_z):
    # 知道相对x, y, z 轴的旋转量，获取旋转矩阵
    return np.linalg.inv(get_revex(rot_x,rot_y,rot_z))

def change_coordinates(p, R, T):
    # 知道旋转矩阵，平移矩阵，获取点p转化后的坐标
    p_re = np.reshape(p, [3, 1])
    return np.dot(R, p_re+T)


def get_point_grid_pos(p, probe1):
    # 输入点p的世界坐标系中的位置，返回格点坐标系中的位置
    grid_world_pos = probe1.get_grid_world_pos()
    revec_world_2_grid = get_revex(probe1.world_rot_x+probe1.world_2_grid_rot_x, probe1.world_rot_y+probe1.world_2_grid_rot_y, probe1.world_rot_z+probe1.world_2_grid_rot_z)
    trvec_world_2_grid = -grid_world_pos
    # print(trvec_world_2_grid)    # test
    ans = change_coordinates(p, revec_world_2_grid, trvec_world_2_grid)
    return ans


# Press the green button in the gutter to run the script.

    # # world数组用来储存空间点阵信息
    # world = []

R = np.zeros([3, 3])
T = np.array([0, 0, 0])

# 新建一个50*50*50的空间点阵，存入world数组中
# for i in range(50):
#     for j in range(50):
#         for k in range(50):
#             new_point=point(i,j,k)
#             world.append(new_point)

def DueData(inputdata):  # 新增的核心程序，对读取的数据进行划分，各自读到对应的数组里
    global FrameState  # 在局部修改全局变量，要进行global的定义
    global Bytenum
    global CheckSum
    global acc
    global w
    global Angle
    for data in inputdata:  # 在输入的数据进行遍历
        # Python2软件版本这里需要插入 data = ord(data)*****************************************************************************************************
        if FrameState == 0:  # 当未确定状态的时候，进入以下判断
            if data == 0x55 and Bytenum == 0:  # 0x55位于第一位时候，开始读取数据，增大bytenum
                CheckSum = data
                Bytenum = 1
                continue
            elif data == 0x51 and Bytenum == 1:  # 在byte不为0 且 识别到 0x51 的时候，改变frame
                CheckSum += data
                FrameState = 1
                Bytenum = 2
            elif data == 0x52 and Bytenum == 1:  # 同理
                CheckSum += data
                FrameState = 2
                Bytenum = 2
            elif data == 0x53 and Bytenum == 1:
                CheckSum += data
                FrameState = 3
                Bytenum = 2
        elif FrameState == 1:  # acc    #已确定数据代表加速度

            if Bytenum < 10:  # 读取8个数据
                ACCData[Bytenum - 2] = data  # 从0开始
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):  # 假如校验位正确
                    acc = get_acc(ACCData)
                CheckSum = 0  # 各数据归零，进行新的循环判断
                Bytenum = 0
                FrameState = 0
        elif FrameState == 2:  # gyro

            if Bytenum < 10:
                GYROData[Bytenum - 2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    w = get_gyro(GYROData)
                CheckSum = 0
                Bytenum = 0
                FrameState = 0
        elif FrameState == 3:  # angle

            if Bytenum < 10:
                AngleData[Bytenum - 2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    Angle = get_angle(AngleData)
#                    d = acc + w + Angle
                 #   print("a(g):%10.3f %10.3f %10.3f w(deg/s):%10.3f %10.3f %10.3f Angle(deg):%10.3f %10.3f %10.3f" % d)
                CheckSum = 0
                Bytenum = 0
                FrameState = 0


def get_acc(datahex):
    axl = datahex[0]
    axh = datahex[1]
    ayl = datahex[2]
    ayh = datahex[3]
    azl = datahex[4]
    azh = datahex[5]

    k_acc = 16.0

    acc_x = (axh << 8 | axl) / 32768.0 * k_acc
    acc_y = (ayh << 8 | ayl) / 32768.0 * k_acc
    acc_z = (azh << 8 | azl) / 32768.0 * k_acc
    if acc_x >= k_acc:
        acc_x -= 2 * k_acc
    if acc_y >= k_acc:
        acc_y -= 2 * k_acc
    if acc_z >= k_acc:
        acc_z -= 2 * k_acc

    return acc_x, acc_y, acc_z


def get_gyro(datahex):
    wxl = datahex[0]
    wxh = datahex[1]
    wyl = datahex[2]
    wyh = datahex[3]
    wzl = datahex[4]
    wzh = datahex[5]
    k_gyro = 2000.0

    gyro_x = (wxh << 8 | wxl) / 32768.0 * k_gyro
    gyro_y = (wyh << 8 | wyl) / 32768.0 * k_gyro
    gyro_z = (wzh << 8 | wzl) / 32768.0 * k_gyro
    if gyro_x >= k_gyro:
        gyro_x -= 2 * k_gyro
    if gyro_y >= k_gyro:
        gyro_y -= 2 * k_gyro
    if gyro_z >= k_gyro:
        gyro_z -= 2 * k_gyro
    return gyro_x, gyro_y, gyro_z


def get_angle(datahex):
    rxl = datahex[0]
    rxh = datahex[1]
    ryl = datahex[2]
    ryh = datahex[3]
    rzl = datahex[4]
    rzh = datahex[5]
    k_angle = 180.0

    angle_x = (rxh << 8 | rxl) / 32768.0 * k_angle
    angle_y = (ryh << 8 | ryl) / 32768.0 * k_angle
    angle_z = (rzh << 8 | rzl) / 32768.0 * k_angle
    if angle_x >= k_angle:
        angle_x -= 2 * k_angle
    if angle_y >= k_angle:
        angle_y -= 2 * k_angle
    if angle_z >= k_angle:
        angle_z -= 2 * k_angle

    return angle_x, angle_y, angle_z


g = 9.7964

frequency = 10  # Hz
t = 1/frequency

lock=Lock()

x_nums = 14  # x方向上的角点个数
y_nums = 3
grid_len = 0.2
world_point = np.zeros((x_nums * y_nums, 3), np.float32)
world_point[:, :2] = np.mgrid[0:x_nums:1, y_nums-1:-1:-1].T.reshape(-1, 2)
world_point = world_point * grid_len

num_frame_in_calibration = 12
world_position = [world_point] * num_frame_in_calibration
image_position = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

world = []
factor = 0.5
for i in range(3):
    for j in range(3):
        for k in range(3):
            new_point = point(i * factor, j * factor, k * factor)
            # if i**2+j**2+k**2<=25:
            #     new_point.info=1
            # else:
            #     new_point.info=0
            new_point.info = 1
            world.append(new_point)


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]
canvas = np.zeros([height, width, 3], np.uint8)

T1 = DH()
T1.setDaemon(True)
T1.start()

T2 = Wit_Sensor()
T2.setDaemon(True)
T2.start()

while (1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    is_grid, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)
    if is_grid:
        if len(image_position) < num_frame_in_calibration:


            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            image_position.append(exact_corners)

        else:
            image_position.pop(0)

            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            image_position.append(exact_corners)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_position, image_position, gray.shape[::-1],
                                                                   None,
                                                                   None)
            canvas = np.zeros([height, width, 3], np.uint8)
            cv2.circle(canvas, (20, 20), 7, [0, 200, 0], -1)

            # tot_error = 0
            # for i in range(len(world_position)):
            #     imgpoints2, _ = cv2.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)
            #     error = cv2.norm(image_position[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            #     tot_error += error
            # print(tot_error/num_frame_in_calibration)


            lock.acquire()
            for i in world:
                if i.info != 0:

                    grid_coordinate_point = get_point_grid_pos([i.world_x, i.world_y, i.world_z], probe)
                    p_test = np.zeros((1, 3), np.float32)
                    p_test[0][0] = grid_coordinate_point[0]
                    p_test[0][1] = grid_coordinate_point[1]
                    p_test[0][2] = grid_coordinate_point[2]
                    p_um = cv2.UMat(p_test)
                    p_pix, _ = cv2.projectPoints(p_um, rvecs[num_frame_in_calibration - 1],
                                                 tvecs[num_frame_in_calibration - 1],
                                                 mtx, dist)
                    graph_pos_w = (int)(p_pix.get()[0][0][0])
                    graph_pos_h = (int)(p_pix.get()[0][0][1])

                    cv2.circle(canvas, (graph_pos_w, graph_pos_h), 7, [0, 200, 0], -1)

            output = cv2.addWeighted(frame, 0.8, canvas, 0.9, 0)
            cv2.imshow("display_video-cal", output)
            lock.release()

    else:
        cv2.circle(canvas, (20, 20), 7, [0, 0, 200], -1)
        output = cv2.addWeighted(frame, 1, canvas, 0.8, 0)
        cv2.imshow("display_video-cal", output)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

# print(get_point_grid_pos([1, 1, 1], probe))







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
