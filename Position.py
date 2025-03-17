import numpy as np
import os
import pandas as pd


# 用户位置随机生成
def generate_gu_positions(num_gu, area_gu, min_spacing, flag_refresh):
    # 初始化
    list_position_gu = []
    list_speed_gu = []
    idx_gu = 0
    num_try = 0
    num_restart = 0

    if flag_refresh:
        flag = 0
    else:
        flag = 1
    position_file = f"position_{num_gu}_gu.csv"
    while flag == 1:
        try:
            with open(position_file):
                matrix_position = pd.read_csv(position_file, header=None)
                matrix_position = np.array(matrix_position)
                for idx_gu in range(num_gu):
                    list_position_gu.append([matrix_position[idx_gu][0], matrix_position[idx_gu][1],
                                             matrix_position[idx_gu][2]])
                    list_speed_gu.append([matrix_position[idx_gu][3], matrix_position[idx_gu][4],
                                          matrix_position[idx_gu][5]])
                area_gu_log = [180, -180, 90, -90]
                for idx_gu in range(num_gu):
                    area_gu_log[0] = min(area_gu_log[0], list_position_gu[idx_gu][0])
                    area_gu_log[1] = max(area_gu_log[1], list_position_gu[idx_gu][0])
                    area_gu_log[2] = min(area_gu_log[2], list_position_gu[idx_gu][1])
                    area_gu_log[3] = max(area_gu_log[3], list_position_gu[idx_gu][1])
                if (abs(area_gu_log[0] - area_gu[0]) > 1.0 or
                        abs(area_gu_log[1] - area_gu[1]) > 1.0 or
                        abs(area_gu_log[2] - area_gu[2]) > 1.0 or
                        abs(area_gu_log[3] - area_gu[3]) > 1.0):
                    print("储存的用户区域不兼容，重新生成")
                    break

                num_gu_list = len(list_position_gu)
                if num_gu_list != num_gu:
                    print("储存的用户数量不兼容，重新生成")
                    break

                spacing_gu_min = 180
                spacing_gu_max = 0
                for idx_gu in range(num_gu):
                    temp_spacing = 180
                    for idx2_gu in range(num_gu):
                        if idx_gu == idx2_gu:
                            continue
                        delta_x = np.radians(np.abs(list_position_gu[idx_gu][0] - list_position_gu[idx2_gu][0]))
                        delta_y = np.radians(np.abs(list_position_gu[idx_gu][1] - list_position_gu[idx2_gu][1]))
                        distance_gu = np.degrees(np.arccos(np.cos(delta_x) * np.cos(delta_y)))
                        temp_spacing = min(temp_spacing, distance_gu)
                    spacing_gu_max = max(spacing_gu_max, temp_spacing)
                    spacing_gu_min = min(spacing_gu_min, temp_spacing)
                if spacing_gu_min < min_spacing:
                    print("储存的用户间隔过近，重新生成")
                    break
                print("读取用户位置成功")
                disp_area_gu = [f"{value:.1f}" for value in area_gu]
                disp_area_gu = '[{}]'.format(', '.join(disp_area_gu))
                output = f"用户范围：{disp_area_gu}，用户数量：{num_gu}，用户间的最小间隔地心角：{spacing_gu_min:.2f}，相邻用户间的最大间隔地心角：{spacing_gu_max:.2f}"
                print(output)
                with open('output.txt', 'a', encoding='utf-8') as file:
                    file.write(f'{output}\n')
                return list_position_gu, list_speed_gu
        except FileNotFoundError:
            print("用户位置读取失败")
            break

    flag = True
    idx_gu = 0
    list_position_gu = []
    list_speed_gu = []
    while flag:
        flag = False
        while idx_gu < num_gu:
            position_gu_lon = np.random.uniform(area_gu[0], area_gu[1])
            position_gu_lat = np.random.uniform(area_gu[2], area_gu[3])
            position_gu_h = 0
            temp_spacing = 180

            for idx2_gu in range(idx_gu):
                delta_x = np.radians(np.abs(position_gu_lon - list_position_gu[idx2_gu][0]))
                delta_y = np.radians(np.abs(position_gu_lat - list_position_gu[idx2_gu][1]))
                distance_gu = np.degrees(np.arccos(np.cos(delta_x) * np.cos(delta_y)))
                temp_spacing = min(temp_spacing, distance_gu)

            num_try += 1
            if num_try > 10:  # 撒点异常，重新撒点
                flag = True
                list_position_gu = []  # 经度/维度/海拔高度
                list_speed_gu = []
                idx_gu = 0
                num_try = 0
                num_restart += 1
                print(f"重新撒点，尝试{num_restart}")
                break

            if temp_spacing >= min_spacing:  # 用户最小间距
                list_position_gu.append([position_gu_lon, position_gu_lat, position_gu_h])
                list_speed_gu.append([0, 0, 0])
                idx_gu += 1
                num_try = 0

    spacing_gu_max = 0
    spacing_gu_min = 180
    for idx_gu in range(num_gu):
        temp_spacing = 180
        for idx2_gu in range(num_gu):
            if idx_gu == idx2_gu:
                continue
            delta_x = np.radians(np.abs(list_position_gu[idx_gu][0] - list_position_gu[idx2_gu][0]))
            delta_y = np.radians(np.abs(list_position_gu[idx_gu][1] - list_position_gu[idx2_gu][1]))
            distance_gu = np.degrees(np.arccos(np.cos(delta_x) * np.cos(delta_y)))
            temp_spacing = min(temp_spacing, distance_gu)
        spacing_gu_max = max(spacing_gu_max, temp_spacing)
        spacing_gu_min = min(spacing_gu_min, temp_spacing)

    output = f"用户范围：{area_gu}，用户数量：{num_gu}，用户间的最小间隔地心角：{spacing_gu_min:.2f}，相邻用户间的最大间隔地心角：{spacing_gu_max:.2f}"
    print(output)
    with open('output.txt', 'a', encoding='utf-8') as file:
        file.write(f'{output}\n')
    matrix_position = np.array(list_position_gu)
    matrix_speed = np.array(list_speed_gu)
    matrix_gu = np.hstack((matrix_position, matrix_speed))
    pd.DataFrame(matrix_gu).to_csv(position_file, index=False, header=False)

    return list_position_gu, list_speed_gu


"""
# 示例使用
num_gu = 150
area_gu = [-10, 10, -10, 10]
min_distance = 1
position_gu = generate_gu_positions(num_gu, area_gu, min_distance)
"""


# 星历读取
def update_position_info(list_satdata, idx_slot, num_slot_second, radius_earth):
    num_sat, _, _ = list_satdata.shape
    idx_slot_second = int(idx_slot % num_slot_second)
    idx_second = int((idx_slot - idx_slot_second) / num_slot_second)
    list_position_sat = []
    list_speed_sat = []

    for idx_sat in range(num_sat):
        position_sat_lon = (idx_slot_second * list_satdata[idx_sat, idx_second + 1, 1] +
                            (num_slot_second - idx_slot_second) * list_satdata[
                                idx_sat, idx_second, 1]) / num_slot_second
        position_sat_lat = (idx_slot_second * list_satdata[idx_sat, idx_second + 1, 0] +
                            (num_slot_second - idx_slot_second) * list_satdata[
                                idx_sat, idx_second, 0]) / num_slot_second
        position_sat_h = (idx_slot_second * list_satdata[idx_sat, idx_second + 1, 2] +
                          (num_slot_second - idx_slot_second) * list_satdata[
                              idx_sat, idx_second, 2]) / num_slot_second
        list_position_sat.append([position_sat_lon, position_sat_lat, position_sat_h])
        lon_sat = np.radians(list_position_sat[idx_sat][0])
        lat_sat = np.radians(list_position_sat[idx_sat][1])
        height_sat = list_position_sat[idx_sat][2]
        d_lon = np.radians(list_satdata[idx_sat, idx_second, 4])
        d_lat = np.radians(list_satdata[idx_sat, idx_second, 3])
        d_height = list_satdata[idx_sat, idx_second, 5]
        speed_lon = d_lon * (radius_earth + height_sat) * np.cos(lat_sat)
        speed_lat = d_lat * (radius_earth + height_sat)
        speed_h = d_height
        speed_x = speed_lon * (-np.sin(lon_sat)) + speed_lat * (-np.sin(lat_sat)) * np.cos(lon_sat) + speed_h * np.cos(
            lat_sat) * np.cos(lon_sat)
        speed_y = speed_lon * np.cos(lon_sat) + speed_lat * (-np.sin(lat_sat)) * np.sin(lon_sat) + speed_h * np.cos(
            lat_sat) * np.sin(lon_sat)
        speed_z = speed_lat * np.cos(lat_sat) + speed_h * np.sin(lat_sat)
        list_speed_sat.append([speed_x, speed_y, speed_z])
        """
        # 验证距离关系
        x_1 = (radius_earth + height_sat) * np.cos(lat_sat) * np.cos(lon_sat)
        y_1 = (radius_earth + height_sat) * np.cos(lat_sat) * np.sin(lon_sat)
        z_1 = (radius_earth + height_sat) * np.sin(lat_sat)
        lon_sat2 = np.radians(list_satdata[idx_sat, idx_second + 1, 1])
        lat_sat2 = np.radians(list_satdata[idx_sat, idx_second + 1, 0])
        height_sat2 = list_satdata[idx_sat, idx_second + 1, 2]
        x_2 = (radius_earth + height_sat2) * np.cos(lat_sat2) * np.cos(lon_sat2)
        y_2 = (radius_earth + height_sat2) * np.cos(lat_sat2) * np.sin(lon_sat2)
        z_2 = (radius_earth + height_sat2) * np.sin(lat_sat2)
        d_x = x_2 - x_1
        d_y = y_2 - y_1
        d_z = z_2 - z_1
        d_h = np.dot([speed_sat[idx_sat, 0], speed_sat[idx_sat, 1], speed_sat[idx_sat, 2]],
                      [x_1, y_1, z_1]) / np.sqrt(x_1 ** 2 + y_1 ** 2 + z_1 ** 2)
        """

    return list_position_sat, list_speed_sat


def calculate_link_info(scenario, list_position_sat, list_position_gu, radius_earth):
    num_sat = len(list_position_sat)
    num_gu = len(list_position_gu)
    distance_2D = np.zeros((num_sat, num_gu))
    distance_3D = np.zeros((num_sat, num_gu))
    angle_dispersion = np.zeros((num_sat, num_gu))
    angle_elevation = np.zeros((num_sat, num_gu))
    type_path = np.zeros((num_sat, num_gu))

    for idx_sat in range(num_sat):
        for idx_gu in range(num_gu):
            d_x = np.radians(abs(list_position_sat[idx_sat][0] - list_position_gu[idx_gu][0]))
            d_y = np.radians(abs(list_position_sat[idx_sat][1] - list_position_gu[idx_gu][1]))
            beta = np.arccos(np.cos(d_x) * np.cos(d_y))
            distance_2D[idx_sat, idx_gu] = beta * radius_earth
            height_sat = list_position_sat[idx_sat][2]
            height_gu = list_position_gu[idx_gu][2]
            distance_3D[idx_sat, idx_gu] = np.sqrt((radius_earth + height_sat) ** 2 + (radius_earth + height_gu) ** 2 -
                                                   2 * (radius_earth + height_sat) * (
                                                           radius_earth + height_gu) * np.cos(beta))
            alpha = np.arccos(((radius_earth + height_gu) ** 2 + distance_3D[idx_sat, idx_gu] ** 2 -
                               (radius_earth + height_sat) ** 2) / (
                                      2 * (radius_earth + height_gu) * distance_3D[idx_sat, idx_gu]))
            if alpha > np.pi / 2:
                angle_elevation[idx_sat, idx_gu] = np.degrees(alpha) - 90
            else:
                angle_elevation[idx_sat, idx_gu] = 0
            angle_dispersion[idx_sat, idx_gu] = 180 - np.degrees(alpha + beta)
            type_path[idx_sat, idx_gu] = los_probability(scenario, angle_elevation[idx_sat, idx_gu])

    return type_path, distance_2D, distance_3D, angle_elevation, angle_dispersion


def generate_visibility_info(matrix_ea, min_ea):
    num_sat, num_gu = matrix_ea.shape
    matrix_v = np.zeros((num_sat, num_gu))
    matrix_v_gu = np.zeros(num_gu)
    matrix_v_sat = np.zeros(num_sat)

    # 循环遍历 num_gu 和 num_sat
    for idx_sat in range(num_sat):
        for idx_gu in range(num_gu):
            if matrix_ea[idx_sat, idx_gu] > min_ea:
                matrix_v[idx_sat, idx_gu] = 1
            matrix_v_gu[idx_gu] += matrix_v[idx_sat, idx_gu]
            matrix_v_sat[idx_sat] += matrix_v[idx_sat, idx_gu]

    return matrix_v, matrix_v_sat, matrix_v_gu


def generate_access_info(matrix_a):
    num_sat, num_gu = matrix_a.shape
    matrix_a_gu = np.zeros(num_gu)
    matrix_a_sat = np.zeros(num_sat)

    # 循环遍历 num_gu 和 num_sat
    for idx_sat in range(num_sat):
        for idx_gu in range(num_gu):
            matrix_a_gu[idx_gu] += matrix_a[idx_sat, idx_gu]
            matrix_a_sat[idx_sat] += matrix_a[idx_sat, idx_gu]

    return matrix_a_sat, matrix_a_gu


def los_probability(scenario, angle_elevation):
    class_ea = round(angle_elevation / 10)
    # 定义 p_los 默认值
    p_los = 0.0
    if scenario == 0:  # Dense Urban Scenario
        if class_ea == 0:
            p_los = 0.0
        elif class_ea == 1:
            p_los = 0.282
        elif class_ea == 2:
            p_los = 0.331
        elif class_ea == 3:
            p_los = 0.398
        elif class_ea == 4:
            p_los = 0.468
        elif class_ea == 5:
            p_los = 0.537
        elif class_ea == 6:
            p_los = 0.612
        elif class_ea == 7:
            p_los = 0.738
        elif class_ea == 8:
            p_los = 0.820
        elif class_ea == 9:
            p_los = 0.981
    elif scenario == 1:  # Urban Scenario
        if class_ea == 0:
            p_los = 0.0
        elif class_ea == 1:
            p_los = 0.246
        elif class_ea == 2:
            p_los = 0.386
        elif class_ea == 3:
            p_los = 0.493
        elif class_ea == 4:
            p_los = 0.613
        elif class_ea == 5:
            p_los = 0.726
        elif class_ea == 6:
            p_los = 0.805
        elif class_ea == 7:
            p_los = 0.919
        elif class_ea == 8:
            p_los = 0.968
        elif class_ea == 9:
            p_los = 0.992
    elif scenario == 2 or scenario == 3:  # Suburban and Rural Scenario
        if class_ea == 0:
            p_los = 0.0
        elif class_ea == 1:
            p_los = 0.782
        elif class_ea == 2:
            p_los = 0.869
        elif class_ea == 3:
            p_los = 0.919
        elif class_ea == 4:
            p_los = 0.929
        elif class_ea == 5:
            p_los = 0.935
        elif class_ea == 6:
            p_los = 0.940
        elif class_ea == 7:
            p_los = 0.949
        elif class_ea == 8:
            p_los = 0.952
        elif class_ea == 9:
            p_los = 0.998

    # 生成一个随机数并根据 p_los 确定 PathType
    temp = np.random.rand()
    if temp < p_los:
        type_path = 1
    else:
        type_path = 0

    return type_path
