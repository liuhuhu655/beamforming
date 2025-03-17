import numpy as np
from scipy.integrate import quad

def generate_channel(time_system, scenario, position_sat, position_gu, num_ant_sat_hori, num_ant_sat_vert, num_ant_sat_pol,
                    num_ant_gu_hori, num_ant_gu_vert, num_ant_gu_pol, d, ka, speed_sat, speed_gu, freq_carrier, radius_earth,
                    distance, elevation_angle, type_path, type_freq):
    # 初始化
    num_ant_sat = num_ant_sat_hori * num_ant_sat_vert * num_ant_sat_pol
    num_ant_gu = num_ant_gu_hori * num_ant_gu_vert * num_ant_gu_pol
    c = 299792458
    lamda = c / freq_carrier
    min_ea = 10
    loss_path = 0.0
    matrix_h = np.zeros((num_ant_sat, num_ant_gu), dtype=complex)

    # 跳过用户对于卫星不可见的情况
    if elevation_angle < min_ea:
        loss_path = 300
        return matrix_h, loss_path

    # 大尺度参数向量[s_SF s_K s_DS s_asd s_asa s_zsd s_zsa]
    param_large_scale = generate_large_scale_param(scenario, elevation_angle, type_freq, type_path)

    # Step3: 计算路径损耗
    # 基本路径损耗loss_b
    loss_fs = 32.45 + 20 * np.log10(freq_carrier / 1e9) + 20 * np.log10(distance)  # 自由空间损耗(dB)
    loss_sf = param_large_scale[0]  # 阴影衰落损耗(dB)
    loss_cl = generate_clutter_loss(scenario, elevation_angle, type_freq, type_path)  # 杂波损耗(dB)
    loss_b = loss_fs + loss_sf + loss_cl
    # 其他损耗（待建模）
    loss_g = 0  # 大气吸收损耗（对于低于6GHz的频率，通常可以忽略）（待添加）
    loss_s = 0  # 闪烁损耗（待添加）
    loss_e = 0  # 室内附加损耗（待添加）
    # 路径损耗
    loss_path = loss_b + loss_g + loss_s + loss_e

    # Step4：生成大尺度参数
    aod = 0
    zod = 90 + elevation_angle
    aoa_los = 180
    zoa_los = 90 - elevation_angle
    delay_spread = param_large_scale[2]
    asd = 0
    zsd = 0
    asa = param_large_scale[4]
    zsa = param_large_scale[6]
    kf_dB = param_large_scale[1]
    kf = 10 ** (kf_dB / 10)

    # Step5: 生成簇延迟
    num_cluster = generate_cluster_num(scenario, elevation_angle, type_freq, type_path)  # 簇的数量
    num_ray_cluster = generate_ray_num()  # 簇内光束数量
    exponential_delay_distribution = np.zeros(num_cluster)
    delay_ray = generate_delay_scaling_param(scenario, elevation_angle, type_freq, type_path)
    x_n_1 = np.random.uniform(0, 1, num_cluster)
    for idx_cluster in range(num_cluster):
        exponential_delay_distribution[idx_cluster] = -delay_ray * delay_spread * np.log(x_n_1[idx_cluster])  # 指数延迟分布

    delay_sorted = np.sort(exponential_delay_distribution)  # 排序
    delay_cluster = delay_sorted - delay_sorted[0]  # 标准化延迟(升序)
    if type_path == 1:  # LOS径
        k_scaling = 0.7705 - 0.0433 * kf_dB + 0.0002 * kf_dB ** 2 + 0.000017 * kf_dB ** 3  # 启发式确定的莱斯K因子相关缩放常数
        delay_cluster = delay_cluster / k_scaling  # 缩放延迟

    # Step6: 生成簇功率
    power_assignment = np.zeros(num_cluster)
    std_shadow = generate_shadowing_std()
    z_n = np.random.normal(0, std_shadow, num_cluster)
    sum_power = 0
    for idx_cluster in range(num_cluster):
        power_assignment[idx_cluster] = (np.exp(-delay_cluster[idx_cluster] * (delay_ray - 1)
                                                / delay_ray / delay_spread) * 10 ** (-z_n[idx_cluster] / 10))
        sum_power += power_assignment[idx_cluster]

    power_cluster = power_assignment / sum_power  # 归一化功率
    if type_path == 1:  # LOS径
        power_cluster = power_cluster / (kf + 1)
        power_cluster[0] += kf / (kf + 1)

    max_power = np.max(power_cluster)
    power_cluster[power_cluster < 10 ** (-25 / 10) * max_power] = 1e-30

    # Step7: 生成方位角和仰角的到达角和离开角
    # 生成簇的AOA/aod
    aod_cluster = np.full(num_cluster, aod)
    aoa_cluster = np.zeros(num_cluster)
    c_phi = generate_scaling_factor_4_azimuth_angle(num_cluster)
    x_n_2 = np.random.uniform(-1, 1, num_cluster)
    y_n_2 = np.random.normal(0, (asa / 7) ** 2, num_cluster)
    if type_path == 1:  # LOS
        c_phi *= 1.1035 - 0.028 * kf_dB - 0.002 * kf_dB ** 2 + 0.0001 * kf_dB ** 3

    angle = 2 * (asa / 1.4) * np.sqrt(-np.log(power_cluster / max_power)) / c_phi
    for idx_cluster in range(num_cluster):
        if type_path == 1:  # LOS
            aoa_cluster[idx_cluster] = (x_n_2[idx_cluster] * angle[idx_cluster] + y_n_2[idx_cluster] -
                                        (x_n_2[0] * angle[0] + y_n_2[0]) + aoa_los)
        elif type_path == 0:  # NLOS
            aoa_cluster[idx_cluster] = x_n_2[idx_cluster] * angle[idx_cluster] + y_n_2[idx_cluster] + aoa_los

    # 生成簇的ZOA/zod
    zod_cluster = np.full(num_cluster, zod)
    zoa_cluster = np.zeros(num_cluster)
    c_theta = generate_scaling_factor_4_zenith_angle(num_cluster)
    x_n_3 = np.random.uniform(-1, 1, num_cluster)
    y_n_3 = np.random.normal(0, (zsa / 7) ** 2, num_cluster)
    if type_path == 1:  # LOS
        c_theta *= 1.3086 + 0.0339 * kf_dB - 0.0077 * kf_dB ** 2 + 0.0002 * kf_dB ** 3
    angle = -zsa * np.log(power_cluster / max_power) / c_theta
    for idx_cluster in range(num_cluster):
        if type_path == 1:  # LOS
            zoa_cluster[idx_cluster] = (x_n_3[idx_cluster] * angle[idx_cluster] + y_n_3[idx_cluster] -
                                        (x_n_3[0] * angle[0] + y_n_3[0]) + zoa_los)
        elif type_path == 0:  # NLOS
            zoa_cluster[idx_cluster] = x_n_3[idx_cluster] * angle[idx_cluster] + y_n_3[idx_cluster] + zoa_los

    # 生成路径的方向角
    d_angel_ray = RayOffsetAngle()  # 生成Ray偏向角
    c_spread = generate_rms_spread_4_cluster(scenario, elevation_angle, type_freq, type_path)  # 向量[c_DS(ns) c_asd(°) c_asa(°) c_zsa(°)]
    table_zsd = spread_zod(scenario, elevation_angle, type_freq, type_path)
    aoa_ray = np.zeros((num_cluster, num_ray_cluster))
    aod_ray = np.zeros((num_cluster, num_ray_cluster))
    zoa_ray = np.zeros((num_cluster, num_ray_cluster))
    zod_ray = np.zeros((num_cluster, num_ray_cluster))
    for idx_cluster in range(num_cluster):
        for idx_ray in range(num_ray_cluster):
            aoa_ray[idx_cluster, idx_ray] = aoa_cluster[idx_cluster] + c_spread[2] * d_angel_ray[idx_ray]
            aod_ray[idx_cluster, idx_ray] = aod_cluster[idx_cluster] + c_spread[1] * d_angel_ray[idx_ray]
            zoa_ray[idx_cluster, idx_ray] = zoa_cluster[idx_cluster] + c_spread[3] * d_angel_ray[idx_ray]
            zod_ray[idx_cluster, idx_ray] = zod_cluster[idx_cluster] + (3/8) * 10 ** table_zsd[0] * d_angel_ray[idx_ray]

    # 调整角度范围
    for idx_cluster in range(num_cluster):
        for idx_ray in range(num_ray_cluster):
            aoa_ray[idx_cluster, idx_ray] = np.mod(aoa_ray[idx_cluster, idx_ray], 360)
            aod_ray[idx_cluster, idx_ray] = np.mod(aod_ray[idx_cluster, idx_ray], 360)
            zoa_ray[idx_cluster, idx_ray] = np.mod(zoa_ray[idx_cluster, idx_ray], 360)
            zod_ray[idx_cluster, idx_ray] = np.mod(zod_ray[idx_cluster, idx_ray], 360)
            if 180 < zoa_ray[idx_cluster, idx_ray] < 360:
                zoa_ray[idx_cluster, idx_ray] = 360 - zoa_ray[idx_cluster, idx_ray]
            if 180 < zod_ray[idx_cluster, idx_ray] < 360:
                zod_ray[idx_cluster, idx_ray] = 360 - zod_ray[idx_cluster, idx_ray]

    # Step8: 为簇内的路径的方位角和仰角配对(随机组合)
    for idx_cluster in range(num_cluster):
        rand_idx_aoa = np.random.permutation(num_ray_cluster)
        rand_idx_aod = np.random.permutation(num_ray_cluster)
        rand_idx_zoa = np.random.permutation(num_ray_cluster)
        rand_idx_zod = np.random.permutation(num_ray_cluster)
        aoa_ray[idx_cluster, :] = aoa_ray[idx_cluster, rand_idx_aoa]
        aod_ray[idx_cluster, :] = aod_ray[idx_cluster, rand_idx_aod]
        zoa_ray[idx_cluster, :] = zoa_ray[idx_cluster, rand_idx_zoa]
        zod_ray[idx_cluster, :] = zod_ray[idx_cluster, rand_idx_zod]

    # Step9: 生成交叉极化功率比
    table_x = generate_power_ratio_4_x_pol(scenario, elevation_angle, type_freq, type_path)
    x_nm = np.random.normal(table_x[0], table_x[1], (num_cluster, num_ray_cluster))
    xpr = 10 ** (x_nm / 10)  # 交叉极化功率比

    # Step10: 绘制初始随机相位
    initial_phase = np.random.uniform(-np.pi, np.pi, (num_cluster, num_ray_cluster, 4))

    # Step11: 为每个簇n和每个接收发射器元件对us生成信道系数
    # 将经纬度转化为坐标
    x_sat = (position_sat[2] + radius_earth) * np.cos(np.radians(position_sat[0])) * np.cos(np.radians(position_sat[1]))
    y_sat = (position_sat[2] + radius_earth) * np.sin(np.radians(position_sat[0])) * np.cos(np.radians(position_sat[1]))
    z_sat = (position_sat[2] + radius_earth) * np.sin(np.radians(position_sat[1]))
    x_gu = (position_gu[2] + radius_earth) * np.cos(np.radians(position_gu[0])) * np.cos(np.radians(position_gu[1]))
    y_gu = (position_gu[2] + radius_earth) * np.sin(np.radians(position_gu[0])) * np.cos(np.radians(position_gu[1]))
    z_gu = (position_gu[2] + radius_earth) * np.sin(np.radians(position_gu[1]))

    # 卫星用户坐标系方向（GCS）
    axia_z = np.array([x_gu, y_gu, z_gu])
    axia_z_norm = axia_z / np.linalg.norm(axia_z)
    axia_temp = np.array([x_gu - x_sat, y_gu - y_sat, z_gu - z_sat])
    axia_x = axia_temp - np.dot(axia_temp, axia_z_norm) * axia_z_norm
    axia_x_norm = axia_x / np.linalg.norm(axia_x)
    axia_y_norm = np.cross(axia_z_norm, axia_x_norm)

    # 卫星天线阵列的坐标系方向(LCS)
    axia_sat_x = -np.array([x_sat, y_sat, z_sat])
    axia_sat_x_norm = axia_sat_x / np.linalg.norm(axia_sat_x)
    axia_sat_y = np.array([-y_sat, x_sat, 0])
    axia_sat_y_norm = axia_sat_y / np.linalg.norm(axia_sat_y)
    axia_sat_z_norm = np.cross(axia_sat_x_norm, axia_sat_y_norm)
    if axia_sat_z_norm[2] < 0:
        axia_sat_y_norm = (-1) * axia_sat_y_norm
        axia_sat_z_norm = (-1) * axia_sat_z_norm
    speed_sat_local = np.array([np.dot(axia_sat_x_norm, speed_sat),
                                np.dot(axia_sat_y_norm, speed_sat),
                                np.dot(axia_sat_z_norm, speed_sat)])
    speed_sat_global = np.array([np.dot(axia_x_norm, speed_sat),
                                 np.dot(axia_y_norm, speed_sat),
                                 np.dot(axia_z_norm, speed_sat)])

    # 计算卫星天线阵列在GCS中的方向增益
    angle_sat = coordinate_local_2_global(axia_sat_x_norm, axia_sat_y_norm, axia_sat_z_norm,
                                          axia_x_norm, axia_y_norm, axia_z_norm)
    alpha_sat = np.radians(angle_sat[0])
    beta_sat = np.radians(angle_sat[1])
    gama_sat = np.radians(angle_sat[2])
    matrix_r_sat = np.array([
        [np.cos(alpha_sat) * np.cos(beta_sat),
         np.cos(alpha_sat) * np.sin(beta_sat) * np.sin(gama_sat) - np.sin(alpha_sat) * np.cos(gama_sat),
         np.cos(alpha_sat) * np.sin(beta_sat) * np.cos(gama_sat) + np.sin(alpha_sat) * np.sin(gama_sat)],
        [np.sin(alpha_sat) * np.cos(beta_sat),
         np.sin(alpha_sat) * np.sin(beta_sat) * np.sin(gama_sat) + np.cos(alpha_sat) * np.cos(gama_sat),
         np.sin(alpha_sat) * np.sin(beta_sat) * np.cos(gama_sat) - np.cos(alpha_sat) * np.sin(gama_sat)],
        [-np.sin(beta_sat),
         np.cos(beta_sat) * np.sin(gama_sat),
         np.cos(beta_sat) * np.cos(gama_sat)]
    ])

    # 用户天线阵列的坐标系方向(LCS)
    axia_gu_x = np.array([x_gu, y_gu, z_gu])
    axia_gu_x_norm = axia_gu_x / np.linalg.norm(axia_gu_x)
    axia_gu_y = np.array([-y_gu, x_gu, 0])
    axia_gu_y_norm = axia_gu_y / np.linalg.norm(axia_gu_y)
    axia_gu_z_norm = np.cross(axia_gu_x_norm, axia_gu_y_norm)
    if axia_gu_z_norm[2] < 0:
        axia_gu_y_norm = (-1) * axia_gu_y_norm
        axia_gu_z_norm = (-1) * axia_gu_z_norm

    # 计算用户天线阵列在GCS中的方向增益
    angle_gu = coordinate_local_2_global(axia_gu_x_norm, axia_gu_y_norm, axia_gu_z_norm,
                                         axia_x_norm, axia_y_norm, axia_z_norm)
    alpha_gu = np.radians(angle_gu[0])
    beta_gu = np.radians(angle_gu[1])
    gama_gu = np.radians(angle_gu[2])
    matrix_r_gu = np.array([
        [np.cos(alpha_gu) * np.cos(beta_gu),
         np.cos(alpha_gu) * np.sin(beta_gu) * np.sin(gama_gu) - np.sin(alpha_gu) * np.cos(gama_gu),
         np.cos(alpha_gu) * np.sin(beta_gu) * np.cos(gama_gu) + np.sin(alpha_gu) * np.sin(gama_gu)],
        [np.sin(alpha_gu) * np.cos(beta_gu),
         np.sin(alpha_gu) * np.sin(beta_gu) * np.sin(gama_gu) + np.cos(alpha_gu) * np.cos(gama_gu),
         np.sin(alpha_gu) * np.sin(beta_gu) * np.cos(gama_gu) - np.cos(alpha_gu) * np.sin(gama_gu)],
        [-np.sin(beta_gu),
         np.cos(beta_gu) * np.sin(gama_gu),
         np.cos(beta_gu) * np.cos(gama_gu)]
    ])

    for s_pol in range(num_ant_sat_pol):
        for u_pol in range(num_ant_gu_pol):
            matrix_f_tx = [[np.zeros([2, 1]) for _ in range(num_ray_cluster)] for _ in range(num_cluster)]
            ray_angle_tx = [[None for _ in range(num_ray_cluster)] for _ in range(num_cluster)]
            r_tx = [[None for _ in range(num_ray_cluster)] for _ in range(num_cluster)]
            for idx_cluster in range(num_cluster + 1):
                for idx_ray in range(num_ray_cluster):
                    if idx_cluster < num_cluster:  # NLOS径
                        theta_tx = np.radians(zod_ray[idx_cluster][idx_ray])
                        phi_tx = np.radians(aod_ray[idx_cluster][idx_ray])
                    else:  # LOS径
                        if idx_ray > 0:
                            continue
                        theta_tx = np.radians(zod)
                        phi_tx = np.radians(aod)

                    theta_tx_local = np.arccos(np.cos(beta_sat) * np.cos(gama_sat) * np.cos(theta_tx) +
                                               (np.sin(beta_sat) * np.cos(gama_sat) * np.cos(phi_tx - alpha_sat) -
                                                np.sin(gama_sat) * np.sin(phi_tx - alpha_sat)) * np.sin(theta_tx))
                    phi_tx_local = np.arctan2((np.cos(beta_sat) * np.sin(gama_sat) * np.cos(theta_tx) +
                                               (np.sin(beta_sat) * np.sin(gama_sat) * np.cos(phi_tx - alpha_sat) +
                                                np.cos(gama_sat) * np.sin(phi_tx - alpha_sat)) * np.sin(theta_tx)),
                                              (np.cos(beta_sat) * np.sin(theta_tx) * np.cos(phi_tx - alpha_sat) -
                                               np.sin(beta_sat) * np.cos(theta_tx)))

                    if idx_cluster < num_cluster:
                        ray_angle_tx[idx_cluster][idx_ray] = [np.degrees(theta_tx_local),
                                                              np.degrees(phi_tx_local)]
                    else:
                        ray_angle_tx_los = [np.degrees(theta_tx_local),
                                            np.degrees(phi_tx_local)]

                    # 天线单元辐射功率pattern （暂时使用地面阵列天线）
                    theta_3dB = 65  # (°)
                    sla_v = 30  # (dB)
                    phi_3dB = 65  # (°)
                    sla_h = 30  # (dB)
                    A_max = 30  # (dB)

                    A_theta_dB = -12 * ((np.degrees(theta_tx_local) - 90) / theta_3dB) ** 2
                    A_theta_dB = max(A_theta_dB, -sla_v)

                    A_phi_dB = -12 * (np.degrees(phi_tx_local) / phi_3dB) ** 2
                    A_phi_dB = max(A_phi_dB, -sla_h)

                    A_dB = A_theta_dB + A_phi_dB
                    A_dB = max(A_dB, -A_max)

                    A_sat = 10 ** (A_dB / 10)

                    # 单极化天线
                    if num_ant_sat_pol == 1:
                        matrix_f_tx_local_theta = np.sqrt(A_sat)
                        matrix_f_tx_local_phi = 0
                    elif num_ant_sat_pol == 2:  # 双极化天线(45/-45)
                        sigma = 45
                        matrix_f_tx_local_theta = np.sqrt(A_sat) * np.cos(np.radians(sigma - 90 * (s_pol - 1)))
                        matrix_f_tx_local_phi = np.sqrt(A_sat) * np.sin(np.radians(sigma - 90 * (s_pol - 1)))

                    # 计算卫星天线阵列在GCS中的方向增益
                    spherical_unit_vector_global_theta = np.array([np.cos(theta_tx) * np.cos(phi_tx),
                                                                   np.cos(theta_tx) * np.sin(phi_tx),
                                                                   -np.sin(theta_tx)])
                    spherical_unit_vector_global_phi = np.array([-np.sin(phi_tx),
                                                                 np.cos(phi_tx),
                                                                 0])
                    spherical_unit_vector_local_theta = np.array([np.cos(theta_tx_local) * np.cos(phi_tx_local),
                                                                  np.cos(theta_tx_local) * np.sin(phi_tx_local),
                                                                  -np.sin(theta_tx_local)])
                    spherical_unit_vector_local_phi = np.array([-np.sin(phi_tx_local),
                                                                np.cos(phi_tx_local),
                                                                0])

                    matrix_transform_tx = np.array([
                        [spherical_unit_vector_global_theta @ matrix_r_sat @ spherical_unit_vector_local_theta,
                         spherical_unit_vector_global_theta @ matrix_r_sat @ spherical_unit_vector_local_phi],
                        [spherical_unit_vector_global_phi @ matrix_r_sat @ spherical_unit_vector_local_theta,
                         spherical_unit_vector_global_phi @ matrix_r_sat @ spherical_unit_vector_local_phi]])

                    matrix_f_tx_local = np.array([matrix_f_tx_local_theta,
                                                  matrix_f_tx_local_phi])
                    matrix_f_tx_global = matrix_transform_tx @ matrix_f_tx_local

                    if idx_cluster < num_cluster:  # NLOS径
                        matrix_f_tx[idx_cluster][idx_ray] = np.array([[matrix_f_tx_global[0]],
                                                                      [matrix_f_tx_global[1]]])
                        r_tx[idx_cluster][idx_ray] = [np.sin(theta_tx) * np.cos(phi_tx),
                                                      np.sin(theta_tx) * np.sin(phi_tx),
                                                      np.cos(theta_tx)]
                    else:  # LOS径
                        matrix_f_tx_los = np.array([[matrix_f_tx_global[0]],
                                                    [matrix_f_tx_global[1]]])
                        r_tx_los = [np.sin(theta_tx) * np.cos(phi_tx),
                                    np.sin(theta_tx) * np.sin(phi_tx),
                                    np.cos(theta_tx)]

            matrix_f_rx = [[np.zeros([2, 1]) for _ in range(num_ray_cluster)] for _ in range(num_cluster)]
            ray_angle_rx = [[None for _ in range(num_ray_cluster)] for _ in range(num_cluster)]
            r_rx = [[None for _ in range(num_ray_cluster)] for _ in range(num_cluster)]

            for idx_cluster in range(num_cluster + 1):
                for idx_ray in range(num_ray_cluster):
                    if idx_cluster < num_cluster:  # NLOS径
                        theta_rx = np.radians(zoa_ray[idx_cluster][idx_ray])
                        phi_rx = np.radians(aoa_ray[idx_cluster][idx_ray])
                    else:  # LOS径
                        if idx_ray > 0:
                            continue
                        theta_rx = np.radians(zoa_los)
                        phi_rx = np.radians(aoa_los)

                    theta_rx_local = np.arccos(np.cos(beta_gu) * np.cos(gama_gu) * np.cos(theta_rx) +
                                               (np.sin(beta_gu) * np.cos(gama_gu) * np.cos(phi_rx - alpha_gu) -
                                                np.sin(gama_gu) * np.sin(phi_rx - alpha_gu)) * np.sin(theta_rx))
                    phi_rx_local = np.arctan2(np.cos(beta_gu) * np.sin(gama_gu) * np.cos(theta_rx) +
                                              (np.sin(beta_gu) * np.sin(gama_gu) * np.cos(phi_rx - alpha_gu) +
                                               np.cos(gama_gu) * np.sin(phi_rx - alpha_gu)) * np.sin(theta_rx),
                                              np.cos(beta_gu) * np.sin(theta_rx) * np.cos(phi_rx - alpha_gu) -
                                              np.sin(beta_gu) * np.cos(theta_rx))

                    if idx_cluster < num_cluster:
                        ray_angle_rx[idx_cluster][idx_ray] = [np.degrees(theta_rx_local),
                                                              np.degrees(phi_rx_local)]
                    else:
                        ray_angle_rx_los = [np.degrees(theta_rx_local),
                                            np.degrees(phi_rx_local)]

                    # 天线单元辐射功率pattern （暂时使用地面阵列天线）
                    theta_3dB = 65  # (°)
                    sla_v = 30  # (dB)
                    phi_3dB = 65  # (°)
                    sla_h = 30  # (dB)
                    A_max = 30  # (dB)

                    A_theta_dB = -12 * ((np.degrees(theta_rx_local) - 90) / theta_3dB) ** 2
                    A_theta_dB = max(A_theta_dB, -sla_v)

                    A_phi_dB = -12 * (np.degrees(phi_rx_local) / phi_3dB) ** 2
                    A_phi_dB = max(A_phi_dB, -sla_h)

                    A_dB = A_theta_dB + A_phi_dB
                    A_dB = max(A_dB, -A_max)

                    A_gu = 10 ** (A_dB / 10)

                    # 单极化天线
                    if num_ant_gu_pol == 1:
                        matrix_f_rx_local_theta = np.sqrt(A_gu)
                        matrix_f_rx_local_phi = 0
                    elif num_ant_gu_pol == 2:  # 双极化天线(45/-45)
                        sigma = 45
                        matrix_f_rx_local_theta = np.sqrt(A_gu) * np.cos(np.radians(sigma - 90 * (u_pol - 1)))
                        matrix_f_rx_local_phi = np.sqrt(A_gu) * np.sin(np.radians(sigma - 90 * (u_pol - 1)))

                    # 计算卫星天线阵列在GCS中的方向增益
                    spherical_unit_vector_global_theta = np.array([np.cos(theta_rx) * np.cos(phi_rx),
                                                                   np.cos(theta_rx) * np.sin(phi_rx),
                                                                   -np.sin(theta_rx)])
                    spherical_unit_vector_global_phi = np.array([-np.sin(phi_rx),
                                                                 np.cos(phi_rx),
                                                                 0])
                    spherical_unit_vector_local_theta = np.array([np.cos(theta_rx_local) * np.cos(phi_rx_local),
                                                                  np.cos(theta_rx_local) * np.sin(phi_rx_local),
                                                                  -np.sin(theta_rx_local)])
                    spherical_unit_vector_local_phi = np.array([-np.sin(phi_rx_local),
                                                                np.cos(phi_rx_local),
                                                                0])

                    matrix_transform_rx = np.array([
                        [spherical_unit_vector_global_theta @ matrix_r_gu @ spherical_unit_vector_local_theta,
                         spherical_unit_vector_global_theta @ matrix_r_gu @ spherical_unit_vector_local_phi],
                        [spherical_unit_vector_global_phi @ matrix_r_gu @ spherical_unit_vector_local_theta,
                         spherical_unit_vector_global_phi @ matrix_r_gu @ spherical_unit_vector_local_phi]])

                    matrix_f_rx_local = np.array([matrix_f_rx_local_theta,
                                                  matrix_f_rx_local_phi])
                    matrix_f_rx_global = matrix_transform_rx @ matrix_f_rx_local

                    if idx_cluster < num_cluster:  # NLOS径
                        matrix_f_rx[idx_cluster][idx_ray] = np.array([[matrix_f_rx_global[0]],
                                                                      [matrix_f_rx_global[1]]])
                        r_rx[idx_cluster][idx_ray] = [np.sin(theta_rx) * np.cos(phi_rx),
                                                      np.sin(theta_rx) * np.sin(phi_rx),
                                                      np.cos(theta_rx)]
                    else:  # LOS径
                        matrix_f_rx_los = np.array([[matrix_f_rx_global[0]],
                                                    [matrix_f_rx_global[1]]])
                        r_rx_los = [np.sin(theta_rx) * np.cos(phi_rx),
                                    np.sin(theta_rx) * np.sin(phi_rx),
                                    np.cos(theta_rx)]

            # 法拉第旋转
            phase_faraday = 108 / (freq_carrier / 1e9) ** 2 * np.pi / 180
            matrix_f_r = np.array([[np.cos(phase_faraday),
                                    -np.sin(phase_faraday)],
                                   [np.sin(phase_faraday),
                                    np.cos(phase_faraday)]])

            # 针对弱簇
            for s_h in range(num_ant_sat_hori):
                for s_v in range(num_ant_sat_vert):
                    s = s_pol * num_ant_sat_hori * num_ant_sat_vert + s_v * num_ant_sat_hori + s_h
                    for u_h in range(num_ant_gu_hori):
                        for u_v in range(num_ant_gu_vert):
                            u = u_pol * num_ant_gu_hori * num_ant_gu_vert + u_v * num_ant_gu_hori + u_h

                            h_cluster = np.zeros(num_cluster, dtype=complex)
                            # 对于弱簇
                            for idx_cluster in range(2, num_cluster):
                                sum_weak = 0.0
                                for idx_ray in range(num_ray_cluster):
                                    if num_ant_sat_pol == 1:
                                        matrix_weak = np.array([[np.exp(1j * initial_phase[idx_cluster][idx_ray][0]),
                                                                 0],
                                                                [0,
                                                                 np.exp(1j * initial_phase[idx_cluster][idx_ray][0])]])
                                    elif num_ant_sat_pol == 2:
                                        matrix_weak = np.array([[np.exp(1j * initial_phase[idx_cluster][idx_ray][0]),
                                                                 np.sqrt(1 / xpr[idx_cluster][idx_ray]) * np.exp(1j * initial_phase[idx_cluster][idx_ray][1])],
                                                                [np.sqrt(1 / xpr[idx_cluster][idx_ray]) * np.exp(1j * initial_phase[idx_cluster][idx_ray][2]),
                                                                 np.exp(1j * initial_phase[idx_cluster][idx_ray][3])]])

                                    # 卫星天线阵列单元相移
                                    phase_elem_s_h = 2 * np.pi * s_h * d / lamda * np.sin(np.radians(ray_angle_tx[idx_cluster][idx_ray][1]))
                                    phase_elem_s_v = 2 * np.pi * s_v * d / lamda * np.sin(np.radians(ray_angle_tx[idx_cluster][idx_ray][0] - 90))
                                    phase_elem_u_h = 2 * np.pi * u_h * d / lamda * np.sin(np.radians(ray_angle_rx[idx_cluster][idx_ray][1]))
                                    phase_elem_u_v = 2 * np.pi * u_v * d / lamda * np.sin(np.radians(ray_angle_rx[idx_cluster][idx_ray][0] - 90))
                                    phase_elem = phase_elem_s_h + phase_elem_s_v + phase_elem_u_h + phase_elem_u_v
                                    # 多普勒频移
                                    phase_dopple = 2 * np.pi * np.dot(r_tx[idx_cluster][idx_ray], speed_sat_local) / lamda * (time_system + delay_cluster[idx_cluster])

                                    sum_weak += ((matrix_f_rx[idx_cluster][idx_ray].T @ matrix_weak @ matrix_f_r @ matrix_f_tx[idx_cluster][idx_ray]) * 
                                                 np.exp(1j * (phase_elem+phase_dopple)))

                                h_cluster[idx_cluster] = np.sqrt(power_cluster[idx_cluster] / num_ray_cluster) * sum_weak

                            # 对于强簇，分为3个中强径
                            delta_delay_spread = np.zeros(num_ray_cluster)
                            for idx_ray in range(num_ray_cluster):
                                if 8 <= idx_ray <= 11:
                                    delta_delay_spread[idx_ray] = 1.28 * c_spread[0] * 1e-9
                                if 12 <= idx_ray <= 15:
                                    delta_delay_spread[idx_ray] = 2.56 * c_spread[0] * 1e-9
                                if 16 <= idx_ray <= 17:
                                    delta_delay_spread[idx_ray] = 1.28 * c_spread[0] * 1e-9

                            for idx_cluster in range(2):
                                sum_strong = 0
                                for idx_ray in range(num_ray_cluster):
                                    if num_ant_sat_pol == 1:
                                        matrix_strong = np.array([[np.exp(1j * initial_phase[idx_cluster][idx_ray][0]), 
                                                                   0],
                                                                  [0, 
                                                                   np.exp(1j * initial_phase[idx_cluster][idx_ray][0])]])
                                    elif num_ant_sat_pol == 2:
                                        matrix_strong = np.array([[np.exp(1j * initial_phase[idx_cluster][idx_ray][0]),
                                                                   np.sqrt(1 / xpr[idx_cluster][idx_ray]) * np.exp(1j * initial_phase[idx_cluster][idx_ray][1])],
                                                                  [np.sqrt(1 / xpr[idx_cluster][idx_ray]) * np.exp(1j * initial_phase[idx_cluster][idx_ray][2]),
                                                                   np.exp(1j * initial_phase[idx_cluster][idx_ray][3])]])

                                    # 天线阵列单元相移
                                    phase_elem_s_h = 2 * np.pi * s_h * d / lamda * np.sin(np.radians(ray_angle_tx[idx_cluster][idx_ray][1]))
                                    phase_elem_s_v = 2 * np.pi * s_v * d / lamda * np.sin(np.radians(ray_angle_tx[idx_cluster][idx_ray][0] - 90))
                                    phase_elem_u_h = 2 * np.pi * u_h * d / lamda * np.sin(np.radians(ray_angle_rx[idx_cluster][idx_ray][1]))
                                    phase_elem_u_v = 2 * np.pi * u_v * d / lamda * np.sin(np.radians(ray_angle_rx[idx_cluster][idx_ray][0] - 90))
                                    phase_elem = phase_elem_s_h + phase_elem_s_v + phase_elem_u_h + phase_elem_u_v
                                    # 多普勒频移
                                    phase_dopple = 2 * np.pi * np.dot(r_tx[idx_cluster][idx_ray], speed_sat_local) / lamda * (time_system + delay_cluster[idx_cluster] + delta_delay_spread[idx_ray])

                                    sum_strong += ((matrix_f_rx[idx_cluster][idx_ray].T @ matrix_strong @ matrix_f_r @ matrix_f_tx[idx_cluster][idx_ray]) * 
                                                   np.exp(1j * (phase_elem + phase_dopple)))

                                h_cluster[idx_cluster] = np.sqrt(power_cluster[idx_cluster] / num_ray_cluster) * sum_strong

                                if abs(h_cluster[idx_cluster]) > 1:
                                    temp = abs(h_cluster[idx_cluster])
                                    sr = 1

                            h_nlos = np.sum(h_cluster)

                            if type_path == 1:  # LOS路径
                                matrix_los = np.array([[1, 0], 
                                                       [0, -1]])

                                # LOS路径相移
                                phase_fs = 2 * np.pi * distance / lamda
                                # 卫星天线阵列单元相移
                                phase_elem_s_h = 2 * np.pi * s_h * d / lamda * np.sin(np.radians(ray_angle_tx_los[1]))
                                phase_elem_s_v = 2 * np.pi * s_v * d / lamda * np.sin(np.radians(ray_angle_tx_los[0] - 90))
                                phase_elem_u_h = 2 * np.pi * u_h * d / lamda * np.sin(np.radians(ray_angle_rx_los[1]))
                                phase_elem_u_v = 2 * np.pi * u_v * d / lamda * np.sin(np.radians(ray_angle_rx_los[0] - 90))
                                phase_elem = phase_elem_s_h + phase_elem_s_v + phase_elem_u_h + phase_elem_u_v
                                # 多普勒频移
                                phase_dopple = 2 * np.pi * np.dot(r_tx_los, speed_sat_local) / lamda * (time_system + delay_cluster[0])

                                h_los = ((matrix_f_rx_los.T @ matrix_los @ matrix_f_r @  matrix_f_tx_los) *
                                         np.exp(-1j * (phase_fs + phase_elem + phase_dopple)))
                                
                                matrix_h[s, u] = np.sqrt(1 / (kf + 1)) * h_nlos + np.sqrt(kf / (kf + 1)) * h_los
                            else:
                                matrix_h[s, u] = h_nlos

                            if abs(matrix_h[s, u]) > 1:
                                temp = abs(matrix_h[s, u])
                                sr = 1
                            if 'maxh' not in locals() or abs(matrix_h[s, u]) > abs(maxh):
                                maxh = matrix_h[s, u]

                            if abs(matrix_h[s, u]) < 0:
                                stop = 1
                                
    return matrix_h, loss_path


# 生成大尺度参数
def generate_large_scale_param(scenario, alpha, type_freq, type_path):
    # 读取互相关矩阵[s_SF s_K s_DS s_asd s_asa s_zsd s_zsa]
    matrix_cross_correlation = cross_correlation(scenario, alpha, type_freq, type_path)

    # 初始化
    num_param = matrix_cross_correlation.shape[0]
    param_large_scale = np.zeros(num_param)

    # 生成多维正态分布，即多个互相关的N(0,1)分布
    X = np.linalg.cholesky(matrix_cross_correlation) @ np.random.normal(0, 1, (num_param, 1))

    # 向量[s_SF s_K s_DS s_asd s_asa s_zsd s_zsa]
    table_sf = shadow_fading(scenario, alpha, type_freq, type_path)
    table_kf = k_factor(scenario, alpha, type_freq, type_path)
    table_ds = spread_delay(scenario, alpha, type_freq, type_path)
    table_asd = spread_aod(scenario, alpha, type_freq, type_path)
    table_asa = spread_aoa(scenario, alpha, type_freq, type_path)
    table_zsd = spread_zod(scenario, alpha, type_freq, type_path)
    table_zsa = spread_zoa(scenario, alpha, type_freq, type_path)

    # 向量[s_SF s_K s_DS s_asd s_asa s_zsd s_zsa]
    param_large_scale[0] = X[0] * table_sf[1] + table_sf[0]
    param_large_scale[1] = X[1] * table_kf[1] + table_kf[0]
    temp = X[2] * table_ds[1] + table_ds[0]
    param_large_scale[2] = 10 ** temp
    temp = X[3] * table_asd[1] + table_asd[0]
    param_large_scale[3] = 10 ** temp
    temp = X[4] * table_asa[1] + table_asa[0]
    param_large_scale[4] = 10 ** temp
    temp = X[5] * table_zsd[1] + table_zsd[0]
    param_large_scale[5] = 10 ** temp
    temp = X[6] * table_zsa[1] + table_zsa[0]
    param_large_scale[6] = 10 ** temp

    for n in range(2, 7):
        if param_large_scale[n] > 180:
            param_large_scale[n] = 180

    return param_large_scale


# 坐标系转换
def coordinate_local_2_global(axis_x_local, axis_y_local, axis_z_local, axis_x_global, axis_y_global, axis_z_global):
    x_3 = axis_x_local / np.linalg.norm(axis_x_local)
    y_3 = axis_y_local / np.linalg.norm(axis_y_local)
    z_3 = axis_z_local / np.linalg.norm(axis_z_local)
    x_0 = axis_x_global / np.linalg.norm(axis_x_global)
    y_0 = axis_y_global / np.linalg.norm(axis_y_global)
    z_0 = axis_z_global / np.linalg.norm(axis_z_global)

    # 保持x_3轴不变，旋转坐标系
    x_2 = x_3.copy()
    # y_2轴垂直于x_2轴和z_0轴
    y_temp = np.cross(x_2, z_0)

    if np.linalg.norm(y_temp) == 0:
        x_2 = x_3.copy()
        y_2 = y_3.copy()
        z_2 = z_3.copy()
        gama = 0
    else:
        y_2 = y_temp / np.linalg.norm(y_temp)
        if np.dot(y_2, y_3) < -1e-7:
            y_2 = -y_2
        z_temp = np.cross(x_2, y_2)
        z_2 = z_temp / np.linalg.norm(z_temp)
        if np.dot(y_3, z_2) > -1e-7:
            gama = np.arccos(np.dot(y_3, y_2))
        else:
            gama = -np.arccos(np.dot(y_3, y_2))

    # 保持y_2轴不变，旋转坐标系
    y_1 = y_2.copy()
    z_1 = z_0.copy()
    x_temp = np.cross(y_1, z_1)
    x_1 = x_temp / np.linalg.norm(x_temp)
    if np.dot(z_2, x_1) > -1e-7:
        beta = np.arccos(np.dot(z_2, z_1))
    else:
        beta = -np.arccos(np.dot(z_2, z_1))

    # 保持z_1轴不变，旋转坐标系，使得坐标系与GCS重合
    if np.dot(x_1, y_0) > -1e-7:
        temp = np.dot(x_1, x_0)
        if np.dot(x_1, x_0) > 1.0:
            alpha = 0.0
        elif np.dot(x_1, x_0) < -1.0:
            alpha = 180.0
        else:
            alpha = np.arccos(np.dot(x_1, x_0))
    else:
        if np.dot(x_1, x_0) > 1.0:
            alpha = 180.0
        elif np.dot(x_1, x_0) < -1.0:
            alpha = 0.0
        else:
            alpha = -np.arccos(np.dot(x_1, x_0))

    # 转化为角度值
    axis_angle = np.array([alpha, beta, gama]) * 180 / np.pi

    return axis_angle.tolist()


# Laplacian分布
def randlap(mean, std):
    # 生成μ=0，σ=1的标准Laplacian分布的随机数(λ=sqrt(2))
    # 概率密度函数 f(x) = exp(-|x|/sqrt(2))/(2*sqrt(2))
    # 概率分布函数 F(x) = exp(x/sqrt(2))/2  (x<0)
    #            F(x) = 1-exp((-1)*x/sqrt(2))/2  (x>=0)

    F = np.random.rand()
    if 0.5 < F <= 1:
        x = -np.sqrt(2) * np.log(2 - 2 * F)
    elif 0 <= F <= 0.5:
        x = np.sqrt(2) * np.log(2 * F)
    else:
        raise ValueError("Random number out of range for Laplacian distribution")

    # 调整随机数对应的均值和方差
    random = x * std + mean

    return random


# 第一类贝塞尔函数(整数阶)
def Bessel(x, n):
    # 定义要积分的函数
    def integrand(r, n, x):
        return np.cos(n * r - x * np.sin(r))

    # 进行积分计算
    result = quad(integrand, 0, np.pi, args=(n, x))

    # 计算贝塞尔函数值
    y = result[0] / np.pi

    return y


"""===============================================表格数据==============================================="""


# 互相关系数矩阵（TS 38.811）
def cross_correlation(scenario, alpha, type_freq, type_path):
    matrix_cross_correlation = np.zeros((7, 7))
    np.fill_diagonal(matrix_cross_correlation, 1)
    class_ea = round(alpha / 10)

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario(LOS)
        matrix_cross_correlation[3, 2] = 0.4
        matrix_cross_correlation[4, 2] = 0.8
        matrix_cross_correlation[4, 0] = -0.5
        matrix_cross_correlation[3, 0] = -0.5
        matrix_cross_correlation[2, 0] = -0.4
        matrix_cross_correlation[4, 3] = 0
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = -0.2
        matrix_cross_correlation[2, 1] = -0.4
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0
        matrix_cross_correlation[6, 0] = -0.8
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = 0
        matrix_cross_correlation[5, 2] = -0.2
        matrix_cross_correlation[6, 2] = 0
        matrix_cross_correlation[5, 3] = 0.5
        matrix_cross_correlation[6, 3] = 0
        matrix_cross_correlation[5, 4] = -0.3
        matrix_cross_correlation[6, 4] = 0.4
        matrix_cross_correlation[6, 5] = 0

    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario(NLOS)
        matrix_cross_correlation[3, 2] = 0.4
        matrix_cross_correlation[4, 2] = 0.6
        matrix_cross_correlation[4, 0] = 0
        matrix_cross_correlation[3, 0] = -0.6
        matrix_cross_correlation[2, 0] = -0.4
        matrix_cross_correlation[4, 3] = 0.4
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = 0
        matrix_cross_correlation[2, 1] = 0
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0
        matrix_cross_correlation[6, 0] = -0.4
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = 0
        matrix_cross_correlation[5, 2] = -0.5
        matrix_cross_correlation[6, 2] = 0
        matrix_cross_correlation[5, 3] = 0.5
        matrix_cross_correlation[6, 3] = -0.1
        matrix_cross_correlation[5, 4] = 0
        matrix_cross_correlation[6, 4] = 0
        matrix_cross_correlation[6, 5] = 0

    elif scenario == 1 and type_path == 1:  # Urban Scenario(LOS)
        matrix_cross_correlation[3, 2] = 0.4
        matrix_cross_correlation[4, 2] = 0.8
        matrix_cross_correlation[4, 0] = -0.5
        matrix_cross_correlation[3, 0] = -0.5
        matrix_cross_correlation[2, 0] = -0.4
        matrix_cross_correlation[4, 3] = 0
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = -0.2
        matrix_cross_correlation[2, 1] = -0.4
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0
        matrix_cross_correlation[6, 0] = -0.8
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = 0
        matrix_cross_correlation[5, 2] = -0.2
        matrix_cross_correlation[6, 2] = 0
        matrix_cross_correlation[5, 3] = 0.5
        matrix_cross_correlation[6, 3] = 0
        matrix_cross_correlation[5, 4] = -0.3
        matrix_cross_correlation[6, 4] = 0.4
        matrix_cross_correlation[6, 5] = 0

    elif scenario == 1 and type_path == 0:  # Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                matrix_cross_correlation[3, 2] = 0.54
                matrix_cross_correlation[4, 2] = 0.38
                matrix_cross_correlation[4, 0] = -0.05
                matrix_cross_correlation[3, 0] = -0.48
                matrix_cross_correlation[2, 0] = -0.22
                matrix_cross_correlation[4, 3] = 0.41
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.02
                matrix_cross_correlation[6, 0] = -0.31
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.69
                matrix_cross_correlation[6, 2] = 0.05
                matrix_cross_correlation[5, 3] = 0.52
                matrix_cross_correlation[6, 3] = 0.05
                matrix_cross_correlation[5, 4] = 0.4
                matrix_cross_correlation[6, 4] = 0.04
                matrix_cross_correlation[6, 5] = -0.03
            elif class_ea == 2:
                matrix_cross_correlation[3, 2] = 0.46
                matrix_cross_correlation[4, 2] = 0.36
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.53
                matrix_cross_correlation[2, 0] = -0.26
                matrix_cross_correlation[4, 3] = 0.4
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.72
                matrix_cross_correlation[6, 2] = 0.09
                matrix_cross_correlation[5, 3] = 0.48
                matrix_cross_correlation[6, 3] = 0.11
                matrix_cross_correlation[5, 4] = 0.39
                matrix_cross_correlation[6, 4] = 0.13
                matrix_cross_correlation[6, 5] = 0.04
            elif class_ea == 3:
                matrix_cross_correlation[3, 2] = 0.56
                matrix_cross_correlation[4, 2] = 0.27
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.52
                matrix_cross_correlation[2, 0] = -0.21
                matrix_cross_correlation[4, 3] = 0.33
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.01
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.68
                matrix_cross_correlation[6, 2] = 0.09
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = 0.13
                matrix_cross_correlation[5, 4] = 0.34
                matrix_cross_correlation[6, 4] = 0.16
                matrix_cross_correlation[6, 5] = 0.07
            elif class_ea == 4:
                matrix_cross_correlation[3, 2] = 0.52
                matrix_cross_correlation[4, 2] = 0.29
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.52
                matrix_cross_correlation[2, 0] = -0.25
                matrix_cross_correlation[4, 3] = 0.37
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.68
                matrix_cross_correlation[6, 2] = 0.09
                matrix_cross_correlation[5, 3] = 0.56
                matrix_cross_correlation[6, 3] = 0.14
                matrix_cross_correlation[5, 4] = 0.37
                matrix_cross_correlation[6, 4] = 0.13
                matrix_cross_correlation[6, 5] = 0.07
            elif class_ea == 5:
                matrix_cross_correlation[3, 2] = 0.6
                matrix_cross_correlation[4, 2] = 0.21
                matrix_cross_correlation[4, 0] = -0.03
                matrix_cross_correlation[3, 0] = -0.54
                matrix_cross_correlation[2, 0] = -0.21
                matrix_cross_correlation[4, 3] = 0.23
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.01
                matrix_cross_correlation[6, 0] = -0.38
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.64
                matrix_cross_correlation[6, 2] = -0.03
                matrix_cross_correlation[5, 3] = 0.62
                matrix_cross_correlation[6, 3] = -0.02
                matrix_cross_correlation[5, 4] = 0.31
                matrix_cross_correlation[6, 4] = 0.13
                matrix_cross_correlation[6, 5] = -0.01
            elif class_ea == 6:
                matrix_cross_correlation[3, 2] = 0.59
                matrix_cross_correlation[4, 2] = 0.24
                matrix_cross_correlation[4, 0] = -0.05
                matrix_cross_correlation[3, 0] = -0.51
                matrix_cross_correlation[2, 0] = -0.19
                matrix_cross_correlation[4, 3] = 0.23
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.01
                matrix_cross_correlation[6, 0] = -0.39
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.65
                matrix_cross_correlation[6, 2] = -0.15
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = -0.11
                matrix_cross_correlation[5, 4] = 0.28
                matrix_cross_correlation[6, 4] = 0.14
                matrix_cross_correlation[6, 5] = -0.12
            elif class_ea == 7:
                matrix_cross_correlation[3, 2] = 0.6
                matrix_cross_correlation[4, 2] = 0.22
                matrix_cross_correlation[4, 0] = -0.02
                matrix_cross_correlation[3, 0] = -0.5
                matrix_cross_correlation[2, 0] = -0.19
                matrix_cross_correlation[4, 3] = 0.22
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.02
                matrix_cross_correlation[6, 0] = -0.37
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.64
                matrix_cross_correlation[6, 2] = -0.13
                matrix_cross_correlation[5, 3] = 0.65
                matrix_cross_correlation[6, 3] = -0.13
                matrix_cross_correlation[5, 4] = 0.23
                matrix_cross_correlation[6, 4] = -0.02
                matrix_cross_correlation[6, 5] = -0.15
            elif class_ea == 8:
                matrix_cross_correlation[3, 2] = 0.57
                matrix_cross_correlation[4, 2] = 0.24
                matrix_cross_correlation[4, 0] = -0.01
                matrix_cross_correlation[3, 0] = -0.48
                matrix_cross_correlation[2, 0] = -0.2
                matrix_cross_correlation[4, 3] = 0.23
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.02
                matrix_cross_correlation[6, 0] = -0.37
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.64
                matrix_cross_correlation[6, 2] = -0.13
                matrix_cross_correlation[5, 3] = 0.65
                matrix_cross_correlation[6, 3] = -0.13
                matrix_cross_correlation[5, 4] = 0.23
                matrix_cross_correlation[6, 4] = -0.02
                matrix_cross_correlation[6, 5] = -0.34
            elif class_ea == 9:
                matrix_cross_correlation[3, 2] = 0.64
                matrix_cross_correlation[4, 2] = 0.24
                matrix_cross_correlation[4, 0] = 0
                matrix_cross_correlation[3, 0] = -0.43
                matrix_cross_correlation[2, 0] = -0.2
                matrix_cross_correlation[4, 3] = 0.21
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.12
                matrix_cross_correlation[6, 0] = -0.36
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.53
                matrix_cross_correlation[6, 2] = -0.19
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = -0.2
                matrix_cross_correlation[5, 4] = 0.29
                matrix_cross_correlation[6, 4] = -0.35
                matrix_cross_correlation[6, 5] = -0.33
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                matrix_cross_correlation[3, 2] = 0.55
                matrix_cross_correlation[4, 2] = 0.38
                matrix_cross_correlation[4, 0] = -0.05
                matrix_cross_correlation[3, 0] = -0.48
                matrix_cross_correlation[2, 0] = -0.21
                matrix_cross_correlation[4, 3] = 0.41
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.02
                matrix_cross_correlation[6, 0] = -0.31
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.68
                matrix_cross_correlation[6, 2] = 0.06
                matrix_cross_correlation[5, 3] = 0.52
                matrix_cross_correlation[6, 3] = 0.06
                matrix_cross_correlation[5, 4] = 0.4
                matrix_cross_correlation[6, 4] = 0.05
                matrix_cross_correlation[6, 5] = -0.02
            elif class_ea == 2:
                matrix_cross_correlation[3, 2] = 0.47
                matrix_cross_correlation[4, 2] = 0.37
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.52
                matrix_cross_correlation[2, 0] = -0.25
                matrix_cross_correlation[4, 3] = 0.42
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0
                matrix_cross_correlation[6, 0] = -0.32
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.72
                matrix_cross_correlation[6, 2] = 0.1
                matrix_cross_correlation[5, 3] = 0.48
                matrix_cross_correlation[6, 3] = 0.12
                matrix_cross_correlation[5, 4] = 0.41
                matrix_cross_correlation[6, 4] = 0.13
                matrix_cross_correlation[6, 5] = 0.04
            elif class_ea == 3:
                matrix_cross_correlation[3, 2] = 0.55
                matrix_cross_correlation[4, 2] = 0.29
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.52
                matrix_cross_correlation[2, 0] = -0.21
                matrix_cross_correlation[4, 3] = 0.34
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.01
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.68
                matrix_cross_correlation[6, 2] = 0.11
                matrix_cross_correlation[5, 3] = 0.59
                matrix_cross_correlation[6, 3] = 0.14
                matrix_cross_correlation[5, 4] = 0.34
                matrix_cross_correlation[6, 4] = 0.16
                matrix_cross_correlation[6, 5] = 0.11
            elif class_ea == 4:
                matrix_cross_correlation[3, 2] = 0.52
                matrix_cross_correlation[4, 2] = 0.3
                matrix_cross_correlation[4, 0] = -0.04
                matrix_cross_correlation[3, 0] = -0.53
                matrix_cross_correlation[2, 0] = -0.26
                matrix_cross_correlation[4, 3] = 0.38
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.01
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.67
                matrix_cross_correlation[6, 2] = 0.13
                matrix_cross_correlation[5, 3] = 0.55
                matrix_cross_correlation[6, 3] = 0.18
                matrix_cross_correlation[5, 4] = 0.38
                matrix_cross_correlation[6, 4] = 0.16
                matrix_cross_correlation[6, 5] = 0.11
            elif class_ea == 5:
                matrix_cross_correlation[3, 2] = 0.55
                matrix_cross_correlation[4, 2] = 0.23
                matrix_cross_correlation[4, 0] = -0.03
                matrix_cross_correlation[3, 0] = -0.57
                matrix_cross_correlation[2, 0] = -0.25
                matrix_cross_correlation[4, 3] = 0.28
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.03
                matrix_cross_correlation[6, 0] = -0.41
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.65
                matrix_cross_correlation[6, 2] = -0.04
                matrix_cross_correlation[5, 3] = 0.54
                matrix_cross_correlation[6, 3] = 0.01
                matrix_cross_correlation[5, 4] = 0.31
                matrix_cross_correlation[6, 4] = 0.18
                matrix_cross_correlation[6, 5] = 0
            elif class_ea == 6:
                matrix_cross_correlation[3, 2] = 0.57
                matrix_cross_correlation[4, 2] = 0.21
                matrix_cross_correlation[4, 0] = -0.05
                matrix_cross_correlation[3, 0] = -0.53
                matrix_cross_correlation[2, 0] = -0.2
                matrix_cross_correlation[4, 3] = 0.2
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = 0.03
                matrix_cross_correlation[6, 0] = -0.4
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.67
                matrix_cross_correlation[6, 2] = -0.14
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = -0.1
                matrix_cross_correlation[5, 4] = 0.25
                matrix_cross_correlation[6, 4] = 0.21
                matrix_cross_correlation[6, 5] = -0.09
            elif class_ea == 7:
                matrix_cross_correlation[3, 2] = 0.61
                matrix_cross_correlation[4, 2] = 0.23
                matrix_cross_correlation[4, 0] = -0.03
                matrix_cross_correlation[3, 0] = -0.5
                matrix_cross_correlation[2, 0] = -0.19
                matrix_cross_correlation[4, 3] = 0.26
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.02
                matrix_cross_correlation[6, 0] = -0.36
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.63
                matrix_cross_correlation[6, 2] = -0.11
                matrix_cross_correlation[5, 3] = 0.64
                matrix_cross_correlation[6, 3] = -0.11
                matrix_cross_correlation[5, 4] = 0.23
                matrix_cross_correlation[6, 4] = 0.02
                matrix_cross_correlation[6, 5] = -0.15
            elif class_ea == 8:
                matrix_cross_correlation[3, 2] = 0.59
                matrix_cross_correlation[4, 2] = 0.23
                matrix_cross_correlation[4, 0] = -0.01
                matrix_cross_correlation[3, 0] = -0.49
                matrix_cross_correlation[2, 0] = -0.2
                matrix_cross_correlation[4, 3] = 0.23
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.05
                matrix_cross_correlation[6, 0] = -0.37
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.61
                matrix_cross_correlation[6, 2] = -0.24
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = -0.24
                matrix_cross_correlation[5, 4] = 0.22
                matrix_cross_correlation[6, 4] = -0.13
                matrix_cross_correlation[6, 5] = -0.29
            elif class_ea == 9:
                matrix_cross_correlation[3, 2] = 0.65
                matrix_cross_correlation[4, 2] = 0.36
                matrix_cross_correlation[4, 0] = -0.03
                matrix_cross_correlation[3, 0] = -0.38
                matrix_cross_correlation[2, 0] = -0.19
                matrix_cross_correlation[4, 3] = 0.31
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.12
                matrix_cross_correlation[6, 0] = -0.33
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.54
                matrix_cross_correlation[6, 2] = -0.19
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = -0.2
                matrix_cross_correlation[5, 4] = 0.29
                matrix_cross_correlation[6, 4] = -0.35
                matrix_cross_correlation[6, 5] = -0.33

    elif scenario == 2 and type_path == 1:  # Suburban Scenario(LOS)
        matrix_cross_correlation[3, 2] = 0.4
        matrix_cross_correlation[4, 2] = 0.8
        matrix_cross_correlation[4, 0] = -0.5
        matrix_cross_correlation[3, 0] = -0.5
        matrix_cross_correlation[2, 0] = -0.4
        matrix_cross_correlation[4, 3] = 0
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = -0.2
        matrix_cross_correlation[2, 1] = -0.4
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0
        matrix_cross_correlation[6, 0] = -0.8
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = 0
        matrix_cross_correlation[5, 2] = -0.2
        matrix_cross_correlation[6, 2] = 0
        matrix_cross_correlation[5, 3] = 0.5
        matrix_cross_correlation[6, 3] = 0
        matrix_cross_correlation[5, 4] = -0.3
        matrix_cross_correlation[6, 4] = 0.4
        matrix_cross_correlation[6, 5] = 0

    elif scenario == 2 and type_path == 0:  # Suburban Scenario(NLOS)
        matrix_cross_correlation[3, 2] = 0.4
        matrix_cross_correlation[4, 2] = 0.6
        matrix_cross_correlation[4, 0] = 0
        matrix_cross_correlation[3, 0] = -0.6
        matrix_cross_correlation[2, 0] = -0.4
        matrix_cross_correlation[4, 3] = 0.4
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = 0
        matrix_cross_correlation[2, 1] = 0
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0
        matrix_cross_correlation[6, 0] = -0.4
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = 0
        matrix_cross_correlation[5, 2] = -0.5
        matrix_cross_correlation[6, 2] = 0
        matrix_cross_correlation[5, 3] = 0.5
        matrix_cross_correlation[6, 3] = -0.1
        matrix_cross_correlation[5, 4] = 0
        matrix_cross_correlation[6, 4] = 0
        matrix_cross_correlation[6, 5] = 0

    elif scenario == 3 and type_path == 1:  # Rural Scenario(LOS)
        matrix_cross_correlation[3, 2] = 0
        matrix_cross_correlation[4, 2] = 0
        matrix_cross_correlation[4, 0] = 0
        matrix_cross_correlation[3, 0] = 0
        matrix_cross_correlation[2, 0] = -0.5
        matrix_cross_correlation[4, 3] = 0
        matrix_cross_correlation[3, 1] = 0
        matrix_cross_correlation[4, 1] = 0
        matrix_cross_correlation[2, 1] = 0
        matrix_cross_correlation[1, 0] = 0
        matrix_cross_correlation[5, 0] = 0.01
        matrix_cross_correlation[6, 0] = -0.17
        matrix_cross_correlation[5, 1] = 0
        matrix_cross_correlation[6, 1] = -0.02
        matrix_cross_correlation[5, 2] = -0.05
        matrix_cross_correlation[6, 2] = 0.27
        matrix_cross_correlation[5, 3] = 0.73
        matrix_cross_correlation[6, 3] = -0.14
        matrix_cross_correlation[5, 4] = -0.2
        matrix_cross_correlation[6, 4] = 0.24
        matrix_cross_correlation[6, 5] = -0.07

    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                matrix_cross_correlation[3, 2] = 0.32
                matrix_cross_correlation[4, 2] = 0.3
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.45
                matrix_cross_correlation[2, 0] = -0.36
                matrix_cross_correlation[4, 3] = 0.45
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.06
                matrix_cross_correlation[6, 0] = -0.07
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.58
                matrix_cross_correlation[6, 2] = 0.06
                matrix_cross_correlation[5, 3] = 0.6
                matrix_cross_correlation[6, 3] = 0.21
                matrix_cross_correlation[5, 4] = 0.33
                matrix_cross_correlation[6, 4] = 0.1
                matrix_cross_correlation[6, 5] = 0.01
            elif class_ea == 2:
                matrix_cross_correlation[3, 2] = 0.19
                matrix_cross_correlation[4, 2] = 0.32
                matrix_cross_correlation[4, 0] = 0
                matrix_cross_correlation[3, 0] = 0.52
                matrix_cross_correlation[2, 0] = -0.39
                matrix_cross_correlation[4, 3] = 0.12
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.04
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.65
                matrix_cross_correlation[6, 2] = 0
                matrix_cross_correlation[5, 3] = 0.37
                matrix_cross_correlation[6, 3] = -0.09
                matrix_cross_correlation[5, 4] = 0.31
                matrix_cross_correlation[6, 4] = 0.21
                matrix_cross_correlation[6, 5] = -0.02
            elif class_ea == 3:
                matrix_cross_correlation[3, 2] = 0.23
                matrix_cross_correlation[4, 2] = 0.32
                matrix_cross_correlation[4, 0] = 0
                matrix_cross_correlation[3, 0] = 0.54
                matrix_cross_correlation[2, 0] = -0.41
                matrix_cross_correlation[4, 3] = 0.07
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.04
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.65
                matrix_cross_correlation[6, 2] = 0
                matrix_cross_correlation[5, 3] = 0.37
                matrix_cross_correlation[6, 3] = -0.09
                matrix_cross_correlation[5, 4] = 0.31
                matrix_cross_correlation[6, 4] = 0.22
                matrix_cross_correlation[6, 5] = -0.12
            elif class_ea == 4:
                matrix_cross_correlation[3, 2] = 0.25
                matrix_cross_correlation[4, 2] = 0.4
                matrix_cross_correlation[4, 0] = 0.01
                matrix_cross_correlation[3, 0] = 0.53
                matrix_cross_correlation[2, 0] = -0.37
                matrix_cross_correlation[4, 3] = 0.22
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.05
                matrix_cross_correlation[6, 0] = -0.17
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.73
                matrix_cross_correlation[6, 2] = -0.09
                matrix_cross_correlation[5, 3] = 0.32
                matrix_cross_correlation[6, 3] = -0.1
                matrix_cross_correlation[5, 4] = 0.37
                matrix_cross_correlation[6, 4] = 0.07
                matrix_cross_correlation[6, 5] = -0.21
            elif class_ea == 5:
                matrix_cross_correlation[3, 2] = 0.15
                matrix_cross_correlation[4, 2] = 0.45
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.55
                matrix_cross_correlation[2, 0] = -0.4
                matrix_cross_correlation[4, 3] = 0.16
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.06
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.79
                matrix_cross_correlation[6, 2] = -0.2
                matrix_cross_correlation[5, 3] = 0.19
                matrix_cross_correlation[6, 3] = -0.12
                matrix_cross_correlation[5, 4] = 0.46
                matrix_cross_correlation[6, 4] = 0.04
                matrix_cross_correlation[6, 5] = -0.27
            elif class_ea == 6:
                matrix_cross_correlation[3, 2] = 0.08
                matrix_cross_correlation[4, 2] = 0.39
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.56
                matrix_cross_correlation[2, 0] = -0.41
                matrix_cross_correlation[4, 3] = 0.14
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.07
                matrix_cross_correlation[6, 0] = -0.2
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.81
                matrix_cross_correlation[6, 2] = -0.22
                matrix_cross_correlation[5, 3] = 0.16
                matrix_cross_correlation[6, 3] = -0.11
                matrix_cross_correlation[5, 4] = 0.44
                matrix_cross_correlation[6, 4] = -0.12
                matrix_cross_correlation[6, 5] = -0.27
            elif class_ea == 7:
                matrix_cross_correlation[3, 2] = 0.13
                matrix_cross_correlation[4, 2] = 0.51
                matrix_cross_correlation[4, 0] = 0.04
                matrix_cross_correlation[3, 0] = 0.56
                matrix_cross_correlation[2, 0] = -0.4
                matrix_cross_correlation[4, 3] = 0.2
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.11
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.79
                matrix_cross_correlation[6, 2] = -0.32
                matrix_cross_correlation[5, 3] = 0.2
                matrix_cross_correlation[6, 3] = -0.1
                matrix_cross_correlation[5, 4] = 0.49
                matrix_cross_correlation[6, 4] = 0.29
                matrix_cross_correlation[6, 5] = -0.38
            elif class_ea == 8:
                matrix_cross_correlation[3, 2] = 0.15
                matrix_cross_correlation[4, 2] = 0.27
                matrix_cross_correlation[4, 0] = 0.01
                matrix_cross_correlation[3, 0] = 0.58
                matrix_cross_correlation[2, 0] = -0.46
                matrix_cross_correlation[4, 3] = -0.04
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.05
                matrix_cross_correlation[6, 0] = -0.23
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.7
                matrix_cross_correlation[6, 2] = -0.41
                matrix_cross_correlation[5, 3] = 0.15
                matrix_cross_correlation[6, 3] = -0.14
                matrix_cross_correlation[5, 4] = 0.27
                matrix_cross_correlation[6, 4] = -0.26
                matrix_cross_correlation[6, 5] = -0.35
            elif class_ea == 9:
                matrix_cross_correlation[3, 2] = 0.64
                matrix_cross_correlation[4, 2] = 0.05
                matrix_cross_correlation[4, 0] = 0.06
                matrix_cross_correlation[3, 0] = 0.47
                matrix_cross_correlation[2, 0] = -0.3
                matrix_cross_correlation[4, 3] = -0.11
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.1
                matrix_cross_correlation[6, 0] = -0.13
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.42
                matrix_cross_correlation[6, 2] = -0.35
                matrix_cross_correlation[5, 3] = 0.28
                matrix_cross_correlation[6, 3] = -0.25
                matrix_cross_correlation[5, 4] = 0.07
                matrix_cross_correlation[6, 4] = -0.36
                matrix_cross_correlation[6, 5] = -0.36

        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                matrix_cross_correlation[3, 2] = 0.33
                matrix_cross_correlation[4, 2] = 0.32
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.45
                matrix_cross_correlation[2, 0] = -0.36
                matrix_cross_correlation[4, 3] = 0.45
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.07
                matrix_cross_correlation[6, 0] = -0.06
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.55
                matrix_cross_correlation[6, 2] = 0.06
                matrix_cross_correlation[5, 3] = 0.61
                matrix_cross_correlation[6, 3] = 0.19
                matrix_cross_correlation[5, 4] = 0.38
                matrix_cross_correlation[6, 4] = 0.12
                matrix_cross_correlation[6, 5] = 0.05
            elif class_ea == 1:
                matrix_cross_correlation[3, 2] = 0.33
                matrix_cross_correlation[4, 2] = 0.32
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.45
                matrix_cross_correlation[2, 0] = -0.36
                matrix_cross_correlation[4, 3] = 0.45
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.07
                matrix_cross_correlation[6, 0] = -0.06
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.55
                matrix_cross_correlation[6, 2] = 0.06
                matrix_cross_correlation[5, 3] = 0.61
                matrix_cross_correlation[6, 3] = 0.19
                matrix_cross_correlation[5, 4] = 0.38
                matrix_cross_correlation[6, 4] = 0.12
                matrix_cross_correlation[6, 5] = 0.05
            elif class_ea == 2:
                matrix_cross_correlation[3, 2] = 0.24
                matrix_cross_correlation[4, 2] = 0.34
                matrix_cross_correlation[4, 0] = 0
                matrix_cross_correlation[3, 0] = 0.52
                matrix_cross_correlation[2, 0] = -0.38
                matrix_cross_correlation[4, 3] = 0.13
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.04
                matrix_cross_correlation[6, 0] = -0.16
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.65
                matrix_cross_correlation[6, 2] = 0.02
                matrix_cross_correlation[5, 3] = 0.41
                matrix_cross_correlation[6, 3] = -0.02
                matrix_cross_correlation[5, 4] = 0.35
                matrix_cross_correlation[6, 4] = 0.21
                matrix_cross_correlation[6, 5] = -0.03
            elif class_ea == 3:
                matrix_cross_correlation[3, 2] = 0.21
                matrix_cross_correlation[4, 2] = 0.33
                matrix_cross_correlation[4, 0] = 0
                matrix_cross_correlation[3, 0] = 0.54
                matrix_cross_correlation[2, 0] = -0.42
                matrix_cross_correlation[4, 3] = 0.08
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.04
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.64
                matrix_cross_correlation[6, 2] = 0.04
                matrix_cross_correlation[5, 3] = 0.39
                matrix_cross_correlation[6, 3] = -0.06
                matrix_cross_correlation[5, 4] = 0.33
                matrix_cross_correlation[6, 4] = 0.22
                matrix_cross_correlation[6, 5] = -0.08
            elif class_ea == 4:
                matrix_cross_correlation[3, 2] = 0.26
                matrix_cross_correlation[4, 2] = 0.43
                matrix_cross_correlation[4, 0] = 0.01
                matrix_cross_correlation[3, 0] = 0.53
                matrix_cross_correlation[2, 0] = -0.36
                matrix_cross_correlation[4, 3] = 0.21
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.05
                matrix_cross_correlation[6, 0] = -0.16
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.73
                matrix_cross_correlation[6, 2] = -0.06
                matrix_cross_correlation[5, 3] = 0.44
                matrix_cross_correlation[6, 3] = -0.08
                matrix_cross_correlation[5, 4] = 0.4
                matrix_cross_correlation[6, 4] = 0.11
                matrix_cross_correlation[6, 5] = -0.2
            elif class_ea == 5:
                matrix_cross_correlation[3, 2] = 0.16
                matrix_cross_correlation[4, 2] = 0.46
                matrix_cross_correlation[4, 0] = 0.01
                matrix_cross_correlation[3, 0] = 0.55
                matrix_cross_correlation[2, 0] = -0.39
                matrix_cross_correlation[4, 3] = 0.12
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.06
                matrix_cross_correlation[6, 0] = -0.19
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.78
                matrix_cross_correlation[6, 2] = -0.16
                matrix_cross_correlation[5, 3] = 0.15
                matrix_cross_correlation[6, 3] = -0.13
                matrix_cross_correlation[5, 4] = 0.46
                matrix_cross_correlation[6, 4] = 0.02
                matrix_cross_correlation[6, 5] = -0.25
            elif class_ea == 6:
                matrix_cross_correlation[3, 2] = 0.12
                matrix_cross_correlation[4, 2] = 0.38
                matrix_cross_correlation[4, 0] = 0.02
                matrix_cross_correlation[3, 0] = 0.56
                matrix_cross_correlation[2, 0] = -0.42
                matrix_cross_correlation[4, 3] = 0.15
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.06
                matrix_cross_correlation[6, 0] = -0.2
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.77
                matrix_cross_correlation[6, 2] = -0.17
                matrix_cross_correlation[5, 3] = 0.2
                matrix_cross_correlation[6, 3] = 0.09
                matrix_cross_correlation[5, 4] = 0.45
                matrix_cross_correlation[6, 4] = -0.08
                matrix_cross_correlation[6, 5] = 0.24
            elif class_ea == 7:
                matrix_cross_correlation[3, 2] = 0.29
                matrix_cross_correlation[4, 2] = 0.37
                matrix_cross_correlation[4, 0] = 0.04
                matrix_cross_correlation[3, 0] = 0.54
                matrix_cross_correlation[2, 0] = -0.36
                matrix_cross_correlation[4, 3] = 0.22
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.09
                matrix_cross_correlation[6, 0] = -0.17
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.74
                matrix_cross_correlation[6, 2] = -0.3
                matrix_cross_correlation[5, 3] = 0.3
                matrix_cross_correlation[6, 3] = -0.09
                matrix_cross_correlation[5, 4] = 0.33
                matrix_cross_correlation[6, 4] = -0.2
                matrix_cross_correlation[6, 5] = -0.37
            elif class_ea == 8:
                matrix_cross_correlation[3, 2] = 0.14
                matrix_cross_correlation[4, 2] = 0.28
                matrix_cross_correlation[4, 0] = 0.01
                matrix_cross_correlation[3, 0] = 0.57
                matrix_cross_correlation[2, 0] = -0.44
                matrix_cross_correlation[4, 3] = -0.03
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.06
                matrix_cross_correlation[6, 0] = -0.22
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.75
                matrix_cross_correlation[6, 2] = -0.35
                matrix_cross_correlation[5, 3] = 0.11
                matrix_cross_correlation[6, 3] = -0.14
                matrix_cross_correlation[5, 4] = 0.29
                matrix_cross_correlation[6, 4] = -0.16
                matrix_cross_correlation[6, 5] = -0.31
            elif class_ea == 9:
                matrix_cross_correlation[3, 2] = 0.59
                matrix_cross_correlation[4, 2] = 0.06
                matrix_cross_correlation[4, 0] = 0.04
                matrix_cross_correlation[3, 0] = 0.46
                matrix_cross_correlation[2, 0] = -0.27
                matrix_cross_correlation[4, 3] = -0.11
                matrix_cross_correlation[3, 1] = 0
                matrix_cross_correlation[4, 1] = 0
                matrix_cross_correlation[2, 1] = 0
                matrix_cross_correlation[1, 0] = 0
                matrix_cross_correlation[5, 0] = -0.08
                matrix_cross_correlation[6, 0] = -0.11
                matrix_cross_correlation[5, 1] = 0
                matrix_cross_correlation[6, 1] = 0
                matrix_cross_correlation[5, 2] = 0.52
                matrix_cross_correlation[6, 2] = -0.28
                matrix_cross_correlation[5, 3] = 0.41
                matrix_cross_correlation[6, 3] = -0.25
                matrix_cross_correlation[5, 4] = 0.06
                matrix_cross_correlation[6, 4] = -0.18
                matrix_cross_correlation[6, 5] = -0.32

    for m in range(7):
        for n in range(m + 1, 7):
            matrix_cross_correlation[m, n] = matrix_cross_correlation[n, m]

    return matrix_cross_correlation


# 阴影衰落表格（TS 38.811）
def shadow_fading(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    std = 0.0

    if scenario == 0:  # Dense Urban Scenario
        if type_freq == 0 and type_path == 1:
            if class_ea == 0:
                std = 3.5
            elif class_ea == 1:
                std = 3.5
            elif class_ea == 2:
                std = 3.4
            elif class_ea == 3:
                std = 2.9
            elif class_ea == 4:
                std = 3.0
            elif class_ea == 5:
                std = 3.1
            elif class_ea == 6:
                std = 2.7
            elif class_ea == 7:
                std = 2.5
            elif class_ea == 8:
                std = 2.3
            elif class_ea == 9:
                std = 1.2
        elif type_freq == 0 and type_path == 0:
            if class_ea == 0:
                std = 15.5
            elif class_ea == 1:
                std = 15.5
            elif class_ea == 2:
                std = 13.9
            elif class_ea == 3:
                std = 12.4
            elif class_ea == 4:
                std = 11.7
            elif class_ea == 5:
                std = 10.6
            elif class_ea == 6:
                std = 10.5
            elif class_ea == 7:
                std = 10.1
            elif class_ea == 8:
                std = 9.2
            elif class_ea == 9:
                std = 9.2
        elif type_freq == 1 and type_path == 1:
            if class_ea == 0:
                std = 2.9
            elif class_ea == 1:
                std = 2.9
            elif class_ea == 2:
                std = 2.4
            elif class_ea == 3:
                std = 2.7
            elif class_ea == 4:
                std = 2.4
            elif class_ea == 5:
                std = 2.4
            elif class_ea == 6:
                std = 2.7
            elif class_ea == 7:
                std = 2.6
            elif class_ea == 8:
                std = 2.8
            elif class_ea == 9:
                std = 0.6
        elif type_freq == 1 and type_path == 0:
            if class_ea == 0:
                std = 17.1
            elif class_ea == 1:
                std = 17.1
            elif class_ea == 2:
                std = 17.1
            elif class_ea == 3:
                std = 15.6
            elif class_ea == 4:
                std = 14.6
            elif class_ea == 5:
                std = 14.2
            elif class_ea == 6:
                std = 12.6
            elif class_ea == 7:
                std = 12.1
            elif class_ea == 8:
                std = 12.3
            elif class_ea == 9:
                std = 12.3
    elif scenario == 1:  # Urban Scenario
        if type_freq == 0 and type_path == 1:
            std = 4
        elif type_freq == 0 and type_path == 0:
            std = 6
        elif type_freq == 1 and type_path == 1:
            std = 4
        elif type_freq == 1 and type_path == 0:
            std = 6
    elif scenario in [2, 3]:  # Suburban and Rural Scenarios
        if type_freq == 0 and type_path == 1:
            if class_ea == 0:
                std = 1.79
            elif class_ea == 1:
                std = 1.79
            elif class_ea == 2:
                std = 1.14
            elif class_ea == 3:
                std = 1.14
            elif class_ea == 4:
                std = 0.92
            elif class_ea == 5:
                std = 1.42
            elif class_ea == 6:
                std = 1.56
            elif class_ea == 7:
                std = 0.85
            elif class_ea == 8:
                std = 0.72
            elif class_ea == 9:
                std = 0.72
        elif type_freq == 0 and type_path == 0:
            if class_ea == 0:
                std = 8.93
            elif class_ea == 1:
                std = 8.93
            elif class_ea == 2:
                std = 9.08
            elif class_ea == 3:
                std = 8.78
            elif class_ea == 4:
                std = 10.25
            elif class_ea == 5:
                std = 10.56
            elif class_ea == 6:
                std = 10.74
            elif class_ea == 7:
                std = 10.17
            elif class_ea == 8:
                std = 11.52
            elif class_ea == 9:
                std = 11.52
        elif type_freq == 1 and type_path == 1:
            if class_ea == 0:
                std = 1.9
            elif class_ea == 1:
                std = 1.9
            elif class_ea == 2:
                std = 1.6
            elif class_ea == 3:
                std = 1.9
            elif class_ea == 4:
                std = 2.3
            elif class_ea == 5:
                std = 2.7
            elif class_ea == 6:
                std = 3.1
            elif class_ea == 7:
                std = 3.0
            elif class_ea == 8:
                std = 3.6
            elif class_ea == 9:
                std = 0.4
        elif type_freq == 1 and type_path == 0:
            if class_ea == 0:
                std = 10.7
            elif class_ea == 1:
                std = 10.7
            elif class_ea == 2:
                std = 10.0
            elif class_ea == 3:
                std = 11.2
            elif class_ea == 4:
                std = 11.6
            elif class_ea == 5:
                std = 11.8
            elif class_ea == 6:
                std = 10.8
            elif class_ea == 7:
                std = 10.8
            elif class_ea == 8:
                std = 10.8
            elif class_ea == 9:
                std = 10.8

    table_sf = np.zeros(2)
    table_sf[0] = 0
    table_sf[1] = std

    return table_sf


# 莱斯因子表格（TS 38.811）
def k_factor(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = 4.4
                std = 3.3
            elif class_ea == 1:
                mean = 4.4
                std = 3.3
            elif class_ea == 2:
                mean = 9.0
                std = 6.6
            elif class_ea == 3:
                mean = 9.3
                std = 6.1
            elif class_ea == 4:
                mean = 7.9
                std = 4.0
            elif class_ea == 5:
                mean = 7.4
                std = 3.0
            elif class_ea == 6:
                mean = 7.0
                std = 2.6
            elif class_ea == 7:
                mean = 6.9
                std = 2.2
            elif class_ea == 8:
                mean = 6.5
                std = 2.1
            elif class_ea == 9:
                mean = 6.8
                std = 1.9
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = 6.1
                std = 2.6
            elif class_ea == 1:
                mean = 6.1
                std = 2.6
            elif class_ea == 2:
                mean = 13.7
                std = 6.8
            elif class_ea == 3:
                mean = 12.9
                std = 6.0
            elif class_ea == 4:
                mean = 10.3
                std = 3.3
            elif class_ea == 5:
                mean = 9.2
                std = 2.2
            elif class_ea == 6:
                mean = 8.4
                std = 1.9
            elif class_ea == 7:
                mean = 8.0
                std = 1.5
            elif class_ea == 8:
                mean = 7.4
                std = 1.6
            elif class_ea == 9:
                mean = 7.6
                std = 1.3
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        mean = 0
        std = 0
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = 31.83
                std = 13.84
            elif class_ea == 1:
                mean = 31.83
                std = 13.84
            elif class_ea == 2:
                mean = 18.78
                std = 13.78
            elif class_ea == 3:
                mean = 10.49
                std = 10.42
            elif class_ea == 4:
                mean = 7.46
                std = 8.01
            elif class_ea == 5:
                mean = 6.52
                std = 8.27
            elif class_ea == 6:
                mean = 5.47
                std = 7.26
            elif class_ea == 7:
                mean = 4.54
                std = 5.53
            elif class_ea == 8:
                mean = 4.03
                std = 4.49
            elif class_ea == 9:
                mean = 3.68
                std = 3.14
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = 40.18
                std = 16.99
            elif class_ea == 1:
                mean = 40.18
                std = 16.99
            elif class_ea == 2:
                mean = 23.62
                std = 18.96
            elif class_ea == 3:
                mean = 12.48
                std = 14.23
            elif class_ea == 4:
                mean = 8.56
                std = 11.06
            elif class_ea == 5:
                mean = 7.42
                std = 11.21
            elif class_ea == 6:
                mean = 5.97
                std = 9.47
            elif class_ea == 7:
                mean = 4.88
                std = 7.24
            elif class_ea == 8:
                mean = 4.22
                std = 5.79
            elif class_ea == 9:
                mean = 3.81
                std = 4.25
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        mean = 0
        std = 0
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = 11.40
                std = 6.26
            elif class_ea == 1:
                mean = 11.40
                std = 6.26
            elif class_ea == 2:
                mean = 19.45
                std = 10.32
            elif class_ea == 3:
                mean = 20.80
                std = 16.34
            elif class_ea == 4:
                mean = 21.20
                std = 15.63
            elif class_ea == 5:
                mean = 21.60
                std = 14.22
            elif class_ea == 6:
                mean = 19.75
                std = 14.19
            elif class_ea == 7:
                mean = 12.00
                std = 5.70
            elif class_ea == 8:
                mean = 12.85
                std = 9.91
            elif class_ea == 9:
                mean = 12.85
                std = 9.91
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = 8.9
                std = 4.4
            elif class_ea == 1:
                mean = 8.9
                std = 4.4
            elif class_ea == 2:
                mean = 14.0
                std = 4.6
            elif class_ea == 3:
                mean = 11.3
                std = 3.7
            elif class_ea == 4:
                mean = 9.0
                std = 3.5
            elif class_ea == 5:
                mean = 7.5
                std = 3.0
            elif class_ea == 6:
                mean = 6.6
                std = 2.6
            elif class_ea == 7:
                mean = 5.9
                std = 1.7
            elif class_ea == 8:
                mean = 5.5
                std = 0.7
            elif class_ea == 9:
                mean = 5.4
                std = 0.3
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        mean = 0
        std = 0
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = 24.72
                std = 5.07
            elif class_ea == 1:
                mean = 24.72
                std = 5.07
            elif class_ea == 2:
                mean = 12.31
                std = 5.75
            elif class_ea == 3:
                mean = 8.05
                std = 5.46
            elif class_ea == 4:
                mean = 6.21
                std = 5.23
            elif class_ea == 5:
                mean = 5.04
                std = 3.95
            elif class_ea == 6:
                mean = 4.42
                std = 3.75
            elif class_ea == 7:
                mean = 3.92
                std = 2.56
            elif class_ea == 8:
                mean = 3.65
                std = 1.77
            elif class_ea == 9:
                mean = 3.59
                std = 1.77
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = 25.43
                std = 7.04
            elif class_ea == 1:
                mean = 25.43
                std = 7.04
            elif class_ea == 2:
                mean = 12.72
                std = 7.47
            elif class_ea == 3:
                mean = 8.40
                std = 7.18
            elif class_ea == 4:
                mean = 6.52
                std = 6.88
            elif class_ea == 5:
                mean = 5.24
                std = 5.28
            elif class_ea == 6:
                mean = 4.57
                std = 4.92
            elif class_ea == 7:
                mean = 4.02
                std = 3.40
            elif class_ea == 8:
                mean = 3.70
                std = 2.22
            elif class_ea == 9:
                mean = 3.62
                std = 2.28
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        mean = 0
        std = 0

    table_kf = np.zeros(2)
    table_kf[0] = mean
    table_kf[1] = std

    return table_kf


# 时延偏移表格（TS 38.811）
def spread_delay(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -7.12
                std = 0.80
            elif class_ea == 1:
                mean = -7.12
                std = 0.80
            elif class_ea == 2:
                mean = -7.28
                std = 0.67
            elif class_ea == 3:
                mean = -7.45
                std = 0.68
            elif class_ea == 4:
                mean = -7.73
                std = 0.66
            elif class_ea == 5:
                mean = -7.91
                std = 0.62
            elif class_ea == 6:
                mean = -8.14
                std = 0.51
            elif class_ea == 7:
                mean = -8.23
                std = 0.45
            elif class_ea == 8:
                mean = -8.28
                std = 0.31
            elif class_ea == 9:
                mean = -8.36
                std = 0.08
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -7.43
                std = 0.90
            elif class_ea == 1:
                mean = -7.43
                std = 0.90
            elif class_ea == 2:
                mean = -7.62
                std = 0.78
            elif class_ea == 3:
                mean = -7.76
                std = 0.80
            elif class_ea == 4:
                mean = -8.02
                std = 0.72
            elif class_ea == 5:
                mean = -8.13
                std = 0.61
            elif class_ea == 6:
                mean = -8.30
                std = 0.47
            elif class_ea == 7:
                mean = -8.34
                std = 0.39
            elif class_ea == 8:
                mean = -8.39
                std = 0.26
            elif class_ea == 9:
                mean = -8.45
                std = 0.01
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -6.84
                std = 0.82
            elif class_ea == 1:
                mean = -6.84
                std = 0.82
            elif class_ea == 2:
                mean = -6.81
                std = 0.61
            elif class_ea == 3:
                mean = -6.94
                std = 0.49
            elif class_ea == 4:
                mean = -7.14
                std = 0.49
            elif class_ea == 5:
                mean = -7.34
                std = 0.51
            elif class_ea == 6:
                mean = -7.53
                std = 0.47
            elif class_ea == 7:
                mean = -7.67
                std = 0.44
            elif class_ea == 8:
                mean = -7.82
                std = 0.42
            elif class_ea == 9:
                mean = -7.84
                std = 0.55
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -6.86
                std = 0.81
            elif class_ea == 1:
                mean = -6.86
                std = 0.81
            elif class_ea == 2:
                mean = -6.84
                std = 0.61
            elif class_ea == 3:
                mean = -7.00
                std = 0.56
            elif class_ea == 4:
                mean = -7.21
                std = 0.56
            elif class_ea == 5:
                mean = -7.42
                std = 0.57
            elif class_ea == 6:
                mean = -7.86
                std = 0.55
            elif class_ea == 7:
                mean = -7.76
                std = 0.47
            elif class_ea == 8:
                mean = -8.07
                std = 0.42
            elif class_ea == 9:
                mean = -7.95
                std = 0.59
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -7.97
                std = 1.00
            elif class_ea == 1:
                mean = -7.97
                std = 1.00
            elif class_ea == 2:
                mean = -8.12
                std = 0.83
            elif class_ea == 3:
                mean = -8.21
                std = 0.68
            elif class_ea == 4:
                mean = -8.31
                std = 0.48
            elif class_ea == 5:
                mean = -8.37
                std = 0.38
            elif class_ea == 6:
                mean = -8.39
                std = 0.24
            elif class_ea == 7:
                mean = -8.38
                std = 0.18
            elif class_ea == 8:
                mean = -8.35
                std = 0.13
            elif class_ea == 9:
                mean = -8.34
                std = 0.09
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -8.52
                std = 0.92
            elif class_ea == 1:
                mean = -8.52
                std = 0.92
            elif class_ea == 2:
                mean = -8.59
                std = 0.79
            elif class_ea == 3:
                mean = -8.51
                std = 0.65
            elif class_ea == 4:
                mean = -8.49
                std = 0.48
            elif class_ea == 5:
                mean = -8.48
                std = 0.46
            elif class_ea == 6:
                mean = -8.44
                std = 0.34
            elif class_ea == 7:
                mean = -8.40
                std = 0.27
            elif class_ea == 8:
                mean = -8.37
                std = 0.19
            elif class_ea == 9:
                mean = -8.35
                std = 0.14
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -7.21
                std = 1.19
            elif class_ea == 1:
                mean = -7.21
                std = 1.19
            elif class_ea == 2:
                mean = -7.63
                std = 0.98
            elif class_ea == 3:
                mean = -7.75
                std = 0.84
            elif class_ea == 4:
                mean = -7.97
                std = 0.73
            elif class_ea == 5:
                mean = -7.99
                std = 0.73
            elif class_ea == 6:
                mean = -8.01
                std = 0.72
            elif class_ea == 7:
                mean = -8.09
                std = 0.71
            elif class_ea == 8:
                mean = -7.97
                std = 0.78
            elif class_ea == 9:
                mean = -8.17
                std = 0.67
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -7.24
                std = 1.26
            elif class_ea == 1:
                mean = -7.24
                std = 1.26
            elif class_ea == 2:
                mean = -7.70
                std = 0.99
            elif class_ea == 3:
                mean = -7.82
                std = 0.86
            elif class_ea == 4:
                mean = -8.04
                std = 0.75
            elif class_ea == 5:
                mean = -8.08
                std = 0.77
            elif class_ea == 6:
                mean = -8.10
                std = 0.76
            elif class_ea == 7:
                mean = -8.16
                std = 0.73
            elif class_ea == 8:
                mean = -8.03
                std = 0.79
            elif class_ea == 9:
                mean = -8.33
                std = 0.70
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -8.16
                std = 0.99
            elif class_ea == 1:
                mean = -8.16
                std = 0.99
            elif class_ea == 2:
                mean = -8.56
                std = 0.96
            elif class_ea == 3:
                mean = -8.72
                std = 0.79
            elif class_ea == 4:
                mean = -8.71
                std = 0.81
            elif class_ea == 5:
                mean = -8.72
                std = 1.12
            elif class_ea == 6:
                mean = -8.66
                std = 1.23
            elif class_ea == 7:
                mean = -8.38
                std = 0.55
            elif class_ea == 8:
                mean = -8.34
                std = 0.63
            elif class_ea == 9:
                mean = -8.34
                std = 0.63
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -8.07
                std = 0.46
            elif class_ea == 1:
                mean = -8.07
                std = 0.46
            elif class_ea == 2:
                mean = -8.61
                std = 0.45
            elif class_ea == 3:
                mean = -8.72
                std = 0.28
            elif class_ea == 4:
                mean = -8.63
                std = 0.17
            elif class_ea == 5:
                mean = -8.54
                std = 0.14
            elif class_ea == 6:
                mean = -8.48
                std = 0.15
            elif class_ea == 7:
                mean = -8.42
                std = 0.09
            elif class_ea == 8:
                mean = -8.39
                std = 0.05
            elif class_ea == 9:
                mean = -8.37
                std = 0.02
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -7.91
                std = 1.42
            elif class_ea == 1:
                mean = -7.91
                std = 1.42
            elif class_ea == 2:
                mean = -8.39
                std = 1.46
            elif class_ea == 3:
                mean = -8.69
                std = 1.46
            elif class_ea == 4:
                mean = -8.59
                std = 1.21
            elif class_ea == 5:
                mean = -8.64
                std = 1.18
            elif class_ea == 6:
                mean = -8.74
                std = 1.13
            elif class_ea == 7:
                mean = -8.98
                std = 1.37
            elif class_ea == 8:
                mean = -9.28
                std = 1.50
            elif class_ea == 9:
                mean = -9.28
                std = 1.50
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -7.43
                std = 0.50
            elif class_ea == 1:
                mean = -7.43
                std = 0.50
            elif class_ea == 2:
                mean = -7.63
                std = 0.61
            elif class_ea == 3:
                mean = -7.86
                std = 0.56
            elif class_ea == 4:
                mean = -7.96
                std = 0.58
            elif class_ea == 5:
                mean = -7.98
                std = 0.59
            elif class_ea == 6:
                mean = -8.45
                std = 0.47
            elif class_ea == 7:
                mean = -8.21
                std = 0.36
            elif class_ea == 8:
                mean = -8.69
                std = 0.29
            elif class_ea == 9:
                mean = -8.69
                std = 0.29
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -9.55
                std = 0.66
            elif class_ea == 1:
                mean = -9.55
                std = 0.66
            elif class_ea == 2:
                mean = -8.68
                std = 0.44
            elif class_ea == 3:
                mean = -8.46
                std = 0.28
            elif class_ea == 4:
                mean = -8.36
                std = 0.19
            elif class_ea == 5:
                mean = -8.29
                std = 0.14
            elif class_ea == 6:
                mean = -8.26
                std = 0.10
            elif class_ea == 7:
                mean = -8.22
                std = 0.10
            elif class_ea == 8:
                mean = -8.20
                std = 0.05
            elif class_ea == 9:
                mean = -8.19
                std = 0.06
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -9.68
                std = 0.46
            elif class_ea == 1:
                mean = -9.68
                std = 0.46
            elif class_ea == 2:
                mean = -8.86
                std = 0.29
            elif class_ea == 3:
                mean = -8.59
                std = 0.18
            elif class_ea == 4:
                mean = -8.46
                std = 0.19
            elif class_ea == 5:
                mean = -8.36
                std = 0.14
            elif class_ea == 6:
                mean = -8.30
                std = 0.15
            elif class_ea == 7:
                mean = -8.26
                std = 0.13
            elif class_ea == 8:
                mean = -8.22
                std = 0.03
            elif class_ea == 9:
                mean = -8.21
                std = 0.07
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0:
                mean = -9.01
                std = 1.59
            elif class_ea == 1:
                mean = -9.01
                std = 1.59
            elif class_ea == 2:
                mean = -8.37
                std = 0.95
            elif class_ea == 3:
                mean = -8.05
                std = 0.92
            elif class_ea == 4:
                mean = -7.92
                std = 0.92
            elif class_ea == 5:
                mean = -7.92
                std = 0.87
            elif class_ea == 6:
                mean = -7.96
                std = 0.87
            elif class_ea == 7:
                mean = -7.91
                std = 0.82
            elif class_ea == 8:
                mean = -7.79
                std = 0.86
            elif class_ea == 9:
                mean = -7.74
                std = 0.81
        elif type_freq == 1:  # Ka-band
            if class_ea == 0:
                mean = -9.13
                std = 1.91
            elif class_ea == 1:
                mean = -9.13
                std = 1.91
            elif class_ea == 2:
                mean = -8.39
                std = 0.94
            elif class_ea == 3:
                mean = -8.10
                std = 0.92
            elif class_ea == 4:
                mean = -7.96
                std = 0.94
            elif class_ea == 5:
                mean = -7.99
                std = 0.89
            elif class_ea == 6:
                mean = -8.05
                std = 0.87
            elif class_ea == 7:
                mean = -8.01
                std = 0.82
            elif class_ea == 8:
                mean = -8.05
                std = 1.65
            elif class_ea == 9:
                mean = -7.91
                std = 0.76

    table_ds = np.zeros(2)
    table_ds[0] = mean
    table_ds[1] = std

    return table_ds


# asd表格（TS 38.811）
def spread_aod(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.06
                std = 0.48
            elif class_ea == 2:
                mean = -2.68
                std = 0.36
            elif class_ea == 3:
                mean = -2.51
                std = 0.38
            elif class_ea == 4:
                mean = -2.40
                std = 0.32
            elif class_ea == 5:
                mean = -2.31
                std = 0.33
            elif class_ea == 6:
                mean = -2.20
                std = 0.39
            elif class_ea == 7:
                mean = -2.00
                std = 0.540
            elif class_ea == 8:
                mean = -1.64
                std = 0.32
            elif class_ea == 9:
                mean = -0.63
                std = 0.53
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.43
                std = 0.54
            elif class_ea == 2:
                mean = -3.06
                std = 0.41
            elif class_ea == 3:
                mean = -2.91
                std = 0.42
            elif class_ea == 4:
                mean = -2.81
                std = 0.34
            elif class_ea == 5:
                mean = -2.74
                std = 0.34
            elif class_ea == 6:
                mean = -2.72
                std = 0.70
            elif class_ea == 7:
                mean = -2.46
                std = 0.40
            elif class_ea == 8:
                mean = -2.30
                std = 0.78
            elif class_ea == 9:
                mean = -1.11
                std = 0.51
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.08
                std = 0.87
            elif class_ea == 2:
                mean = -1.68
                std = 0.73
            elif class_ea == 3:
                mean = -1.46
                std = 0.53
            elif class_ea == 4:
                mean = -1.43
                std = 0.50
            elif class_ea == 5:
                mean = -1.44
                std = 0.58
            elif class_ea == 6:
                mean = -1.33
                std = 0.49
            elif class_ea == 7:
                mean = -1.31
                std = 0.65
            elif class_ea == 8:
                mean = -1.11
                std = 0.69
            elif class_ea == 9:
                mean = -0.11
                std = 0.53
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.12
                std = 0.94
            elif class_ea == 2:
                mean = -1.74
                std = 0.79
            elif class_ea == 3:
                mean = -1.56
                std = 0.66
            elif class_ea == 4:
                mean = -1.54
                std = 0.63
            elif class_ea == 5:
                mean = -1.45
                std = 0.56
            elif class_ea == 6:
                mean = -1.64
                std = 0.78
            elif class_ea == 7:
                mean = -1.37
                std = 0.56
            elif class_ea == 8:
                mean = -1.29
                std = 0.76
            elif class_ea == 9:
                mean = -0.41
                std = 0.59
    elif scenario == 1 and type_path == 1:  # Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.60
                std = 0.79
            elif class_ea == 2:
                mean = -2.48
                std = 0.80
            elif class_ea == 3:
                mean = -2.44
                std = 0.91
            elif class_ea == 4:
                mean = -2.60
                std = 1.02
            elif class_ea == 5:
                mean = -2.71
                std = 1.17
            elif class_ea == 6:
                mean = -2.76
                std = 1.17
            elif class_ea == 7:
                mean = -2.78
                std = 1.20
            elif class_ea == 8:
                mean = -2.65
                std = 1.45
            elif class_ea == 9:
                mean = -2.27
                std = 1.85
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.18
                std = 0.79
            elif class_ea == 2:
                mean = -3.05
                std = 0.87
            elif class_ea == 3:
                mean = -2.98
                std = 1.04
            elif class_ea == 4:
                mean = -3.11
                std = 1.06
            elif class_ea == 5:
                mean = -3.19
                std = 1.12
            elif class_ea == 6:
                mean = -3.25
                std = 1.14
            elif class_ea == 7:
                mean = -3.33
                std = 1.25
            elif class_ea == 8:
                mean = -3.22
                std = 1.35
            elif class_ea == 9:
                mean = -2.83
                std = 1.62
    elif scenario == 1 and type_path == 0:  # Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -1.55
                std = 0.87
            elif class_ea == 2:
                mean = -1.61
                std = 0.88
            elif class_ea == 3:
                mean = -1.73
                std = 1.15
            elif class_ea == 4:
                mean = -1.95
                std = 1.13
            elif class_ea == 5:
                mean = -1.94
                std = 1.21
            elif class_ea == 6:
                mean = -1.88
                std = 0.99
            elif class_ea == 7:
                mean = -2.10
                std = 1.77
            elif class_ea == 8:
                mean = -1.80
                std = 1.54
            elif class_ea == 9:
                mean = -1.77
                std = 1.40
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -1.58
                std = 0.89
            elif class_ea == 2:
                mean = -1.67
                std = 0.89
            elif class_ea == 3:
                mean = -1.84
                std = 1.30
            elif class_ea == 4:
                mean = -2.02
                std = 1.15
            elif class_ea == 5:
                mean = -2.06
                std = 1.23
            elif class_ea == 6:
                mean = -1.99
                std = 1.02
            elif class_ea == 7:
                mean = -2.19
                std = 1.78
            elif class_ea == 8:
                mean = -1.88
                std = 1.55
            elif class_ea == 9:
                mean = -2.00
                std = 1.40
    elif scenario == 2 and type_path == 1:  # Suburban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.57
                std = 1.62
            elif class_ea == 2:
                mean = -3.80
                std = 1.74
            elif class_ea == 3:
                mean = -3.77
                std = 1.72
            elif class_ea == 4:
                mean = -3.57
                std = 1.60
            elif class_ea == 5:
                mean = -3.42
                std = 1.49
            elif class_ea == 6:
                mean = -3.27
                std = 1.43
            elif class_ea == 7:
                mean = -3.08
                std = 1.36
            elif class_ea == 8 or class_ea == 9:
                mean = -2.75
                std = 1.26
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.55
                std = 0.48
            elif class_ea == 2:
                mean = -3.69
                std = 0.41
            elif class_ea == 3:
                mean = -3.59
                std = 0.41
            elif class_ea == 4:
                mean = -3.38
                std = 0.35
            elif class_ea == 5:
                mean = -3.23
                std = 0.35
            elif class_ea == 6:
                mean = -3.19
                std = 0.43
            elif class_ea == 7:
                mean = -2.83
                std = 0.33
            elif class_ea == 8:
                mean = -2.66
                std = 0.44
            elif class_ea == 9:
                mean = -1.22
                std = 0.31
    elif scenario == 2 and type_path == 0:  # Suburban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.54
                std = 1.80
            elif class_ea == 2:
                mean = -3.63
                std = 1.43
            elif class_ea == 3:
                mean = -3.66
                std = 1.68
            elif class_ea == 4:
                mean = -3.66
                std = 1.48
            elif class_ea == 5:
                mean = -3.66
                std = 1.55
            elif class_ea == 6:
                mean = -3.57
                std = 1.38
            elif class_ea == 7:
                mean = -3.18
                std = 1.62
            elif class_ea == 8 or class_ea == 9:
                mean = -2.71
                std = 1.63
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.89
                std = 0.41
            elif class_ea == 2:
                mean = -2.76
                std = 0.41
            elif class_ea == 3:
                mean = -2.64
                std = 0.41
            elif class_ea == 4:
                mean = -2.41
                std = 0.52
            elif class_ea == 5:
                mean = -2.42
                std = 0.70
            elif class_ea == 6:
                mean = -2.53
                std = 0.50
            elif class_ea == 7:
                mean = -2.35
                std = 0.58
            elif class_ea == 8 or class_ea == 9:
                mean = -2.31
                std = 0.73
    elif scenario == 3 and type_path == 1:  # Rural Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -3.42
                std = 0.89
            elif class_ea == 2:
                mean = -3.00
                std = 0.63
            elif class_ea == 3:
                mean = -2.86
                std = 0.52
            elif class_ea == 4:
                mean = -2.78
                std = 0.45
            elif class_ea == 5:
                mean = -2.70
                std = 0.42
            elif class_ea == 6:
                mean = -2.66
                std = 0.41
            elif class_ea == 7:
                mean = -2.53
                std = 0.42
            elif class_ea == 8:
                mean = -2.21
                std = 0.50
            elif class_ea == 9:
                mean = -1.78
                std = 0.91
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -4.03
                std = 0.91
            elif class_ea == 2:
                mean = -3.55
                std = 0.70
            elif class_ea == 3:
                mean = -3.45
                std = 0.55
            elif class_ea == 4:
                mean = -3.38
                std = 0.52
            elif class_ea == 5:
                mean = -3.33
                std = 0.46
            elif class_ea == 6:
                mean = -3.29
                std = 0.43
            elif class_ea == 7:
                mean = -3.24
                std = 0.46
            elif class_ea == 8:
                mean = -2.90
                std = 0.44
            elif class_ea == 9:
                mean = -2.50
                std = 0.82
    elif scenario == 3 and type_path == 0:  # Rural Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.90
                std = 1.34
            elif class_ea == 2:
                mean = -2.50
                std = 1.18
            elif class_ea == 3:
                mean = -2.12
                std = 1.08
            elif class_ea == 4:
                mean = -1.99
                std = 1.06
            elif class_ea == 5:
                mean = -1.90
                std = 1.05
            elif class_ea == 6:
                mean = -1.85
                std = 1.06
            elif class_ea == 7:
                mean = -1.69
                std = 1.14
            elif class_ea == 8:
                mean = -1.46
                std = 1.16
            elif class_ea == 9:
                mean = -1.32
                std = 1.30
        elif type_freq == 1:  # Ka-band
            if class_ea == 0 or class_ea == 1:
                mean = -2.9
                std = 1.32
            elif class_ea == 2:
                mean = -2.53
                std = 1.18
            elif class_ea == 3:
                mean = -2.16
                std = 1.08
            elif class_ea == 4:
                mean = -2.04
                std = 1.09
            elif class_ea == 5:
                mean = -1.99
                std = 1.08
            elif class_ea == 6:
                mean = -1.95
                std = 1.06
            elif class_ea == 7:
                mean = -1.81
                std = 1.17
            elif class_ea == 8:
                mean = -1.56
                std = 1.20
            elif class_ea == 9:
                mean = -1.53
                std = 1.27

    table_asd = np.zeros(2)
    table_asd[0] = mean
    table_asd[1] = std

    return table_asd


# asa表格（TS 38.811）
def spread_aoa(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.94
                std = 0.70
            elif class_ea == 2:
                mean = 0.87
                std = 0.66
            elif class_ea == 3:
                mean = 0.92
                std = 0.68
            elif class_ea == 4:
                mean = 0.79
                std = 0.64
            elif class_ea == 5:
                mean = 0.72
                std = 0.63
            elif class_ea == 6:
                mean = 0.60
                std = 0.54
            elif class_ea == 7:
                mean = 0.55
                std = 0.52
            elif class_ea == 8:
                mean = 0.71
                std = 0.53
            elif class_ea == 9:
                mean = 0.81
                std = 0.62
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.65
                std = 0.82
            elif class_ea == 2:
                mean = 0.53
                std = 0.78
            elif class_ea == 3:
                mean = 0.60
                std = 0.83
            elif class_ea == 4:
                mean = 0.43
                std = 0.78
            elif class_ea == 5:
                mean = 0.36
                std = 0.77
            elif class_ea == 6:
                mean = 0.16
                std = 0.84
            elif class_ea == 7:
                mean = 0.18
                std = 0.64
            elif class_ea == 8:
                mean = 0.24
                std = 0.81
            elif class_ea == 9:
                mean = 0.36
                std = 0.65

    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 1.00
                std = 1.60
            elif class_ea == 2:
                mean = 1.44
                std = 0.87
            elif class_ea == 3:
                mean = 1.54
                std = 0.64
            elif class_ea == 4:
                mean = 1.53
                std = 0.56
            elif class_ea == 5:
                mean = 1.48
                std = 0.54
            elif class_ea == 6:
                mean = 1.39
                std = 0.68
            elif class_ea == 7:
                mean = 1.42
                std = 0.55
            elif class_ea == 8:
                mean = 1.38
                std = 0.60
            elif class_ea == 9:
                mean = 1.23
                std = 0.60
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 1.02
                std = 1.44
            elif class_ea == 2:
                mean = 1.44
                std = 0.77
            elif class_ea == 3:
                mean = 1.48
                std = 0.70
            elif class_ea == 4:
                mean = 1.46
                std = 0.60
            elif class_ea == 5:
                mean = 1.40
                std = 0.59
            elif class_ea == 6:
                mean = 0.97
                std = 1.27
            elif class_ea == 7:
                mean = 1.33
                std = 0.56
            elif class_ea == 8:
                mean = 1.12
                std = 1.04
            elif class_ea == 9:
                mean = 1.04
                std = 0.63

    elif scenario == 1 and type_path == 1:  # Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.18
                std = 0.74
            elif class_ea == 2:
                mean = 0.42
                std = 0.90
            elif class_ea == 3:
                mean = 0.41
                std = 1.30
            elif class_ea == 4:
                mean = 0.18
                std = 1.69
            elif class_ea == 5:
                mean = -0.07
                std = 2.04
            elif class_ea == 6:
                mean = -0.43
                std = 2.54
            elif class_ea == 7:
                mean = -0.64
                std = 2.47
            elif class_ea == 8:
                mean = -0.91
                std = 2.69
            elif class_ea == 9:
                mean = -0.54
                std = 1.66
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -0.40
                std = 0.77
            elif class_ea == 2:
                mean = -0.15
                std = 0.97
            elif class_ea == 3:
                mean = -0.18
                std = 1.58
            elif class_ea == 4:
                mean = -0.31
                std = 1.69
            elif class_ea == 5:
                mean = -0.58
                std = 2.13
            elif class_ea == 6:
                mean = -0.90
                std = 2.51
            elif class_ea == 7:
                mean = -1.16
                std = 2.47
            elif class_ea == 8:
                mean = -1.48
                std = 2.61
            elif class_ea == 9:
                mean = -1.14
                std = 1.70

    elif scenario == 1 and type_path == 0:  # Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.17
                std = 2.97
            elif class_ea == 2:
                mean = 0.32
                std = 2.99
            elif class_ea == 3:
                mean = 0.52
                std = 2.71
            elif class_ea == 4:
                mean = 0.61
                std = 2.26
            elif class_ea == 5:
                mean = 0.68
                std = 2.08
            elif class_ea == 6:
                mean = 0.64
                std = 1.93
            elif class_ea == 7:
                mean = 0.58
                std = 1.71
            elif class_ea == 8:
                mean = 0.71
                std = 0.96
            elif class_ea == 9:
                mean = 0.17
                std = 1.16
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.13
                std = 2.99
            elif class_ea == 2:
                mean = 0.19
                std = 3.12
            elif class_ea == 3:
                mean = 0.44
                std = 2.69
            elif class_ea == 4:
                mean = 0.48
                std = 2.45
            elif class_ea == 5:
                mean = 0.56
                std = 2.17
            elif class_ea == 6:
                mean = 0.55
                std = 1.93
            elif class_ea == 7:
                mean = 0.48
                std = 1.72
            elif class_ea == 8:
                mean = 0.53
                std = 1.51
            elif class_ea == 9:
                mean = 0.32
                std = 1.20

    elif scenario == 2 and type_path == 1:  # Suburban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.05
                std = 1.84
            elif class_ea == 2:
                mean = -0.38
                std = 1.94
            elif class_ea == 3:
                mean = -0.56
                std = 1.75
            elif class_ea == 4:
                mean = -0.59
                std = 1.82
            elif class_ea == 5:
                mean = -0.58
                std = 1.87
            elif class_ea == 6:
                mean = -0.55
                std = 1.92
            elif class_ea in [7, 8, 9]:
                mean = -0.28
                std = 1.16
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.89
                std = 0.67
            elif class_ea == 2:
                mean = 0.31
                std = 0.78
            elif class_ea == 3:
                mean = 0.02
                std = 0.75
            elif class_ea == 4:
                mean = -0.10
                std = 0.65
            elif class_ea == 5:
                mean = -0.19
                std = 0.55
            elif class_ea == 6:
                mean = -0.54
                std = 0.96
            elif class_ea == 7:
                mean = -0.24
                std = 0.43
            elif class_ea == 8:
                mean = -0.52
                std = 0.93
            elif class_ea == 9:
                mean = -0.15
                std = 0.44

    elif scenario == 2 and type_path == 0:  # Suburban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.91
                std = 1.70
            elif class_ea == 2:
                mean = 0.70
                std = 1.33
            elif class_ea == 3:
                mean = 0.38
                std = 1.52
            elif class_ea == 4:
                mean = 0.30
                std = 1.46
            elif class_ea == 5:
                mean = 0.28
                std = 1.44
            elif class_ea == 6:
                mean = 0.23
                std = 1.44
            elif class_ea == 7:
                mean = 0.10
                std = 1.24
            elif class_ea in [8, 9]:
                mean = 0.04
                std = 1.04
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 1.49
                std = 0.40
            elif class_ea == 2:
                mean = 1.24
                std = 0.82
            elif class_ea == 3:
                mean = 1.06
                std = 0.71
            elif class_ea == 4:
                mean = 0.91
                std = 0.55
            elif class_ea == 5:
                mean = 0.98
                std = 0.58
            elif class_ea == 6:
                mean = 0.49
                std = 1.37
            elif class_ea == 7:
                mean = 0.73
                std = 0.49
            elif class_ea == 8 or class_ea == 9:
                mean = -0.04
                std = 1.48

    elif scenario == 3 and type_path == 1:  # Rural Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -9.45
                std = 7.83
            elif class_ea == 2:
                mean = -4.45
                std = 6.86
            elif class_ea == 3:
                mean = -2.39
                std = 5.14
            elif class_ea == 4:
                mean = -1.28
                std = 3.44
            elif class_ea == 5:
                mean = -0.99
                std = 2.59
            elif class_ea == 6:
                mean = -1.05
                std = 2.42
            elif class_ea == 7:
                mean = -0.90
                std = 1.78
            elif class_ea == 8:
                mean = -0.89
                std = 1.65
            elif class_ea == 9:
                mean = -0.81
                std = 1.26
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -9.74
                std = 7.52
            elif class_ea == 2:
                mean = -4.88
                std = 6.67
            elif class_ea == 3:
                mean = -2.60
                std = 4.63
            elif class_ea == 4:
                mean = -1.92
                std = 3.45
            elif class_ea == 5:
                mean = -1.56
                std = 2.44
            elif class_ea == 6:
                mean = -1.66
                std = 2.38
            elif class_ea == 7:
                mean = -1.59
                std = 1.67
            elif class_ea == 8:
                mean = -1.58
                std = 1.44
            elif class_ea == 9:
                mean = -1.51
                std = 1.13

    elif scenario == 3 and type_path == 0:  # Rural Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -3.33
                std = 6.22
            elif class_ea == 2:
                mean = -0.74
                std = 4.22
            elif class_ea == 3:
                mean = 0.08
                std = 3.02
            elif class_ea == 4:
                mean = 0.32
                std = 2.45
            elif class_ea == 5:
                mean = 0.53
                std = 1.63
            elif class_ea == 6:
                mean = 0.33
                std = 2.08
            elif class_ea == 7:
                mean = 0.55
                std = 1.58
            elif class_ea == 8:
                mean = 0.45
                std = 2.01
            elif class_ea == 9:
                mean = 0.40
                std = 2.19
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -3.4
                std = 6.28
            elif class_ea == 2:
                mean = -0.51
                std = 3.75
            elif class_ea == 3:
                mean = 0.06
                std = 2.95
            elif class_ea == 4:
                mean = 0.20
                std = 2.65
            elif class_ea == 5:
                mean = 0.40
                std = 1.85
            elif class_ea == 6:
                mean = 0.32
                std = 1.83
            elif class_ea == 7:
                mean = 0.46
                std = 1.57
            elif class_ea == 8:
                mean = 0.33
                std = 1.99
            elif class_ea == 9:
                mean = 0.24
                std = 2.18

    table_asa = np.zeros(2)
    table_asa[0] = mean
    table_asa[1] = std

    return table_asa


# zsd表格（TS 38.811）
def spread_zod(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -2.52
                std = 0.50
            elif class_ea == 2:
                mean = -2.29
                std = 0.53
            elif class_ea == 3:
                mean = -2.19
                std = 0.58
            elif class_ea == 4:
                mean = -2.24
                std = 0.51
            elif class_ea == 5:
                mean = -2.30
                std = 0.46
            elif class_ea == 6:
                mean = -2.48
                std = 0.35
            elif class_ea == 7:
                mean = -2.64
                std = 0.31
            elif class_ea == 8:
                mean = -2.68
                std = 0.39
            elif class_ea == 9:
                mean = -2.61
                std = 0.28
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -2.75
                std = 0.55
            elif class_ea == 2:
                mean = -2.64
                std = 0.64
            elif class_ea == 3:
                mean = -2.49
                std = 0.69
            elif class_ea == 4:
                mean = -2.51
                std = 0.57
            elif class_ea == 5:
                mean = -2.54
                std = 0.50
            elif class_ea == 6:
                mean = -2.71
                std = 0.37
            elif class_ea == 7:
                mean = -2.85
                std = 0.31
            elif class_ea == 8:
                mean = -3.01
                std = 0.45
            elif class_ea == 9:
                mean = -3.08
                std = 0.27

    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -2.08
                std = 0.58
            elif class_ea == 2:
                mean = -1.66
                std = 0.50
            elif class_ea == 3:
                mean = -1.48
                std = 0.40
            elif class_ea == 4:
                mean = -1.46
                std = 0.37
            elif class_ea == 5:
                mean = -1.53
                std = 0.47
            elif class_ea == 6:
                mean = -1.61
                std = 0.43
            elif class_ea == 7:
                mean = -1.77
                std = 0.50
            elif class_ea == 8:
                mean = -1.90
                std = 0.42
            elif class_ea == 9:
                mean = -1.99
                std = 0.50
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -2.11
                std = 0.59
            elif class_ea == 2:
                mean = -1.69
                std = 0.51
            elif class_ea == 3:
                mean = -1.52
                std = 0.46
            elif class_ea == 4:
                mean = -1.51
                std = 0.43
            elif class_ea == 5:
                mean = -1.54
                std = 0.45
            elif class_ea == 6:
                mean = -1.84
                std = 0.63
            elif class_ea == 7:
                mean = -1.86
                std = 0.51
            elif class_ea == 8:
                mean = -2.16
                std = 0.74
            elif class_ea == 9:
                mean = -2.21
                std = 0.61

    elif scenario == 1 and type_path == 1:  # Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -2.54
                std = 2.62
            elif class_ea == 2:
                mean = -2.67
                std = 2.96
            elif class_ea == 3:
                mean = -2.03
                std = 0.86
            elif class_ea == 4:
                mean = -2.28
                std = 1.19
            elif class_ea == 5:
                mean = -2.48
                std = 1.40
            elif class_ea == 6:
                mean = -2.56
                std = 0.85
            elif class_ea == 7:
                mean = -2.96
                std = 1.61
            elif class_ea == 8:
                mean = -3.08
                std = 1.49
            elif class_ea == 9:
                mean = -3.00
                std = 1.09
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -2.61
                std = 2.41
            elif class_ea == 2:
                mean = -2.82
                std = 2.59
            elif class_ea == 3:
                mean = -2.48
                std = 1.02
            elif class_ea == 4:
                mean = -2.76
                std = 1.27
            elif class_ea == 5:
                mean = -2.93
                std = 1.38
            elif class_ea == 6:
                mean = -3.05
                std = 0.96
            elif class_ea == 7:
                mean = -3.45
                std = 1.51
            elif class_ea == 8:
                mean = -3.66
                std = 1.49
            elif class_ea == 9:
                mean = -3.56
                std = 0.89

    elif scenario == 1 and type_path == 0:  # Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -2.86
                std = 2.77
            elif class_ea == 2:
                mean = -2.64
                std = 2.79
            elif class_ea == 3:
                mean = -2.05
                std = 1.53
            elif class_ea == 4:
                mean = -2.18
                std = 1.67
            elif class_ea == 5:
                mean = -2.24
                std = 1.95
            elif class_ea == 6:
                mean = -2.21
                std = 1.87
            elif class_ea == 7:
                mean = -2.69
                std = 2.72
            elif class_ea == 8:
                mean = -2.81
                std = 2.98
            elif class_ea == 9:
                mean = -4.29
                std = 4.37
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -2.87
                std = 2.76
            elif class_ea == 2:
                mean = -2.68
                std = 2.76
            elif class_ea == 3:
                mean = -2.12
                std = 1.54
            elif class_ea == 4:
                mean = -2.27
                std = 1.77
            elif class_ea == 5:
                mean = -2.50
                std = 2.36
            elif class_ea == 6:
                mean = -2.47
                std = 2.33
            elif class_ea == 7:
                mean = -2.83
                std = 2.84
            elif class_ea == 8:
                mean = -2.82
                std = 2.87
            elif class_ea == 9:
                mean = -4.55
                std = 4.27

    elif scenario == 2 and type_path == 1:  # Suburban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -1.06
                std = 0.96
            elif class_ea == 2:
                mean = -1.21
                std = 0.95
            elif class_ea == 3:
                mean = -1.28
                std = 0.49
            elif class_ea == 4:
                mean = -1.32
                std = 0.79
            elif class_ea == 5:
                mean = -1.39
                std = 0.97
            elif class_ea == 6:
                mean = -1.36
                std = 1.17
            elif class_ea == 7:
                mean = -1.08
                std = 0.62
            elif class_ea in [8, 9]:
                mean = -1.31
                std = 0.76
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -3.37
                std = 0.28
            elif class_ea == 2:
                mean = -3.28
                std = 0.27
            elif class_ea == 3:
                mean = -3.04
                std = 0.26
            elif class_ea == 4:
                mean = -2.88
                std = 0.21
            elif class_ea == 5:
                mean = -2.83
                std = 0.18
            elif class_ea == 6:
                mean = -2.86
                std = 0.17
            elif class_ea == 7:
                mean = -2.95
                std = 0.10
            elif class_ea == 8:
                mean = -3.21
                std = 0.07
            elif class_ea == 9:
                mean = -3.49
                std = 0.24

    elif scenario == 2 and type_path == 0:  # Suburban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -2.01
                std = 1.79
            elif class_ea == 2:
                mean = -1.67
                std = 1.31
            elif class_ea == 3:
                mean = -1.75
                std = 1.42
            elif class_ea == 4:
                mean = -1.49
                std = 1.28
            elif class_ea == 5:
                mean = -1.53
                std = 1.40
            elif class_ea == 6:
                mean = -1.57
                std = 1.24
            elif class_ea == 7:
                mean = -1.48
                std = 0.98
            elif class_ea in [8, 9]:
                mean = -1.62
                std = 0.88
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -3.09
                std = 0.32
            elif class_ea == 2:
                mean = -2.93
                std = 0.47
            elif class_ea == 3:
                mean = -2.91
                std = 0.46
            elif class_ea == 4:
                mean = -2.78
                std = 0.54
            elif class_ea == 5:
                mean = -2.70
                std = 0.45
            elif class_ea == 6:
                mean = -3.03
                std = 0.36
            elif class_ea == 7:
                mean = -2.90
                std = 0.42
            elif class_ea in [8, 9]:
                mean = -3.20
                std = 0.30

    elif scenario == 3 and type_path == 1:  # Rural Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -6.03
                std = 5.19
            elif class_ea == 2:
                mean = -4.31
                std = 4.18
            elif class_ea == 3:
                mean = -2.57
                std = 0.61
            elif class_ea == 4:
                mean = -2.59
                std = 0.79
            elif class_ea == 5:
                mean = -2.59
                std = 0.65
            elif class_ea == 6:
                mean = -2.65
                std = 0.52
            elif class_ea == 7:
                mean = -2.69
                std = 0.78
            elif class_ea == 8:
                mean = -2.65
                std = 1.01
            elif class_ea == 9:
                mean = -2.65
                std = 0.71
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -7.45
                std = 5.30
            elif class_ea == 2:
                mean = -5.25
                std = 4.42
            elif class_ea == 3:
                mean = -3.16
                std = 0.68
            elif class_ea == 4:
                mean = -3.15
                std = 0.73
            elif class_ea == 5:
                mean = -3.20
                std = 0.77
            elif class_ea == 6:
                mean = -3.27
                std = 0.61
            elif class_ea == 7:
                mean = -3.42
                std = 0.74
            elif class_ea == 8:
                mean = -3.36
                std = 0.79
            elif class_ea == 9:
                mean = -3.35
                std = 0.65

    elif scenario == 3 and type_path == 0:  # Rural Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -4.92
                std = 3.96
            elif class_ea == 2:
                mean = -4.06
                std = 4.07
            elif class_ea == 3:
                mean = -2.33
                std = 1.70
            elif class_ea == 4:
                mean = -2.24
                std = 2.01
            elif class_ea == 5:
                mean = -2.24
                std = 2.00
            elif class_ea == 6:
                mean = -2.22
                std = 1.82
            elif class_ea == 7:
                mean = -2.19
                std = 1.66
            elif class_ea == 8:
                mean = -2.41
                std = 2.58
            elif class_ea == 9:
                mean = -2.45
                std = 2.52
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -5.47
                std = 4.39
            elif class_ea == 2:
                mean = -4.06
                std = 4.04
            elif class_ea == 3:
                mean = -2.32
                std = 1.54
            elif class_ea == 4:
                mean = -2.19
                std = 1.73
            elif class_ea == 5:
                mean = -2.16
                std = 1.50
            elif class_ea == 6:
                mean = -2.24
                std = 1.64
            elif class_ea == 7:
                mean = -2.29
                std = 1.66
            elif class_ea == 8:
                mean = -2.65
                std = 2.86
            elif class_ea == 9:
                mean = -2.23
                std = 1.12

    table_zsd = np.zeros(2)
    table_zsd[0] = mean
    table_zsd[1] = std

    return table_zsd


# zsa表格（TS 38.811）
def spread_zoa(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 0.82
                std = 0.03
            elif class_ea == 2:
                mean = 0.50
                std = 0.09
            elif class_ea == 3:
                mean = 0.82
                std = 0.05
            elif class_ea == 4:
                mean = 1.23
                std = 0.03
            elif class_ea == 5:
                mean = 1.43
                std = 0.06
            elif class_ea == 6:
                mean = 1.56
                std = 0.05
            elif class_ea == 7:
                mean = 1.66
                std = 0.05
            elif class_ea == 8:
                mean = 1.73
                std = 0.02
            elif class_ea == 9:
                mean = 1.79
                std = 0.01
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.82
                std = 0.05
            elif class_ea == 2:
                mean = 0.47
                std = 0.11
            elif class_ea == 3:
                mean = 0.80
                std = 0.05
            elif class_ea == 4:
                mean = 1.23
                std = 0.04
            elif class_ea == 5:
                mean = 1.42
                std = 0.10
            elif class_ea == 6:
                mean = 1.56
                std = 0.06
            elif class_ea == 7:
                mean = 1.65
                std = 0.07
            elif class_ea == 8:
                mean = 1.73
                std = 0.02
            elif class_ea == 9:
                mean = 1.79
                std = 0.01

    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 1.00
                std = 0.63
            elif class_ea == 2:
                mean = 0.94
                std = 0.65
            elif class_ea == 3:
                mean = 1.15
                std = 0.42
            elif class_ea == 4:
                mean = 1.35
                std = 0.28
            elif class_ea == 5:
                mean = 1.44
                std = 0.25
            elif class_ea == 6:
                mean = 1.56
                std = 0.16
            elif class_ea == 7:
                mean = 1.64
                std = 0.18
            elif class_ea == 8:
                mean = 1.70
                std = 0.09
            elif class_ea == 9:
                mean = 1.70
                std = 0.17
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 1.01
                std = 0.56
            elif class_ea == 2:
                mean = 0.96
                std = 0.55
            elif class_ea == 3:
                mean = 1.13
                std = 0.43
            elif class_ea == 4:
                mean = 1.30
                std = 0.37
            elif class_ea == 5:
                mean = 1.40
                std = 0.32
            elif class_ea == 6:
                mean = 1.41
                std = 0.45
            elif class_ea == 7:
                mean = 1.63
                std = 0.17
            elif class_ea == 8:
                mean = 1.68
                std = 0.14
            elif class_ea == 9:
                mean = 1.70
                std = 0.17

    elif scenario == 1 and type_path == 1:  # Urban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -0.63
                std = 2.60
            elif class_ea == 2:
                mean = -0.15
                std = 3.31
            elif class_ea == 3:
                mean = 0.54
                std = 1.10
            elif class_ea == 4:
                mean = 0.35
                std = 1.59
            elif class_ea == 5:
                mean = 0.27
                std = 1.62
            elif class_ea == 6:
                mean = 0.26
                std = 0.97
            elif class_ea == 7:
                mean = -0.12
                std = 1.99
            elif class_ea == 8:
                mean = -0.21
                std = 1.82
            elif class_ea == 9:
                mean = -0.07
                std = 1.43
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -0.67
                std = 2.22
            elif class_ea == 2:
                mean = -0.34
                std = 3.04
            elif class_ea == 3:
                mean = 0.07
                std = 1.33
            elif class_ea == 4:
                mean = -0.08
                std = 1.45
            elif class_ea == 5:
                mean = -0.21
                std = 1.62
            elif class_ea == 6:
                mean = -0.25
                std = 1.06
            elif class_ea == 7:
                mean = -0.61
                std = 1.88
            elif class_ea == 8:
                mean = -0.79
                std = 1.87
            elif class_ea == 9:
                mean = -0.58
                std = 1.19

    elif scenario == 1 and type_path == 0:  # Urban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -0.97
                std = 2.35
            elif class_ea == 2:
                mean = 0.49
                std = 2.11
            elif class_ea == 3:
                mean = 1.03
                std = 1.29
            elif class_ea == 4:
                mean = 1.12
                std = 1.45
            elif class_ea == 5:
                mean = 1.30
                std = 1.07
            elif class_ea == 6:
                mean = 1.32
                std = 1.20
            elif class_ea == 7:
                mean = 1.35
                std = 1.10
            elif class_ea == 8:
                mean = 1.31
                std = 1.35
            elif class_ea == 9:
                mean = 1.50
                std = 0.56
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -1.13
                std = 2.66
            elif class_ea == 2:
                mean = 0.49
                std = 2.03
            elif class_ea == 3:
                mean = 0.95
                std = 1.54
            elif class_ea == 4:
                mean = 1.15
                std = 1.02
            elif class_ea == 5:
                mean = 1.14
                std = 1.61
            elif class_ea == 6:
                mean = 1.13
                std = 1.84
            elif class_ea == 7:
                mean = 1.16
                std = 1.81
            elif class_ea == 8:
                mean = 1.28
                std = 1.35
            elif class_ea == 9:
                mean = 1.42
                std = 0.60

    elif scenario == 2 and type_path == 1:  # Suburban Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -1.78
                std = 0.62
            elif class_ea == 2:
                mean = -1.84
                std = 0.81
            elif class_ea == 3:
                mean = -1.67
                std = 0.57
            elif class_ea == 4:
                mean = -1.59
                std = 0.86
            elif class_ea == 5:
                mean = -1.55
                std = 1.05
            elif class_ea == 6:
                mean = -1.51
                std = 1.23
            elif class_ea == 7:
                mean = -1.27
                std = 0.54
            elif class_ea in [8, 9]:
                mean = -1.28
                std = 0.67
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.63
                std = 0.35
            elif class_ea == 2:
                mean = 0.76
                std = 0.30
            elif class_ea == 3:
                mean = 1.11
                std = 0.28
            elif class_ea == 4:
                mean = 1.37
                std = 0.23
            elif class_ea == 5:
                mean = 1.53
                std = 0.23
            elif class_ea == 6:
                mean = 1.65
                std = 0.17
            elif class_ea == 7:
                mean = 1.74
                std = 0.11
            elif class_ea == 8:
                mean = 1.82
                std = 0.05
            elif class_ea == 9:
                mean = 1.87
                std = 0.02

    elif scenario == 2 and type_path == 0:  # Suburban Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -1.90
                std = 1.63
            elif class_ea == 2:
                mean = -1.70
                std = 1.24
            elif class_ea == 3:
                mean = -1.75
                std = 1.54
            elif class_ea == 4:
                mean = -1.80
                std = 1.25
            elif class_ea == 5:
                mean = -1.80
                std = 1.21
            elif class_ea == 6:
                mean = -1.85
                std = 1.20
            elif class_ea == 7:
                mean = -1.45
                std = 1.38
            elif class_ea in [8, 9]:
                mean = -1.19
                std = 1.58
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 0.81
                std = 0.36
            elif class_ea == 2:
                mean = 1.06
                std = 0.41
            elif class_ea == 3:
                mean = 1.12
                std = 0.40
            elif class_ea == 4:
                mean = 1.14
                std = 0.39
            elif class_ea == 5:
                mean = 1.29
                std = 0.35
            elif class_ea == 6:
                mean = 1.38
                std = 0.36
            elif class_ea == 7:
                mean = 1.36
                std = 0.29
            elif class_ea in [8, 9]:
                mean = 1.38
                std = 0.20

    elif scenario == 3 and type_path == 1:  # Rural Scenario(LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -4.20
                std = 6.30
            elif class_ea == 2:
                mean = -2.31
                std = 5.04
            elif class_ea == 3:
                mean = -0.28
                std = 0.81
            elif class_ea == 4:
                mean = -0.38
                std = 1.16
            elif class_ea == 5:
                mean = -0.38
                std = 0.82
            elif class_ea == 6:
                mean = -0.46
                std = 0.67
            elif class_ea == 7:
                mean = -0.49
                std = 1.00
            elif class_ea == 8:
                mean = -0.53
                std = 1.18
            elif class_ea == 9:
                mean = -0.46
                std = 0.91
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -5.85
                std = 6.51
            elif class_ea == 2:
                mean = -3.27
                std = 5.36
            elif class_ea == 3:
                mean = -0.88
                std = 0.93
            elif class_ea == 4:
                mean = -0.93
                std = 0.96
            elif class_ea == 5:
                mean = -0.99
                std = 0.97
            elif class_ea == 6:
                mean = -1.04
                std = 0.83
            elif class_ea == 7:
                mean = -1.17
                std = 1.01
            elif class_ea == 8:
                mean = -1.19
                std = 1.01
            elif class_ea == 9:
                mean = -1.13
                std = 0.85

    elif scenario == 3 and type_path == 0:  # Rural Scenario(NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = -0.88
                std = 3.26
            elif class_ea == 2:
                mean = -0.07
                std = 3.29
            elif class_ea == 3:
                mean = 0.75
                std = 1.92
            elif class_ea == 4:
                mean = 0.72
                std = 1.92
            elif class_ea == 5:
                mean = 0.95
                std = 1.45
            elif class_ea == 6:
                mean = 0.97
                std = 1.62
            elif class_ea == 7:
                mean = 1.10
                std = 1.43
            elif class_ea == 8:
                mean = 0.97
                std = 1.88
            elif class_ea == 9:
                mean = 1.35
                std = 0.62
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = -1.19
                std = 3.81
            elif class_ea == 2:
                mean = -0.11
                std = 3.33
            elif class_ea == 3:
                mean = 0.72
                std = 1.93
            elif class_ea == 4:
                mean = 0.69
                std = 1.91
            elif class_ea == 5:
                mean = 0.84
                std = 1.70
            elif class_ea == 6:
                mean = 0.99
                std = 1.27
            elif class_ea == 7:
                mean = 0.95
                std = 1.86
            elif class_ea == 8:
                mean = 0.92
                std = 1.84
            elif class_ea == 9:
                mean = 1.29
                std = 0.59

    table_zsa = np.zeros(2)
    table_zsa[0] = mean
    table_zsa[1] = std

    return table_zsa


# 杂波损耗表格（TS 38.811）
def generate_clutter_loss(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    loss_clutter = 0.0

    if type_path == 1:  # 当UE处于LOS状态时，杂波损耗可以忽略不计
        loss_clutter = 0.0
    else:
        if scenario == 0:  # Dense Urban Scenario
            if type_freq == 0:  # S-band
                if class_ea == 0:
                    loss_clutter = 34.3
                elif class_ea == 1:
                    loss_clutter = 34.3
                elif class_ea == 2:
                    loss_clutter = 30.9
                elif class_ea == 3:
                    loss_clutter = 29.0
                elif class_ea == 4:
                    loss_clutter = 27.7
                elif class_ea == 5:
                    loss_clutter = 26.8
                elif class_ea == 6:
                    loss_clutter = 26.2
                elif class_ea == 7:
                    loss_clutter = 25.8
                elif class_ea == 8:
                    loss_clutter = 25.5
                elif class_ea == 9:
                    loss_clutter = 25.5
            elif type_freq == 1:  # Ka-band
                if class_ea == 0:
                    loss_clutter = 44.3
                elif class_ea == 1:
                    loss_clutter = 44.3
                elif class_ea == 2:
                    loss_clutter = 39.9
                elif class_ea == 3:
                    loss_clutter = 37.5
                elif class_ea == 4:
                    loss_clutter = 35.8
                elif class_ea == 5:
                    loss_clutter = 34.6
                elif class_ea == 6:
                    loss_clutter = 33.8
                elif class_ea == 7:
                    loss_clutter = 33.3
                elif class_ea == 8:
                    loss_clutter = 33.0
                elif class_ea == 9:
                    loss_clutter = 32.9
        elif scenario == 1:  # Urban Scenario
            if type_freq == 0:  # S-band
                if class_ea == 0:
                    loss_clutter = 34.3
                elif class_ea == 1:
                    loss_clutter = 34.3
                elif class_ea == 2:
                    loss_clutter = 30.9
                elif class_ea == 3:
                    loss_clutter = 29.0
                elif class_ea == 4:
                    loss_clutter = 27.7
                elif class_ea == 5:
                    loss_clutter = 26.8
                elif class_ea == 6:
                    loss_clutter = 26.2
                elif class_ea == 7:
                    loss_clutter = 25.8
                elif class_ea == 8:
                    loss_clutter = 25.5
                elif class_ea == 9:
                    loss_clutter = 25.5
            elif type_freq == 1:  # Ka-band
                if class_ea == 0:
                    loss_clutter = 44.3
                elif class_ea == 1:
                    loss_clutter = 44.3
                elif class_ea == 2:
                    loss_clutter = 39.9
                elif class_ea == 3:
                    loss_clutter = 37.5
                elif class_ea == 4:
                    loss_clutter = 35.8
                elif class_ea == 5:
                    loss_clutter = 34.6
                elif class_ea == 6:
                    loss_clutter = 33.8
                elif class_ea == 7:
                    loss_clutter = 33.3
                elif class_ea == 8:
                    loss_clutter = 33.0
                elif class_ea == 9:
                    loss_clutter = 32.9
        elif scenario in [2, 3]:  # Suburban and Rural Scenarios
            if type_freq == 0:  # S-band
                if class_ea == 0:
                    loss_clutter = 19.52
                elif class_ea == 1:
                    loss_clutter = 19.52
                elif class_ea == 2:
                    loss_clutter = 18.17
                elif class_ea == 3:
                    loss_clutter = 18.42
                elif class_ea == 4:
                    loss_clutter = 18.28
                elif class_ea == 5:
                    loss_clutter = 18.63
                elif class_ea == 6:
                    loss_clutter = 17.68
                elif class_ea == 7:
                    loss_clutter = 16.50
                elif class_ea == 8:
                    loss_clutter = 16.30
                elif class_ea == 9:
                    loss_clutter = 16.30
            elif type_freq == 1:  # Ka-band
                if class_ea == 0:
                    loss_clutter = 29.5
                elif class_ea == 1:
                    loss_clutter = 29.5
                elif class_ea == 2:
                    loss_clutter = 24.6
                elif class_ea == 3:
                    loss_clutter = 21.9
                elif class_ea == 4:
                    loss_clutter = 20.0
                elif class_ea == 5:
                    loss_clutter = 18.7
                elif class_ea == 6:
                    loss_clutter = 17.8
                elif class_ea == 7:
                    loss_clutter = 17.2
                elif class_ea == 8:
                    loss_clutter = 16.9
                elif class_ea == 9:
                    loss_clutter = 16.8
    return loss_clutter


# 延迟缩放参数表格（TS 38.811）
def generate_delay_scaling_param(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    delay_ray = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        delay_ray = 2.5
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        delay_ray = 2.3
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        delay_ray = 2.5
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        delay_ray = 2.3
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                delay_ray = 2.20
            elif class_ea == 2:
                delay_ray = 3.36
            elif class_ea == 3:
                delay_ray = 3.50
            elif class_ea == 4:
                delay_ray = 2.81
            elif class_ea == 5:
                delay_ray = 2.39
            elif class_ea == 6:
                delay_ray = 2.73
            elif class_ea == 7:
                delay_ray = 2.07
            elif class_ea == 8 or class_ea == 9:
                delay_ray = 2.04
        elif type_freq == 1:  # Ka-band
            delay_ray = 2.5
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea == 0 or class_ea == 1:
                delay_ray = 2.28
            elif class_ea == 2:
                delay_ray = 2.33
            elif class_ea == 3:
                delay_ray = 2.43
            elif class_ea == 4:
                delay_ray = 2.26
            elif class_ea == 5:
                delay_ray = 2.71
            elif class_ea == 6:
                delay_ray = 2.10
            elif class_ea == 7:
                delay_ray = 2.19
            elif class_ea == 8 or class_ea == 9:
                delay_ray = 2.06
        elif type_freq == 1:  # Ka-band
            delay_ray = 2.3
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        delay_ray = 3.8
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        delay_ray = 1.7

    return delay_ray


# 每个簇的阴影标准差（TS 38.811）
def generate_shadowing_std():
    std_shadow = 3.0
    return std_shadow


# AOA/aod缩放因子（TS 38.811）
def generate_scaling_factor_4_azimuth_angle(num_cluster):
    if num_cluster <= 2:
        c_phi = 0.501
    elif 2 < num_cluster <= 3:
        c_phi = 0.68
    elif 3 < num_cluster <= 4:
        c_phi = 0.779
    elif 4 < num_cluster <= 5:
        c_phi = 0.86
    elif 5 < num_cluster <= 8:
        c_phi = 1.018
    elif 8 < num_cluster <= 10:
        c_phi = 1.09
    elif 10 < num_cluster <= 11:
        c_phi = 1.123
    elif 11 < num_cluster <= 12:
        c_phi = 1.146
    elif 12 < num_cluster <= 14:
        c_phi = 1.19
    elif 14 < num_cluster <= 15:
        c_phi = 1.211
    elif 15 < num_cluster <= 16:
        c_phi = 1.226
    elif 16 < num_cluster <= 19:
        c_phi = 1.273
    elif 19 < num_cluster <= 20:
        c_phi = 1.289
    else:
        warning = f'{num_cluster} cannot be found in the index of "total number of clusters"'
        print(warning)
        c_phi = None

    return c_phi


# ZOA/zod缩放因子（TS 38.811）
def generate_scaling_factor_4_zenith_angle(num_cluster):
    if num_cluster <= 2:
        c_theta = 0.43
    elif num_cluster <= 3:
        c_theta = 0.594
    elif num_cluster <= 4:
        c_theta = 0.697
    elif num_cluster <= 8:
        c_theta = 0.889
    elif num_cluster <= 10:
        c_theta = 0.957
    elif num_cluster <= 11:
        c_theta = 1.031
    elif num_cluster <= 12:
        c_theta = 1.104
    elif num_cluster <= 15:
        c_theta = 1.108
    elif num_cluster <= 19:
        c_theta = 1.184
    elif num_cluster <= 20:
        c_theta = 1.178
    else:
        warning = f'{num_cluster} cannot be found in the index of "total number of clusters"'
        print(warning)
        c_theta = None

    return c_theta


# 簇总数（TS 38.811）
def generate_cluster_num(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        num_cluster = 3
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        num_cluster = 4
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        if class_ea <= 1:
            num_cluster = 4
        else:
            num_cluster = 3
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        if class_ea <= 6:
            num_cluster = 3
        else:
            num_cluster = 2
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if class_ea <= 6:
            num_cluster = 3
        else:
            num_cluster = 2
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        if class_ea <= 5:
            num_cluster = 4
        else:
            num_cluster = 3
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        num_cluster = 2
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        if class_ea <= 2:
            num_cluster = 3
        else:
            num_cluster = 2
    else:
        num_cluster = None  # Handle any cases that do not match

    return num_cluster


# 簇内路径数量（TS 38.811）
def generate_ray_num():
    num_ray_cluster = 20

    return num_ray_cluster


# 路径偏移角（TS 38.811）
def RayOffsetAngle():
    da_ray = np.zeros(20)
    da_ray[0] = 0.0447
    da_ray[2] = 0.1413
    da_ray[4] = 0.2492
    da_ray[6] = 0.3715
    da_ray[8] = 0.5129
    da_ray[10] = 0.6797
    da_ray[12] = 0.8844
    da_ray[14] = 1.1481
    da_ray[16] = 1.5195
    da_ray[18] = 2.1551
    da_ray[1] = -0.0447
    da_ray[3] = -0.1413
    da_ray[5] = -0.2492
    da_ray[7] = -0.3715
    da_ray[9] = -0.5129
    da_ray[11] = -0.6797
    da_ray[13] = -0.8844
    da_ray[15] = -1.1481
    da_ray[17] = -1.5195
    da_ray[19] = -2.1551

    return da_ray


# 簇均方根扩展（TS 38.811）
def generate_rms_spread_4_cluster(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    # 向量 [c_DS(ns) c_asd(°) c_asa(°) c_zsa(°)]
    c_spread = np.zeros(4)

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            c_spread = [3.9, 0.0, 11.0, 7.0]
        elif type_freq == 1:  # Ka-band
            c_spread = [1.6, 0.0, 11.0, 7.0]
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        if type_freq == 0:  # S-band
            c_spread = [3.9, 0.0, 15.0, 7.0]
        elif type_freq == 1:  # Ka-band
            c_spread = [3.9, 0.0, 15.0, 7.0]
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                c_spread = [3.9, 0.09, 12.55, 1.25]
            elif class_ea == 2:
                c_spread = [3.9, 0.09, 12.76, 3.23]
            elif class_ea == 3:
                c_spread = [3.9, 0.12, 14.36, 4.39]
            elif class_ea == 4:
                c_spread = [3.9, 0.16, 16.42, 5.72]
            elif class_ea == 5:
                c_spread = [3.9, 0.2, 17.13, 6.17]
            elif class_ea == 6:
                c_spread = [3.9, 0.28, 19.01, 7.36]
            elif class_ea == 7:
                c_spread = [3.9, 0.44, 19.31, 7.3]
            elif class_ea == 8:
                c_spread = [3.9, 0.9, 22.39, 7.7]
            elif class_ea == 9:
                c_spread = [3.9, 2.87, 27.8, 9.25]
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                c_spread = [1.6, 0.09, 11.78, 1.14]
            elif class_ea == 2:
                c_spread = [1.6, 0.09, 11.6, 2.78]
            elif class_ea == 3:
                c_spread = [1.6, 0.11, 13.05, 3.87]
            elif class_ea == 4:
                c_spread = [1.6, 0.15, 14.56, 4.94]
            elif class_ea == 5:
                c_spread = [1.6, 0.18, 15.35, 5.41]
            elif class_ea == 6:
                c_spread = [1.6, 0.27, 16.97, 6.31]
            elif class_ea == 7:
                c_spread = [1.6, 0.42, 17.96, 6.66]
            elif class_ea == 8:
                c_spread = [1.6, 0.86, 20.68, 7.31]
            elif class_ea == 9:
                c_spread = [1.6, 2.55, 25.08, 9.23]
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                c_spread = [3.9, 0.08, 15.07, 1.66]
            elif class_ea == 2:
                c_spread = [3.9, 0.1, 16.2, 4.71]
            elif class_ea == 3:
                c_spread = [3.9, 0.14, 18.14, 7.33]
            elif class_ea == 4:
                c_spread = [3.9, 0.23, 19.96, 9.82]
            elif class_ea == 5:
                c_spread = [3.9, 0.33, 21.53, 11.52]
            elif class_ea == 6:
                c_spread = [3.9, 0.53, 22.44, 11.75]
            elif class_ea == 7:
                c_spread = [3.9, 1.0, 23.59, 10.93]
            elif class_ea == 8:
                c_spread = [3.9, 1.4, 26.57, 12.19]
            elif class_ea == 9:
                c_spread = [3.9, 6.63, 32.7, 16.68]
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                c_spread = [1.6, 0.08, 14.72, 1.57]
            elif class_ea == 2:
                c_spread = [1.6, 0.1, 14.62, 4.3]
            elif class_ea == 3:
                c_spread = [1.6, 0.14, 16.4, 6.64]
            elif class_ea == 4:
                c_spread = [1.6, 0.22, 17.86, 9.21]
            elif class_ea == 5:
                c_spread = [1.6, 0.31, 19.74, 10.32]
            elif class_ea == 6:
                c_spread = [1.6, 0.49, 19.73, 10.3]
            elif class_ea == 7:
                c_spread = [1.6, 0.97, 20.5, 10.2]
            elif class_ea == 8:
                c_spread = [1.6, 1.52, 28.16, 12.27]
            elif class_ea == 9:
                c_spread = [1.6, 5.36, 25.83, 12.75]
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if type_freq == 0:  # S-band
            c_spread = [1.6, 0.0, 11.0, 7.0]
        elif type_freq == 1:  # Ka-band
            c_spread = [1.6, 0.0, 11.0, 7.0]
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        if type_freq == 0:  # S-band
            c_spread = [1.6, 0.0, 15.0, 7.0]
        elif type_freq == 1:  # Ka-band
            c_spread = [1.6, 0.0, 15.0, 7.0]
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                c_spread = [0.0, 0.39, 10.81, 1.94]
            elif class_ea == 2:
                c_spread = [0.0, 0.31, 8.09, 1.83]
            elif class_ea == 3:
                c_spread = [0.0, 0.29, 13.7, 2.28]
            elif class_ea == 4:
                c_spread = [0.0, 0.37, 20.05, 2.93]
            elif class_ea == 5:
                c_spread = [0.0, 0.61, 24.51, 2.84]
            elif class_ea == 6:
                c_spread = [0.0, 0.9, 26.35, 3.17]
            elif class_ea == 7:
                c_spread = [0.0, 1.43, 31.84, 3.88]
            elif class_ea == 8:
                c_spread = [0.0, 2.87, 36.62, 4.17]
            elif class_ea == 9:
                c_spread = [0.0, 5.48, 36.77, 4.29]
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                c_spread = [0.0, 0.36, 4.63, 0.75]
            elif class_ea == 2:
                c_spread = [0.0, 0.3, 6.83, 1.25]
            elif class_ea == 3:
                c_spread = [0.0, 0.25, 12.91, 1.93]
            elif class_ea == 4:
                c_spread = [0.0, 0.35, 18.9, 2.37]
            elif class_ea == 5:
                c_spread = [0.0, 0.53, 22.44, 2.66]
            elif class_ea == 6:
                c_spread = [0.0, 0.88, 25.69, 3.23]
            elif class_ea == 7:
                c_spread = [0.0, 1.39, 27.95, 3.71]
            elif class_ea == 8:
                c_spread = [0.0, 2.7, 31.45, 4.17]
            elif class_ea == 9:
                c_spread = [0.0, 4.97, 28.01, 4.14]
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                c_spread = [0.0, 0.03, 18.16, 2.32]
            elif class_ea == 2:
                c_spread = [0.0, 0.05, 26.82, 7.34]
            elif class_ea == 3:
                c_spread = [0.0, 0.07, 21.99, 8.28]
            elif class_ea == 4:
                c_spread = [0.0, 0.1, 22.86, 8.76]
            elif class_ea == 5:
                c_spread = [0.0, 0.15, 25.93, 9.68]
            elif class_ea == 6:
                c_spread = [0.0, 0.22, 27.79, 9.94]
            elif class_ea == 7:
                c_spread = [0.0, 0.5, 28.5, 8.9]
            elif class_ea == 8:
                c_spread = [0.0, 1.04, 37.53, 13.74]
            elif class_ea == 9:
                c_spread = [0.0, 2.11, 29.23, 12.16]
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                c_spread = [0.0, 0.03, 18.21, 2.13]
            elif class_ea == 2:
                c_spread = [0.0, 0.05, 24.08, 6.52]
            elif class_ea == 3:
                c_spread = [0.0, 0.07, 22.06, 7.72]
            elif class_ea == 4:
                c_spread = [0.0, 0.09, 21.4, 8.45]
            elif class_ea == 5:
                c_spread = [0.0, 0.16, 24.26, 8.92]
            elif class_ea == 6:
                c_spread = [0.0, 0.22, 24.15, 8.76]
            elif class_ea == 7:
                c_spread = [0.0, 0.51, 25.99, 9.0]
            elif class_ea == 8:
                c_spread = [0.0, 0.89, 36.07, 13.6]
            elif class_ea == 9:
                c_spread = [0.0, 1.68, 24.51, 10.56]

    return c_spread


# 交叉极化功率比（TS 38.811）
def generate_power_ratio_4_x_pol(scenario, alpha, type_freq, type_path):
    class_ea = round(alpha / 10)
    mean = 0.0
    std = 0.0

    if scenario == 0 and type_path == 1:  # Dense Urban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 24.4
                std = 3.8
            elif class_ea == 2:
                mean = 23.6
                std = 4.7
            elif class_ea == 3:
                mean = 23.2
                std = 4.6
            elif class_ea == 4:
                mean = 22.6
                std = 4.9
            elif class_ea == 5:
                mean = 21.8
                std = 5.7
            elif class_ea == 6:
                mean = 20.5
                std = 6.9
            elif class_ea == 7:
                mean = 19.3
                std = 8.1
            elif class_ea == 8:
                mean = 17.4
                std = 10.3
            elif class_ea == 9:
                mean = 12.3
                std = 15.2
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 24.7
                std = 2.1
            elif class_ea == 2:
                mean = 24.4
                std = 2.8
            elif class_ea == 3:
                mean = 24.4
                std = 2.7
            elif class_ea == 4:
                mean = 24.2
                std = 2.7
            elif class_ea == 5:
                mean = 23.9
                std = 3.1
            elif class_ea == 6:
                mean = 23.3
                std = 3.9
            elif class_ea == 7:
                mean = 22.6
                std = 4.8
            elif class_ea == 8:
                mean = 21.2
                std = 6.8
            elif class_ea == 9:
                mean = 17.6
                std = 12.7
    elif scenario == 0 and type_path == 0:  # Dense Urban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 23.8
                std = 4.4
            elif class_ea == 2:
                mean = 21.9
                std = 6.3
            elif class_ea == 3:
                mean = 19.7
                std = 8.1
            elif class_ea == 4:
                mean = 18.1
                std = 9.3
            elif class_ea == 5:
                mean = 16.3
                std = 11.5
            elif class_ea == 6:
                mean = 14.0
                std = 13.3
            elif class_ea == 7:
                mean = 12.1
                std = 14.9
            elif class_ea == 8:
                mean = 8.7
                std = 17.0
            elif class_ea == 9:
                mean = 6.4
                std = 12.3
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 23.7
                std = 4.5
            elif class_ea == 2:
                mean = 21.8
                std = 6.3
            elif class_ea == 3:
                mean = 19.6
                std = 8.2
            elif class_ea == 4:
                mean = 18.0
                std = 9.4
            elif class_ea == 5:
                mean = 16.3
                std = 11.5
            elif class_ea == 6:
                mean = 15.9
                std = 12.4
            elif class_ea == 7:
                mean = 12.3
                std = 15.0
            elif class_ea in [8, 9]:
                mean = 10.5
                std = 15.7
    elif scenario == 1 and type_path == 1:  # Urban Scenario (LOS)
        mean = 8.0
        std = 4.0
    elif scenario == 1 and type_path == 0:  # Urban Scenario (NLOS)
        mean = 7.0
        std = 3.0
    elif scenario == 2 and type_path == 1:  # Suburban Scenario (LOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 21.3
                std = 7.6
            elif class_ea == 2:
                mean = 21.0
                std = 8.9
            elif class_ea == 3:
                mean = 21.2
                std = 8.5
            elif class_ea == 4:
                mean = 21.1
                std = 8.4
            elif class_ea == 5:
                mean = 20.7
                std = 9.2
            elif class_ea == 6:
                mean = 20.6
                std = 9.8
            elif class_ea == 7:
                mean = 20.3
                std = 10.8
            elif class_ea == 8:
                mean = 19.8
                std = 12.2
            elif class_ea == 9:
                mean = 19.1
                std = 13.0
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 23.2
                std = 5.0
            elif class_ea == 2:
                mean = 23.6
                std = 4.5
            elif class_ea == 3:
                mean = 23.5
                std = 4.7
            elif class_ea == 4:
                mean = 23.4
                std = 5.2
            elif class_ea == 5:
                mean = 23.2
                std = 5.7
            elif class_ea == 6:
                mean = 23.3
                std = 5.9
            elif class_ea == 7:
                mean = 23.4
                std = 6.2
            elif class_ea == 8:
                mean = 23.2
                std = 7.0
            elif class_ea == 9:
                mean = 23.1
                std = 7.6
    elif scenario == 2 and type_path == 0:  # Suburban Scenario (NLOS)
        if type_freq == 0:  # S-band
            if class_ea in [0, 1]:
                mean = 20.6
                std = 8.5
            elif class_ea == 2:
                mean = 16.7
                std = 12.0
            elif class_ea == 3:
                mean = 13.2
                std = 12.8
            elif class_ea == 4:
                mean = 11.3
                std = 13.8
            elif class_ea == 5:
                mean = 9.6
                std = 12.5
            elif class_ea == 6:
                mean = 7.5
                std = 11.2
            elif class_ea == 7:
                mean = 9.1
                std = 10.1
            elif class_ea in [8, 9]:
                mean = 11.7
                std = 13.1
        elif type_freq == 1:  # Ka-band
            if class_ea in [0, 1]:
                mean = 22.5
                std = 5.0
            elif class_ea == 2:
                mean = 19.4
                std = 8.5
            elif class_ea == 3:
                mean = 15.5
                std = 10.0
            elif class_ea == 4:
                mean = 13.9
                std = 10.6
            elif class_ea == 5:
                mean = 11.7
                std = 10.0
            elif class_ea == 6:
                mean = 9.8
                std = 9.1
            elif class_ea in [7, 8, 9]:
                mean = 15.6
                std = 9.1
    elif scenario == 3 and type_path == 1:  # Rural Scenario (LOS)
        mean = 12.0
        std = 4.0
    elif scenario == 3 and type_path == 0:  # Rural Scenario (NLOS)
        mean = 7.0
        std = 3.0

    table_x = np.zeros(2)
    table_x[0] = mean
    table_x[1] = std

    return table_x


# 路径类别判断（TS 38.811）
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

    # 生成一个随机数并根据 p_los 确定 type_path
    temp = np.random.rand()
    if temp < p_los:
        type_path = 1
    else:
        type_path = 0

    return type_path
