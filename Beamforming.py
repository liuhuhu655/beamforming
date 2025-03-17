import numpy as np
from Position import generate_access_info


def calculate_sum_rate(type_measure, list_matrix_h, list_matrix_w, matrix_a, V, intf_rx, vCNR, bandwidth, power_tx):
    num_sat = len(list_matrix_h)
    num_gu = len(list_matrix_h[0])
    num_ant = list_matrix_h[0][0].shape[0]
    sum_rate = 0
    list_sinr = []
    matrix_a_sat, matrix_a_gu = generate_access_info(matrix_a)
    list_link = np.zeros(num_gu, dtype=int)

    for idx_sat in range(num_sat):
        for idx_gu in range(num_gu):
            if matrix_a[idx_sat, idx_gu] == 1:
                list_link[idx_gu] = idx_sat

    if type_measure == "CNR":
        for idx_gu in range(num_gu):
            power_signal = 0
            for idx_sat in range(num_sat):
                power_signal += matrix_a[idx_sat, idx_gu] * np.abs((list_matrix_h[idx_sat][idx_gu].conj().T @ list_matrix_h[idx_sat][idx_gu]).item()) * power_tx / max(matrix_a_sat[idx_sat], 2) / num_ant
            if power_signal == 0:
                list_sinr.append(-30)
                continue
            snr = power_signal / intf_rx
            list_sinr.append(10 * np.log10(snr))
            for idx_sat in range(num_sat):
                sum_rate += matrix_a[idx_sat, idx_gu] * (bandwidth / 1e6) / max(1, matrix_a_sat[idx_sat]) * np.log2(1 + snr)
        avg_rate = sum_rate / num_gu
        return sum_rate, avg_rate, list_sinr

    # Calculate SINR
    for idx_gu in range(num_gu):
        power_signal = 0
        power_intf_gu = 0
        power_intf_sat = 0

        for idx_sat in range(num_sat):
            signal = matrix_a[idx_sat, idx_gu] * list_matrix_h[idx_sat][idx_gu].conj().T @ list_matrix_w[idx_sat][idx_gu]
            power_signal += np.abs(signal) ** 2
        if power_signal == 0:
            list_sinr.append(-30)
            continue

        for idx_sat in range(num_sat):
            if matrix_a[idx_sat, idx_gu] == 0:
                continue
            for idx2_gu in range(num_gu):
                if idx2_gu == idx_gu:
                    continue
                intf_gu = matrix_a[idx_sat, idx2_gu] * list_matrix_h[idx_sat][idx_gu].conj().T @ list_matrix_w[idx_sat][idx2_gu]
                power_intf_gu += np.abs(intf_gu) ** 2

        for idx_sat in range(num_sat):
            if matrix_a[idx_sat, idx_gu] == 1 or V[idx_sat, idx_gu] == 0:
                continue
            for idx2_gu in range(num_gu):
                if idx2_gu == idx_gu:
                    continue
                intf_sat = matrix_a[idx_sat, idx2_gu] * list_matrix_h[idx_sat][idx_gu].conj().T @ list_matrix_w[idx_sat][idx2_gu]
                power_intf_sat += np.abs(intf_sat) ** 2

        sinr = power_signal / (power_intf_gu + power_intf_sat + intf_rx)
        list_sinr.append(10 * np.log10(sinr))

    sorted_sinr = np.sort(list_sinr)
    count_num_gu = 0
    for idx_gu in range(num_gu):
        if idx_gu <= num_gu * 0.03 or idx_gu <= 2:
            continue
        if idx_gu >= num_gu * 0.97 or idx_gu >= num_gu - 1:
            continue
        count_num_gu += 1
        sum_rate += bandwidth * np.log2(1 + sorted_sinr[count_num_gu])

    avg_rate = sum_rate / count_num_gu if count_num_gu > 0 else 0
    return sum_rate, avg_rate, list_sinr


def adjust_matrix_w(list_matrix_w, power_tx, matrix_a):
    # 初始化
    num_sat = len(list_matrix_w)
    num_gu = len(list_matrix_w[0])
    num_ant = list_matrix_w[0][0].shape[0]
    list_matrix_w = [[np.zeros((num_ant, 1)) for _ in range(num_gu)] for _ in range(num_sat)]

    for idx_sat in range(num_sat):
        sum_power = 0
        sum_gu_num = 0

        # Calculate sum_power
        for idx_gu in range(num_gu):
            if matrix_a[idx_sat, idx_gu] > 0:
                for m in range(num_ant):
                    sum_power += abs(list_matrix_w[idx_sat][idx_gu][m, 0]) ** 2
                sum_gu_num += 1

        if sum_power == 0:
            sum_power = 1  # Avoid division by zero

        # Adjust list_matrix_w based on calculated sum_power
        for idx_gu in range(num_gu):
            if matrix_a[idx_sat, idx_gu] > 0:
                list_matrix_w[idx_sat][idx_gu] = list_matrix_w[idx_sat][idx_gu] * np.sqrt(power_tx / sum_power)

    return list_matrix_w


def analog_beamforming(matrix_h, codebook, num_cand_cw):
    # 初始化
    num_codeword = len(codebook)
    num_ant = codebook[0].shape[0]
    size_tb = np.zeros(num_codeword)
    num_cand_cw = min(num_cand_cw, num_codeword)

    if np.abs(matrix_h).max() < 1e-30:
        return np.zeros((num_ant, 1))

    if num_codeword == 1:
        return codebook[0]

    for idx_codeword in range(num_codeword):
        size_tb[idx_codeword] = np.abs(matrix_h.conj().T @ codebook[idx_codeword]) ** 2

    list_codeword = np.argsort(size_tb)[::-1]
    D = np.zeros((num_ant, num_cand_cw), dtype=complex)

    for idx_cw in range(num_cand_cw):
        D[:, idx_cw] = codebook[list_codeword[idx_cw]].flatten()

    x = np.linalg.pinv(D) @ matrix_h
    matrix_w = D @ x

    c_adjust = matrix_w[0]
    matrix_w = matrix_w / c_adjust

    for n in range(num_ant):
        matrix_w[n] = matrix_w[n] / np.abs(matrix_w[n])

    matrix_w = matrix_w / np.sqrt(num_ant)

    return matrix_w


def generate_codebook(num_ant_sat_hori, num_ant_sat_vert, num_ant_sat_pol):
    num_ant = num_ant_sat_hori * num_ant_sat_vert * num_ant_sat_pol
    codebook = []

    for l in range(num_ant_sat_pol):
        for m in range(num_ant_sat_vert):
            for n in range(num_ant_sat_hori):
                idx_codeword = l * num_ant_sat_hori * num_ant_sat_vert + m * num_ant_sat_hori + n
                codeword = np.zeros((num_ant, 1), dtype=complex)

                for s_pol in range(num_ant_sat_pol):
                    phi_pol = -np.pi * s_pol * l
                    for s_v in range(num_ant_sat_vert):
                        phi_y = -np.pi * 2 * s_v * m / num_ant_sat_vert
                        if phi_y < -np.pi:
                            phi_y += 2 * np.pi
                        for s_h in range(num_ant_sat_hori):
                            s = s_pol * num_ant_sat_hori * num_ant_sat_vert + s_v * num_ant_sat_hori + s_h
                            phi_x = -np.pi * 2 * s_h * n / num_ant_sat_hori
                            if phi_x < -np.pi:
                                phi_x += 2 * np.pi
                            codeword[s, 0] = np.exp(1j * (phi_pol + phi_x + phi_y))
                codebook.append(codeword)

    return codebook
