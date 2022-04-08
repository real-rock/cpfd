import time

import numpy as np
import pandas as pd
import datetime


def read_lines(src):
    f = open(src)
    lines = f.readlines()
    f.close()
    return lines


def processing(lines):
    size = len(lines)
    dts = []
    d = np.zeros((size, 3))
    for idx, line in enumerate(lines):
        line = line.replace('\n', '')
        ele = line.split(', ')
        err_idx = ele[0].rfind('\x00')
        if err_idx != -1:
            ele[0] = ele[0][err_idx + 1:]
        dt = datetime.datetime.strptime(ele[0], '%Y-%m-%d %H:%M:%S')
        dts.append(dt)
        ele = ele[1:]
        for j, e in enumerate(ele):
            e.replace(' ', '')
            d[idx][j] = float(e)
    return dts, d


if __name__ == '__main__':
    endpoints = ['134', '107', '120', '121', '124', '181', '196', '199']
    src_dir = '../datasets/indoor_particles/'
    output_dir = '../datasets/indoor_particles/csv/'
    cols = ['DATE', 'PM1', 'PM2.5', 'PM10']

    for endpoint in endpoints:
        print(f'[INFO] Endpoint: {endpoint} process start')
        src = src_dir + 'particle' + endpoint + '.txt'
        dst = output_dir + 'particle' + endpoint + '.csv'
        print(f'[INFO] source file: {src}, output file: {dst}')
        try:
            ls = read_lines(src)
            dt, data = processing(ls)
        except Exception as e:
            print(f'[ERROR] error file: {src}')
            print(f'error message: {e}')
            continue
        print(f'[INFO] Data processing finished, no error found')
        raw = {cols[0]: dt, cols[1]: data.transpose()[0], cols[2]: data.transpose()[1], cols[3]: data.transpose()[2]}
        df = pd.DataFrame(raw)
        df.to_csv(dst, index=False)
        print(f'[INFO] {dst} successfully saved')
        print()
