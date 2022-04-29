import datetime
import numpy as np
import pandas as pd
import indoor_processing


class ParticleParser:
    def __init__(self, min_date, max_date):
        self.__min_date = min_date
        self.__max_date = max_date
        self.__dates = self.__get_dates(min_date, max_date)

        self.__endpoints = ['107', '120', '121', '124', '134', '199', '190']
        self.__out = '196'
        self.__h_out = '181'
        self.__cols = ['PM1', 'PM2.5', 'PM10']

        h_out_src = f'../datasets/indoor_particles/csv/particle{self.__out}.csv'
        out_src = '../datasets/indoor_particles/csv/particle199.csv'
        h_out_df = pd.read_csv(h_out_src)
        out_df = pd.read_csv(out_src)

        self.__dfs = self.__read_indoor_csv(self.__endpoints, '../datasets/indoor_particles/csv/')
        self.__trim_dfs()

    @staticmethod
    def __get_dates(min_date, max_date):
        return pd.date_range(min_date, max_date, freq='min').strftime('%Y-%m-%d %H:%M:%S').tolist()

    @staticmethod
    def __read_indoor_csv(endpoints, src):
        dfs = []

        for endpoint in endpoints:
            src = f'{src}/particle{endpoint}.csv'
            print('[INFO] src: ' + src)
            df = pd.read_csv(src)
            dts = []
            for time in df['DATE'].values:
                time = time[:16]
                dts.append(datetime.datetime.strptime(time, '%Y-%m-%d %H:%M'))
            df['DATE'] = dts
            df = df.drop_duplicates('DATE', keep='last')
            df.index = df['DATE']
            if endpoint != '190':
                dfs.append(df.drop(columns=['DATE']))
            else:
                dfs[4] = pd.concat([dfs[4], df.drop(columns=['DATE'])], axis=0)
                dfs[4].sort_index(inplace=True)
                dfs[4].drop(dfs[4][dfs[4].index.duplicated(keep='last')].index, inplace=True)
        return dfs

    def __trim_dfs(self):
        print('[INFO] __trim_dfs() data frames shape')
        for idx, df in enumerate(self.__dfs):
            self.__dfs[idx] = df[(df.index > self.__min_date) & (df.index < self.__max_date)]
            print(f'[INFO] idx: {idx}, shape: {df.shape}')


if __name__ == '__main__':
    start = datetime.datetime.strptime('2022-03-07 14:48:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime('2022-04-15 15:15:00', '%Y-%m-%d %H:%M:%S')
    ParticleParser(start, end)
