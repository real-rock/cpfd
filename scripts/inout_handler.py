import pandas as pd
import datetime


class InoutHandler:
    def __init__(self, src='~/Desktop/workspace/cpfd/datasets/inout/inout.csv',
                 dst_dir='~/Desktop/workspace/cpfd/datasets/inout/'):
        self.__src_file = src
        self.__output_dir = dst_dir
        self.__df = None
        self.read_df()
        self.person_df, self.window_df, self.ap_df = self.get_dfs()

    def read_df(self, start_date='2022-03-11 09:00:00', end_date='2022-04-15 15:15:00'):
        self.__df = pd.read_csv(self.__src_file)
        self.__trim_columns()
        self.__trim_dates()
        self.__df = self.__df.dropna()
        self.__df = self.__df[self.__df['DATE'] > start_date]
        self.__df = self.__df[self.__df['DATE'] < end_date]

    def __trim_columns(self):
        self.__df = self.__df[['time', 'person', 'acitivity']]
        self.__df.columns = ['DATE', 'PERSON', 'ACTIVITY']

    def __trim_dates(self):
        dates = []
        for date in self.__df['DATE'].values:
            try:
                date = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M')
                dates.append(date)
            except Exception as e:
                dates.append(None)
                print(date, e)
        self.__df['DATE'] = dates

    def get_dfs(self):
        window = self.__df[self.__df['PERSON'] == 'Window']
        window = window.drop(columns=['PERSON'])
        ap = self.__df[self.__df['PERSON'] == 'Air Purifier']
        ap = ap.drop(columns=['PERSON'])
        person = self.__df[self.__df['PERSON'].isin(
            ['LEE', 'SON', 'PARK2', 'GUEST1', 'GUEST2', 'GOO', 'PARK', 'HEO', 'KIM', 'GUEST3']
        )]
        person = self.__count_person(person)
        person = person.drop(columns=['PERSON', 'ACTIVITY'])
        return person, window, ap

    @staticmethod
    def __count_person(df):
        init_number = 0
        person_count = []

        for row in df.values:
            if row[2] == 'In':
                init_number += 1
            elif row[2] == 'Out':
                init_number -= 1

            person_count.append(init_number)
        df['PERSON_NUMBER'] = person_count
        return df

    def save(self, to=None, name=None):
        if name is None:
            name = ['person', 'window', 'air_purifier']
        if to is None:
            to = self.__output_dir
        self.person_df.to_csv(to + name[0] + '.csv', index=False)
        self.window_df.to_csv(to + name[1] + '.csv', index=False)
        self.ap_df.to_csv(to + name[2] + '.csv', index=False)


if __name__ == '__main__':
    handler = InoutHandler()
    handler.save()
