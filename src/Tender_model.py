import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime as dt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm.notebook import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
import catboost as cb
from sklearn.metrics import mean_absolute_error

from string import punctuation, digits, ascii_letters
punctuation += '—' + '«' + '»' + digits + ascii_letters
punctuation = punctuation.replace('-', '')
from pathlib import Path
import nltk
nltk.download('punkt')
nltk.download('stopwords')

way= Path(__file__).parent


class Model():

    def __init__(self):
        self.data = pd.DataFrame()

    def build_text(self, txt):
        sp_symbols = ['\n\n', '!', '?', '…']
        sp_symbols1 = ['\n', '\t', '\r', '\xa0', '  ']
        for symbol in sp_symbols:
            txt = txt.replace(symbol, ' ', -1)
        for symbol in sp_symbols1:
            txt = txt.replace(symbol, ' ', -1)

        return txt

    def text_to_sent(self, incorrect_string, skip=True):
        # Принимает на вход некорректную строку (с лишними символами), возвращает корректную

        lower_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz-0123456789'
        symbols = ['--', '- ', ' -', ' - ']
        for symbol in symbols:
            incorrect_string = incorrect_string.replace(symbol, '—', -1)

        incorrect_string = incorrect_string.strip()
        if (incorrect_string and incorrect_string[0] == '-'):
            incorrect_string = incorrect_string.replace('-', '—', 1)
        incorrect_string = incorrect_string.strip()
        incorrect_string = incorrect_string.lower()

        dataset = []
        for i in incorrect_string:
            if i in lower_letters:
                dataset.append(i)
            else:
                dataset.append(' ')

        dataset = ''.join(dataset)
        tokens = word_tokenize(dataset)
        tokens1 = []

        if (skip):
            for token in tokens:
                count1 = 0
                count2 = 0
                if (len(token) <= 2 and token in vocabular_short):
                    tokens1.append(token)
                elif (len(token) > 4):
                    tokens1.append(token)
                elif (len(token) == 3 or len(token) == 4):
                    for letter in token:
                        if letter in letters:
                            count1 += 1
                        else:
                            count2 += 1
                    if (count1 > 0 and count2 > 0):
                        tokens1.append(token)

        else:
            tokens = [i for i in tokens if len(i) > 0]
            string = ' '.join(tokens)
            return string

        tokens1 = [i for i in tokens if len(i) > 0]
        string = ' '.join(tokens1)
        return string

    def process_code_categories(self):
        data = self.data.copy()

        def check_kpgs(arr):
            if (isinstance(arr, list)):
                arr1 = []
                arr2 = []
                for i in arr:
                    arr1.append(i[:2])
                    arr2.append(i[:5])
                arr3 = arr + arr1 + arr2
                arr3 = np.array(arr3)
                return np.unique(arr3)
            else:
                return arr

        data['КПГЗ new'] = data['КПГЗ'].str.split(';')
        data['ОКПД new'] = data['ОКПД 2'].str.split(';')
        data['ОКПД_new'] = data['ОКПД new'].apply(lambda x: check_kpgs(x))
        data['КПГЗ_new'] = data['КПГЗ new'].apply(lambda x: check_kpgs(x))

        data = data.drop(columns=['ОКПД 2', 'КПГЗ'])
        data = data.rename(columns={'ОКПД_new': 'OKPD', 'КПГЗ_new': 'KPGZ'})

        kpgz = pd.read_csv(Path(way,'files','kpgz_codes.csv'), index_col='code')

        okpd = pd.read_csv(Path(way,'files','okpd_codes.csv'), index_col='code')

        def kpgz_full(code):
            description = ''
            while (description == ''):
                try:
                    description = kpgz.loc[code].values[0]
                except:
                    if code.count('.') != 0:
                        code = code[:code.rindex('.')]
                    else:
                        return ''

            return description

        def okpd_full(code):
            description = ''
            while (description == ''):
                try:
                    description = okpd.loc[code].values[0]
                except:
                    if code.count('.') != 0:
                        code = code[:code.rindex('.')]
                    else:
                        return ''

            return description

        data["text_description"] = ''
        data["text_description"] = data["text_description"].astype(str)

        def process_kpgz(arr):
            ans = []
            if not isinstance(arr, float):
                for code in arr:
                    ans.append(kpgz_full(code))
                return ans
            return []

        def process_okpd(arr):
            ans = []
            if not isinstance(arr, float):
                for code in arr:
                    ans.append(okpd_full(code))
                return ans
            return []

        def string_to_list(x):
            if not isinstance(x, float):
                x = x.replace("'", '').replace(']', '').replace('[', '').split(
                    ' ')
                return x
            else:
                return []

        arr = data['KPGZ'].apply(lambda x: process_kpgz(x))
        arr2 = data['OKPD'].apply(lambda x: process_okpd(x))

        data['OKPD'] = arr2
        data['KPGZ'] = arr
        data['text_description'] = ''

        def replace_list_nan(x):
            try:
                return ' '.join(x)
            except:
                return ''

        data['KPGZ'] = data['KPGZ'].apply(lambda x: replace_list_nan(x))
        data['OKPD'] = data['OKPD'].apply(lambda x: replace_list_nan(x))

        data['text_description'] = data['OKPD'] + data['KPGZ']

        data['text_description'] = data['text_description'].apply(
            lambda x: self.build_text(x))

        data['text_description'] = data['text_description'].apply(
            lambda x: self.text_to_sent(x, skip=False))
        data['text_description'] = data['text_description'].apply(
            lambda x: x.lower())

        text_data = data[['id', 'text_description']]
        return text_data

    #     def train_model(self, ref):
    #         self.data = pd.read_excel(ref)
    #         sent_dataset = [self.build_text(self.data['Наименование КС'].values[i]) for i in tqdm(range(self.data.shape[0]))]
    #         name = [self.text_to_sent(sent_dataset[i], skip=False) for i in tqdm(range(self.data.shape[0]))]
    #         self.data['Наименование КС'] = name
    #         self.data['Регион'] = self.data['Регион'].str.lower()

    # #       ------------------------------------------------------------------------- Даты
    #         self.data['Дата'] = pd.to_datetime(self.data['Дата'])
    #         self.data['Год'] = self.data['Дата'].dt.year
    #         self.data['Месяц'] = self.data['Дата'].dt.month
    #         self.data['День'] = self.data['Дата'].dt.day
    #         self.data['День недели'] = self.data['Дата'].dt.day_of_week
    #         self.data['День года'] = self.data['Дата'].dt.day_of_year
    #         self.data['Неделя'] = self.data['Дата'].dt.week
    #         self.data['Число'] = self.data['Дата'].map(lambda x: x.strftime('%Y-%m-%d'))
    #         self.data['Число'] = pd.to_datetime(self.data['Число'])

    # #       ------------------------------------------------------------------------- Числовые
    #         count_inn = self.data.groupby('ИНН').count()
    #         count_inn = count_inn.rename(columns={'Итоговая цена': 'Кол-во ИНН'})['Кол-во ИНН']
    #         sum_inn = self.data.groupby('ИНН').sum()
    #         sum_inn = sum_inn.rename(columns={'Итоговая цена': 'Сумма ИНН'})['Сумма ИНН']
    #         self.data = pd.merge(self.data, count_inn, on='ИНН')
    #         self.data = pd.merge(self.data, sum_inn, on='ИНН')
    #         self.data['Процент'] = (self.data['НМЦК'] - self.data['Итоговая цена']) / self.data['НМЦК']
    #         self.data['Кол-во ОКПД'] = self.data['ОКПД 2'].str.count(';') + 1
    #         self.data['Кол-во КПГЗ'] = self.data['КПГЗ'].str.count(';') + 1

    # #       ------------------------------------------------------------------------- Доп данные
    #         self.data['Кол-во ОКПД'] = self.data['Кол-во ОКПД'].fillna(0)
    #         self.data['Кол-во КПГЗ'] = self.data['Кол-во КПГЗ'].fillna(0)
    #         weather = pd.read_csv('weather.csv')
    #         weather['время'] = pd.to_datetime(weather['время'])
    #         data1 = pd.merge(self.data, weather, left_on='Число', right_on='время')
    #         cat = pd.read_csv('cat.csv')
    #         data2 = pd.merge(data1, cat[['id', 'text_description']], on='id')
    #         train = data2.drop(columns=['Число', 'id', 'ОКПД 2', 'КПГЗ', 'время', 'Итоговая цена', 'Дата', 'Ставки'])
    # #         train1 = train[train['Кол-во ОКПД'] < 50]
    # #         train1 = train1[train1['Кол-во КПГЗ'] < 40]
    # #         train1 = train1[train1['Участники'] < 20]
    # #         train1 = train1[(train1['Процент'] < 0.8) | (train1['Процент'] == 1)]
    # #         train1 = train1[train1['НМЦК'] < 2000000]
    #         train2 = train.sample(frac=1, random_state=0)
    #         X = train2.drop(columns=['Процент', 'Участники'])
    #         y = train2[['Процент', 'Участники']]

    # #       ------------------------------------------------------------------------- Модель
    #         model = cb.CatBoostRegressor(loss_function='MAE',
    #                                      iterations=300,
    #                                      learning_rate=0.3)

    #         model.fit(X, y['Участники'],
    #                   cat_features=['Наименование КС', 'Регион', 'ИНН', 'text_description',
    #                                'Год', 'Месяц', 'День', 'День недели', 'Неделя', 'День года'],
    #                   verbose=True)
    #         model.save_model('catboost_members', format="cbm")

    def predict_file(self, ref):

       
        columns = pd.read_excel(Path(way,'files','columns.xlsx')).columns
        self.data = pd.read_excel(ref)
        sent_dataset = [self.build_text(self.data['Наименование КС'].values[i])
                        for i in range(self.data.shape[0])]
        name = [self.text_to_sent(sent_dataset[i], skip=False) for i in
                range(self.data.shape[0])]
        self.data['Наименование КС'] = name
        self.data['Регион'] = self.data['Регион'].str.lower()
        self.data['Дата'] = pd.to_datetime(self.data['Дата'])
        self.data['Год'] = self.data['Дата'].dt.year
        self.data['Месяц'] = self.data['Дата'].dt.month
        self.data['День'] = self.data['Дата'].dt.day
        self.data['День недели'] = self.data['Дата'].dt.day_of_week
        self.data['День года'] = self.data['Дата'].dt.day_of_year
        self.data['Неделя'] = self.data['Дата'].dt.week
        self.data['Число'] = self.data['Дата'].map(
            lambda x: x.strftime('%Y-%m-%d'))
        self.data['Число'] = pd.to_datetime(self.data['Число'])
        count_inn = self.data.groupby('ИНН').count()
        count_inn = count_inn.rename(columns={'День': 'Кол-во ИНН'})[
            'Кол-во ИНН']
        sum_inn = self.data.groupby('ИНН').sum()
        sum_inn = sum_inn.rename(columns={'День': 'Сумма ИНН'})['Сумма ИНН']
        self.data = pd.merge(self.data, count_inn, on='ИНН')
        self.data = pd.merge(self.data, sum_inn, on='ИНН')
        self.data['Кол-во ОКПД'] = self.data['ОКПД 2'].str.count(';') + 1
        self.data['Кол-во КПГЗ'] = self.data['КПГЗ'].str.count(';') + 1
        self.data['Кол-во ОКПД'] = self.data['Кол-во ОКПД'].fillna(0)
        self.data['Кол-во КПГЗ'] = self.data['Кол-во КПГЗ'].fillna(0)
        weather = pd.read_csv(Path(way,'files','weather1.csv'))
        weather['время'] = pd.to_datetime(weather['время'])
        data1 = pd.merge(self.data, weather, left_on='Число', right_on='время')
        cat = self.process_code_categories()
        data2 = pd.merge(data1, cat[['id', 'text_description']], on='id')
        test = data2.drop(
            columns=['Число', 'id', 'ОКПД 2', 'КПГЗ', 'время', 'Дата'])
        test = test[columns]

        catboost_members = cb.CatBoostRegressor().load_model(
            Path(way,'files','catboost_members'))
        pred = catboost_members.predict(test)

        catboost_percents = cb.CatBoostRegressor().load_model(
            Path(way,'files','catboost_percents'))
        pred1 = catboost_percents.predict(test)

        answer = pd.DataFrame({'id': self.data.id,
                               'Участники': pred,
                               'Уровень снижения': pred1})
        answer.to_csv(Path(way,'files','Mister MISISter_2191574_TenderHack_Moscow.csv'),
                      index=False)

        return True

    def predict_object(self, voc):
        way2= Path(way,'files','columns.xlsx')
        
        columns = pd.read_excel(Path(way,'files','columns.xlsx')).columns
        self.data = pd.DataFrame(voc)
        sent_dataset = [self.build_text(self.data['Наименование КС'].values[i])
                        for i in range(self.data.shape[0])]
        name = [self.text_to_sent(sent_dataset[i], skip=False) for i in
                range(self.data.shape[0])]
        self.data['Наименование КС'] = name
        self.data['Регион'] = self.data['Регион'].str.lower()
        self.data['Дата'] = pd.to_datetime(self.data['Дата'])
        self.data['Год'] = self.data['Дата'].dt.year
        self.data['Месяц'] = self.data['Дата'].dt.month
        self.data['День'] = self.data['Дата'].dt.day
        self.data['День недели'] = self.data['Дата'].dt.day_of_week
        self.data['День года'] = self.data['Дата'].dt.day_of_year
        self.data['Неделя'] = self.data['Дата'].dt.week
        self.data['Число'] = self.data['Дата'].map(
            lambda x: x.strftime('%Y-%m-%d'))
        self.data['Число'] = pd.to_datetime(self.data['Число'])
        count_inn = self.data.groupby('ИНН').count()
        count_inn = count_inn.rename(columns={'День': 'Кол-во ИНН'})[
            'Кол-во ИНН']
        sum_inn = self.data.groupby('ИНН').sum()
        sum_inn = sum_inn.rename(columns={'День': 'Сумма ИНН'})['Сумма ИНН']
        self.data = pd.merge(self.data, count_inn, on='ИНН')
        self.data = pd.merge(self.data, sum_inn, on='ИНН')
        self.data['Кол-во ОКПД'] = self.data['ОКПД 2'].str.count(';') + 1
        self.data['Кол-во КПГЗ'] = self.data['КПГЗ'].str.count(';') + 1
        self.data['Кол-во ОКПД'] = self.data['Кол-во ОКПД'].fillna(0)
        self.data['Кол-во КПГЗ'] = self.data['Кол-во КПГЗ'].fillna(0)
        weather = pd.read_csv(Path(way,'files','weather1.csv'))
        weather['время'] = pd.to_datetime(weather['время'])
        data1 = pd.merge(self.data, weather, left_on='Число', right_on='время')
        cat = self.process_code_categories()
        data2 = pd.merge(data1, cat[['id', 'text_description']], on='id')
        test = data2.drop(
            columns=['Число', 'id', 'ОКПД 2', 'КПГЗ', 'время', 'Дата'])
        test = test[columns]

        catboost_members = cb.CatBoostRegressor().load_model(
             Path(way,'files','catboost_members'))
        pred = catboost_members.predict(test)

        catboost_percents = cb.CatBoostRegressor().load_model(
             Path(way,'files','catboost_percents'))
        pred1 = catboost_percents.predict(test)

        return (pred1[0], pred[0])

# model = Model()
# pred = model.predict_file("short_train.xlsx")
# pred = model.predict_object({"id": [287205],
#                     'Наименование КС':['стулья'],
#                     "ОКПД 2":["01.02"],
#                     "КПГЗ": ["NaN"],
#                     "Регион": ["Москва"],
#                     'НМЦК' : [50000],
#                     "Дата": ["12.12.2021"],
#                     "ИНН": ["384ра0к8а2р48п038ы"],
#                     })