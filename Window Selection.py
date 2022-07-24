import pandas as pd
import numpy as np
from scipy import signal
from TsTransformationPy import *
from lightgbm import LGBMClassifier
import math, json, timeit

# ======================================================================================================================
# DSP Module for auto definition size of window
# ======================================================================================================================

class BestWindow:
    orig_all_cols: list
    orig_feats_cols: list
    target: str
    binary: bool
    fs = 1  # sampling rate
    N = 5  # N tries get window
    max_feats = 50000

# ----------------------------------------------------------------------------------------------------------------------
    # definition parameters
    def set_params(self, df, target):
        self.target = target
        self.orig_all_cols = df.columns.to_list()
        self.orig_feats_cols = df.drop(self.target, axis=1).columns.to_list()
        if df[self.target].nunique() == 2:
            self.binary = True
        elif df[self.target].nunique() > 2:
            self.binary = False
        else:
            raise Exception('Cannot define type of task (binary or multi classification)')

# ----------------------------------------------------------------------------------------------------------------------
    # get different windows via FFT signal
    def get_windows(self, df, min_up_bound):
        all_wins = [2]
        curr_win = []

        for col in self.orig_feats_cols:
            x = df[col] - df[col].mean()
            f, Pxx = signal.periodogram(x, fs=self.fs, window='hann', scaling='spectrum')
            max_array = np.sort(Pxx)[::-1]

            for i in range(self.N):
                idx, = np.where(Pxx == max_array[i])
                if f[idx][0] == 0:
                    break
                s = 1 / f[idx]
                win = math.ceil(self.fs * s)
                curr_win.append(win)
                if (win not in all_wins) and win <= min_up_bound:
                    all_wins.append(win)
            print(f'{col}: {curr_win}')

        all_wins.sort()
        return all_wins

    def check_windows(self, df):
        up_bound_0 = self.max_feats // len(self.orig_feats_cols)
        up_bound_1 = df[self.target].value_counts().iloc[-1]

        val_target = df[self.target][0]
        count = 1
        for val in df[self.target]:
            if val != val_target:
                count += 1
                val_target = val

        up_bound_2 = len(df) // count
        min_up_bound = min(up_bound_0, up_bound_1, up_bound_2)
        return min_up_bound

# ----------------------------------------------------------------------------------------------------------------------
# Drop discard, transformation and lightGBM
    # support function for "df_without_discard()"
    @staticmethod
    def split_list_index(idx_list):
        d = {}
        d_num = 0
        start = 0
        val = idx_list[0]
        count = 0
        for j in range(len(idx_list)):
            if idx_list[j] != (val + count):
                d[d_num] = idx_list[start:j]
                d_num += 1
                start = j
                val = idx_list[j] + 1
                if (j != (len(idx_list) - 1)) and (val != idx_list[j+1]):
                    d[d_num] = [idx_list[j]]
                    d_num += 1
                    start = j + 1
                    val = idx_list[j+1]
                count = 0
            else:
                count += 1
            if j == len(idx_list) - 1:
                d[d_num] = idx_list[start:j+1]
        return d

    # Drop discard
    @staticmethod
    def df_without_discard(df, win, target='target'):
        drop_discard = []
        for i in df[target].value_counts().index:
            idx_list = df[df[target]==i].index.tolist()
            if len(idx_list) == (idx_list[-1] - idx_list[0] + 1):
                num_sample = len(idx_list) // win
                drop_discard += idx_list[num_sample * win:]
            else:
                d = BestWindow().split_list_index(idx_list)
                for k in d.keys():
                    if len(d[k]) >= win:
                        num_sample = len(d[k]) // win
                        drop_discard += d[k][num_sample * win:]
                    else:
                        drop_discard += d[k]

        df.drop(drop_discard, axis=0, inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    # size of datatype
    @staticmethod
    def get_size_dtype(df):
        if df.dtypes.mode()[0] == 'int8' or df.dtypes.mode()[0] == 'uint8':
            size_dtype = 1
            size_dtype_fe = 2
        elif df.dtypes.mode()[0] == 'int16' or df.dtypes.mode()[0] == 'uint16':
            size_dtype = 2
            size_dtype_fe = 4
        else:
            size_dtype = 4
            size_dtype_fe = 4
        return size_dtype, size_dtype_fe

    # cut, transformation, lgbm
    def cut_transform_model(self, df, wins):
        res = {'win': None, 'score': 10000}
        size_dtype, size_dtype_fe = self.get_size_dtype(df)
        print(f'     Select a window from this list: {wins}\n')
        print('     ' + '-' * 25)
        wins = [256]#SETTING WINDOW SIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for win in wins:
            df_trans = df.copy()
            # cut
            df_trans = self.df_without_discard(df_trans, win, self.target)
            # transformation
            ttp = TsTransformationPy()
            df_trans = ttp.dsp_transform_by_series(df_trans, win, 'S')
            df_trans.to_csv(r'C:\Users\akhen\Desktop\IEEE DATASETS\SEIZURE\RAW DATA NO FILTERING\256_win_RawTrain_80.csv',index=False)
            # lgbm
            objective = 'multiclass'
            logloss = 'multi_logloss'

            if self.binary:
                objective = 'binary'
                logloss = 'binary_logloss'
            x_df = df_trans.drop([self.target], axis=1)
            y_df = df_trans[self.target]
            eval_set = (df_trans.drop([self.target], axis=1), df_trans[self.target])
            lgbm = LGBMClassifier(objective=objective, min_child_samples=2, num_leaves=2, max_depth=2, random_state=42)
            lgbm.fit(x_df, y_df, eval_set=eval_set, verbose=False)
            score = lgbm.best_score_['valid_0'][logloss]
            print(f'     Window: {win}')
            print(f'     Score: {score}')
            print(f'     ~RAM: {len(self.orig_feats_cols)*(win*size_dtype + 9*size_dtype_fe)/1024}+ Kb')
            print('     ' + '-' * 25)
            if score < res['score']:
                res['win'] = win
                res['score'] = score

        best_window = res['win']
        return best_window

# ----------------------------------------------------------------------------------------------------------------------
    # get user's window settings
    @staticmethod
    def get_user_settings():
        with open('config.json', "r") as file:
            mech_vibration_set = json.load(file)
        mech_vibration_set = mech_vibration_set['tinyMlSettings']['mechVibrationPPSettings']
        win_auto_select = mech_vibration_set['windowAutoSelect']

        if str(win_auto_select).lower() == 'true':
            win_size = None
        elif str(win_auto_select).lower() == 'false':
            win_size = int(mech_vibration_set['signalFrequency'] * mech_vibration_set['signalDuration'] / 1000)
        else:
            raise Exception(f'Cannot convert {win_auto_select} to boolean')

        if str(mech_vibration_set['performDimensionReduction']).lower() == 'true':
            dim_reduction = True
        elif str(mech_vibration_set['performDimensionReduction']).lower() == 'false':
            dim_reduction = False
        else:
            raise Exception(f"Cannot convert {mech_vibration_set['performDimensionReduction']} to boolean")

        return win_size, dim_reduction

# ----------------------------------------------------------------------------------------------------------------------
    # main
    def execute(self, data, target='target'):
        self.set_params(data, target)
        best_window, dim_reduction = self.get_user_settings()

        if best_window:
            print(f"   - The size of the user's window will be used, value: {best_window}")
        else:
            print("   - Auto choice window size started\n")
            start = timeit.default_timer()
            min_up_bound = self.check_windows(data)
            wins = self.get_windows(data, min_up_bound)
            if len(wins) == 1:
                best_window = wins[0]
            elif len(wins) > 1:
                best_window = self.cut_transform_model(data, wins)
            else:
                raise Exception('Cannot define window signal size')

            stop = timeit.default_timer()
            print(f"     Auto choice window size finished, value: {best_window}, time: {np.round((stop - start), 2)} sec")
        print('    ', '-' * 55)
        return best_window, self.orig_feats_cols, dim_reduction


# example of run
if __name__ == "__main__":
    input_data = pd.read_csv(r'C:\Users\akhen\Desktop\IEEE DATASETS\SEIZURE\RAW DATA NO FILTERING\RawTrain_80.csv')

    target = 'Outcome'
    input_data.rename(columns={target: 'target'}, inplace=True)

    bw = BestWindow()
    chosen_win = bw.execute(input_data)