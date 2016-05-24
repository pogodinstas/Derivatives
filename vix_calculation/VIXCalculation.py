import pandas as pd
import numpy as np
import Quandl
import datetime as dt
import re


class SNPValue:
    def __init__(self, calculation_date):
        self.calculation_date = calculation_date
        self.snp = float(
                Quandl.get('YAHOO/INDEX_GSPC.6', rows=10, authtoken='H-hALWWAUR5M3AtEpU-s').loc[self.calculation_date])

    def get_snp(self):
        return self.snp


class OptionSeries:
    def __init__(self, calls, puts):

        self.calls = pd.read_csv(calls, sep=';', index_col='Strike')[['Bid', 'Ask']]
        self.puts = pd.read_csv(puts, sep=';', index_col='Strike')[['Bid', 'Ask']]

        for item in [self.calls, self.puts]:

            new_index = []
            for _index in item.index.values:
                new_index.append(int(re.findall('(\d+)\-', _index)[0]))
            item.index = new_index
            item.index.name = 'Strike'

    def get_calls(self):
        return self.calls

    def get_puts(self):
        return self.puts


class VIXParameters(OptionSeries, SNPValue):
    """
    provide date in YYYY-MM-DD HH:MM format
    """

    def __init__(self, calls, puts, expiration_time, calculation_date):
        OptionSeries.__init__(self, calls, puts)
        SNPValue.__init__(self, calculation_date)
        self.expiration_time = dt.datetime.strptime(expiration_time, '%Y-%m-%d %H:%M')  # parsing dates
        self.ATM_strike = min(self.calls.index, key=lambda x: abs(x - self.snp))  # ATM strike

    def T_parameter(self, current_time=dt.datetime.now()):

        # current_time = dt.datetime(2016, 05, 13, 23, 14) # dt.datetime.now()

        M_current_day = 24 * 60 - (current_time.hour * 60 + current_time.minute)  # minutes in current day
        M_Settlement = self.expiration_time.hour * 60 + self.expiration_time.minute  # minutes in settlement days

        M_other_days = int((self.expiration_time - current_time).total_seconds() / 60)
        Minutes_in_year = dt.timedelta(365).total_seconds() / 60  # total minutes in year
        T = (M_current_day + M_Settlement + M_other_days) / Minutes_in_year

        return T

    def R_parameter(self, T):

        C = self.calls.loc[self.ATM_strike].mean()  # call price
        P = self.puts.loc[self.ATM_strike].mean()  # put price

        R = (-1 / T) * np.log2(self.ATM_strike / (self.snp + P - C))  # calculates T

        return R

    def F_parameter(self, R, T):

        C_P_diff = abs(self.calls.mean(axis=1) - self.puts.mean(axis=1))

        strike = C_P_diff.idxmin()  # index of minimal difference
        abs_dif = C_P_diff.min()  # minimal difference

        F = strike + np.exp(R * T) * abs_dif

        return F

    def K0_parameter(self, F):

        K0 = max(self.calls.loc[self.calls.index < F].index)
        return K0

    def select_options(self, K0):

        avg_bid = (self.calls['Bid'].loc[K0] + self.puts['Bid'].loc[K0]) / 2  # average bid-ask for K0 strike
        avg_ask = (self.calls['Ask'].loc[K0] + self.puts['Ask'].loc[K0]) / 2

        self.calls = self.calls.loc[K0 + 1:]  # selecting OTM
        self.puts = self.puts.loc[:K0 - 1]

        # selecting calls
        for i in range(len(self.calls.index)):
            if self.calls['Bid'].iloc[i] == 0 and self.calls['Bid'].iloc[i + 1] == 0:  # checkin two zero-bid condition
                to_drop = range(i, len(self.calls.index))
                self.calls = self.calls.drop(self.calls.index[to_drop])
                break
        self.calls = self.calls[self.calls['Bid'] > 0]  # drop remaining zero-bids

        # selecting puts
        for i in range(len(self.puts.index), 0):
            print i
            if self.puts['Bid'].iloc[i] == 0 and self.puts['Bid'].iloc[i + 1] == 0:
                to_drop = range(i, 0)
                self.puts = self.puts.drop(self.puts.index[to_drop])
                break
        self.puts = self.puts[self.puts['Bid'] > 0]

        self.calls['type'] = 'call'
        self.puts['type'] = 'put'

        df_atm = pd.DataFrame(data={'Bid': [avg_bid], 'Ask': [avg_ask], 'type': ['call put average'], 'Strike': [K0]})
        df_atm = df_atm.set_index('Strike')

        df_final_series = pd.concat([self.puts, df_atm, self.calls])  # concatenating puts and calls

        return df_final_series

    def Q_parameter(self, final_series):

        final_series['Q'] = final_series.mean(1)

        return final_series

    def dK_parameter(self, final_series):

        final_series['dK'] = 0
        for i in range(len(final_series.index)):

            if i == 0:
                dk = final_series.index[i + 1] - final_series.index[i]

            elif i == len(final_series.index) - 1:
                dk = final_series.index[i] - final_series.index[i - 1]

            else:
                dk = (final_series.index[i + 1] - final_series.index[i - 1]) / float(2)

            final_series['dK'].iloc[i] += dk

        return final_series


def calculate_volatility(T, R, F, K0, final_series):
    cum_sum = 0
    for i in final_series.index:
        cum_sum += (final_series['dK'].loc[i] / (i ** 2)) * np.exp(R * T) * final_series['Q'].loc[i]

    variance = (2 / T) * cum_sum - (1 / T) * (F / K0 - 1) ** 2

    return variance


def calculate_vix(variance1, variance2, T1, T2):
    N365 = dt.timedelta(365).total_seconds() / 60.0
    N30 = dt.timedelta(30).total_seconds() / 60.0
    NT1 = T1 * N365
    NT2 = T2 * N365

    VIX = 100 * np.sqrt(
            (T1 * variance1 * ((NT2 - N30) / (NT2 - NT1)) + T2 * variance2 * ((N30 - NT1) / (NT2 - NT1))) * (
            N365 / N30))

    return round(VIX, 2)

if __name__ == '__main__':

    near_options = VIXParameters('1006_calls.csv', '1006_puts.csv', '2016-06-10 08:30', '2016-05-13')
    next_options = VIXParameters('1706_calls.csv', '1706_puts.csv', '2016-06-17 15:00', '2016-05-13')

    print 'Current S&P 500 index value:', near_options.get_snp()
    print

    print 'Importing results, near-term calls example:'
    print near_options.get_calls()
    print

    T1 = near_options.T_parameter(dt.datetime(2016, 05, 13, 15, 14))  # we calculating VIX on 13.05.2016 15:14 Chicago time
    T2 = next_options.T_parameter(dt.datetime(2016, 05, 13, 15, 14))
    print 'T parameter calculations results:'
    print 'T for near-term series:', T1
    print 'T for next-term series:', T2
    print

    R1 = near_options.R_parameter(T1)
    R2 = next_options.R_parameter(T2)
    R = (R1 + R2) / 2
    print 'R parameter calculation results:'
    print 'R for near-term series:', R1
    print 'R for next-term series:', R2
    print 'R as average of near- and next-term:', R
    print

    F1 = near_options.F_parameter(R, T1)
    F2 = next_options.F_parameter(R, T2)
    print 'F parameter calculation results:'
    print 'F for near-term series:', F1
    print 'F for next-term series:', F2
    print

    K01 = near_options.K0_parameter(F1)
    K02 = next_options.K0_parameter(F2)
    print 'K0 parameter calculation results:'
    print 'K0 for near-term series:', K01
    print 'K0 for next-term series:', K02
    print

    near_series = near_options.select_options(K01)
    next_series = next_options.select_options(K02)

    near_series = near_options.Q_parameter(near_series)
    next_series = next_options.Q_parameter(next_series)

    near_series = near_options.dK_parameter(near_series)
    next_series = next_options.dK_parameter(next_series)

    print 'Q and dK calculation resuts'
    print 'Near-term series:'
    print near_series
    print 'Next-term series:'
    print next_series
    print

    variance1 = calculate_volatility(T1, R, F1, K01, near_series)
    variance2 = calculate_volatility(T2, R, F2, K02, next_series)
    print 'Variance calculation resuts'
    print 'Variance for near-term series:', variance1
    print 'Variance for next-term series:', variance2
    print

    vix = calculate_vix(variance1, variance2, T1, T2)
    print '==========='
    print 'VIX =', vix
    print '==========='
