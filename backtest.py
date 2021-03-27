import io
import requests
import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt

# url to fetch the data from csv file
url = 'https://raw.githubusercontent.com/guzman-energy/assignment-sde/main/data/python_api_data.csv'

# make request and get content
r = requests.get(url).content

# convert records into dataframe by reading data from csv
main_df = pd.read_csv(io.StringIO(r.decode('utf-8')), error_bad_lines=False, index_col=0)

df = main_df


# Backtest class inheriting from DataFrame class
class Backtest(DataFrame):

    @property
    def report(self):
        # keeping track of he hours
        df['he_count'] = 1

        # keeping track of traded deals
        clear_filter = main_df['clear'] == True
        df['clear_count'] = np.where(1, clear_filter, 0)
        deal_df = df[df['clear'] == True]

        cleared_df = deal_df.groupby(['target_date', 'he'], as_index=False)['bid_mw', 'he_count', 'pnl'].apply(sum)

        temp_df = cleared_df
        fil1 = temp_df['pnl'] > 0
        temp_df = temp_df[fil1]
        temp_df = temp_df[['target_date', 'he', 'bid_mw']]
        temp_df.rename(columns={"bid_mw": "mwh_with_positive_profit"}, inplace=True)

        total_volume_cleared = cleared_df['bid_mw'].sum()
        cleared_df['total_volume_cleared'] = total_volume_cleared
        cleared_df.rename(columns={"he_count": "distinct_hours_cleared", "bid_mw": "daily_volume_cleared"},
                          inplace=True)

        cleared_df = cleared_df.merge(temp_df, on=['target_date', 'he'], how='left')
        cleared_df = cleared_df[
            ['target_date', 'he', 'daily_volume_cleared', 'distinct_hours_cleared', 'total_volume_cleared',
             'mwh_with_positive_profit']]

        mixed_df = df.groupby(['target_date', 'he'], as_index=False)['bid_mw', 'he_count', 'pnl'].apply(sum)
        total_volume_bid = mixed_df['bid_mw'].sum()
        mixed_df['total_volume_bid'] = total_volume_bid
        mixed_df.rename(columns={"he_count": "distinct_hours_bid", "bid_mw": "daily_volume_bid"}, inplace=True)

        one_day_hours = 24
        distinct_days_in_back_test = len(pd.unique(mixed_df['target_date']))
        total_hours_backtest = one_day_hours * distinct_days_in_back_test
        mixed_df['total_hours_backtest'] = total_hours_backtest
        mixed_df['distinct_days_in_back_test'] = distinct_days_in_back_test
        mixed_df['one_day_hours'] = one_day_hours

        total_profit = mixed_df['pnl'].sum()
        mixed_df['total_profit'] = total_profit

        mixed_df = mixed_df.merge(cleared_df, on=['target_date', 'he'], how='left')
        mixed_df = mixed_df.fillna(0)

        # start date
        minpnl = mixed_df['pnl'].min()
        minrow = mixed_df['pnl'].argmin()
        mindate = mixed_df.iloc[minrow]['target_date']
        mixed_df['start_date'] = mindate

        # end date
        maxpnl = mixed_df['pnl'].max()
        maxrow = mixed_df['pnl'].argmax()
        maxdate = mixed_df.iloc[maxrow]['target_date']
        mixed_df['end_date'] = maxdate

        # % of possible volume bid
        mixed_df['%_of_possible_volume_bid'] = mixed_df['daily_volume_bid'] / mixed_df['one_day_hours']

        # % of hours bid
        mixed_df['%_of_hours_bid'] = mixed_df['distinct_hours_bid'] / mixed_df['total_hours_backtest']

        # average bid size
        mixed_df['average_bid_size'] = mixed_df['daily_volume_bid'] / mixed_df['distinct_hours_bid']

        # % volume bid cleared
        mixed_df['%_of_volume_bid_cleared'] = mixed_df['total_volume_cleared'] / mixed_df['total_volume_bid']

        # % of hours bid cleared
        mixed_df['%_of_hours_bid_cleared'] = mixed_df['distinct_hours_cleared'] / mixed_df['distinct_hours_bid']

        # average clear size
        mixed_df['average_clear_size'] = mixed_df['daily_volume_cleared'] / mixed_df['distinct_hours_cleared']

        # profit per mwh
        mixed_df['profit_per_mwh'] = mixed_df['total_profit'] / mixed_df['total_volume_cleared']

        # win %
        mixed_df['%_win'] = mixed_df['mwh_with_positive_profit'] / mixed_df['daily_volume_cleared']

        # expected daily bid
        mixed_df['expected_daily_bid'] = mixed_df['daily_volume_bid'] / mixed_df['distinct_days_in_back_test']

        # expected daily profit
        mixed_df['expected_daily_profit'] = mixed_df['pnl'] / mixed_df['distinct_days_in_back_test']

        result_df = mixed_df[['target_date', 'he', 'pnl', 'start_date',
                              'end_date', '%_of_possible_volume_bid', '%_of_hours_bid',
                              'average_bid_size', '%_of_volume_bid_cleared', '%_of_hours_bid_cleared',
                              'average_clear_size', 'profit_per_mwh', '%_win', 'expected_daily_bid',
                              'expected_daily_profit']]

        return result_df


backtest = Backtest()
df1 = backtest.report


def plot(df_to_plot):
    # matplotlib
    plt.style.use('seaborn')

    pnl_price = df_to_plot['pnl']
    pnl_date = df_to_plot['target_date']

    plt.plot_date(pnl_date, pnl_price, linestyle='solid')

    plt.gcf().autofmt_xdate()

    plt.title('PnL Prices Vs Dates')
    plt.xlabel('Date')
    plt.ylabel('PnL Price')

    plt.tight_layout()

    plt.show()


plot(df1)
