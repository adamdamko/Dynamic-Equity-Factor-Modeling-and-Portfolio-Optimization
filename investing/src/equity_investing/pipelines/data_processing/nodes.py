# This is the data_processing nodes file
# Import libraries
# import simfin as sf
import pandas as pd
from pandas_profiling import ProfileReport


def join_data(companies: pd.DataFrame, industries: pd.DataFrame, prices: pd.DataFrame, ratios: pd.DataFrame
              ) -> pd.DataFrame:
    """
    This function creates raw data set from SimFin 'us-companies',
    'industries', 'us-shareprices-daily', and 'us-derived-shareprices-daily'  files.

    Args:
        companies: 'us-companies' data set from SimFin
        industries: 'industries' data set from SimFin
        prices: 'us-shareprices-daily' data set from SimFin
        ratios: 'us-derived_shareprices-daily-asreported' US daily series

    Returns:
        The function returns a pandas dataframe of panel data
        with companies as groups and their daily prices along
        with company metadata.
    """
    # Join company and industry
    companies['IndustryId'] = pd.to_numeric(companies['IndustryId'])
    industries['IndustryId'] = pd.to_numeric(industries['IndustryId'])
    comp_ind = pd.merge(companies, industries, on='IndustryId', how='left', validate='m:1')
    # Join company, industry, and price data
    raw_data = pd.merge(prices, comp_ind, how='left', on=['SimFinId', 'Ticker'], validate='m:1')
    # Join raw_data and ratios data
    raw_data = pd.merge(raw_data, ratios, how='left', on=['SimFinId', 'Ticker', 'Date'], validate='1:1')

    return raw_data


def intro_clean_data(raw_joined_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function selects necessary columns and cleans data types.

    Args:
        raw_joined_data:

    Returns:
        Pandas dataframe of cleaned data.
    """
    # Select columns
    raw_data_2 = raw_joined_data[['Ticker', 'SimFinId', 'Company Name', 'Sector', 'Industry',
                                  'Date', 'Open', 'Adj. Close', 'Volume', 'Market-Cap', 'Price to Book Value',
                                  'Price to Free Cash Flow (quarterly)', 'Price to Free Cash Flow (ttm)']]
    # Rename columns
    raw_data_2 = raw_data_2.rename(columns={'Ticker': 'ticker',
                                            'SimFinId': 'simfinid',
                                            'Company Name': 'comp_name',
                                            'Sector': 'sector',
                                            'Industry': 'industry',
                                            'Date': 'date',
                                            'Open': 'open',
                                            'Adj. Close': 'adj_close',
                                            'Volume': 'volume',
                                            'Market-Cap': 'market_cap',
                                            'Price to Book Value': 'p_b',
                                            'Price to Free Cash Flow (quarterly)': 'p_fcf_quart',
                                            'Price to Free Cash Flow (ttm)': 'p_fcf_ttm'
                                            })
    # Make 'date' as datetime
    raw_data_2.loc[:, 'date'] = pd.to_datetime(raw_data_2['date'])
    # Make 'sector' and 'industry' as categorical type
    raw_data_2.loc[:, 'sector'] = raw_data_2.loc[:, 'sector'].astype('category')
    raw_data_2.loc[:, 'industry'] = raw_data_2.loc[:, 'industry'].astype('category')

    return raw_data_2


def clean_data_eda(intro_cleaned_data: pd.DataFrame):
    """
    This function outputs a pandas profiling json object for viewing.

    Args:
        intro_cleaned_data:

    Returns:
        Pandas profiling json object.
    """
    eda_profile = ProfileReport(intro_cleaned_data, title='Equity Investing Raw Data EDA', explorative=True)
    # As json file
    eda_profile_html = eda_profile.to_html()

    return eda_profile_html


########################################### SimFin API Class (not used) ################################################


# class SimFinAPIDownload:
#     """
#     This class pulls datasets from the SimFin API.
#     """
#     @staticmethod
#     def pull_daily_prices(api_params: Dict) -> pd.DataFrame:
#         # Set your SimFin+ API-key for downloading data.
#         sf.set_api_key(api_params['api_key'])
#
#         # Set the local directory where data-files are stored.
#         sf.set_data_dir(api_params['api_directory'])
#
#         # Download the data from the SimFin server and load into a Pandas DataFrame.
#         df = sf.load_shareprices(variant='daily', market='us')
#
#         return df
#
#     @staticmethod
#     def pull_us_companies(api_params: Dict) -> pd.DataFrame:
#         # Set your SimFin+ API-key for downloading data.
#         sf.set_api_key(api_params['api_key'])
#
#         # Set the local directory where data-files are stored.
#         sf.set_data_dir(api_params['api_directory'])
#
#         # Download the data from the SimFin server and load into a Pandas DataFrame.
#         df = sf.load_companies(market='us')
#
#         return df
#
#     @staticmethod
#     def pull_industries(api_params: Dict) -> pd.DataFrame:
#         # Set your SimFin+ API-key for downloading data.
#         sf.set_api_key(api_params['api_key'])
#
#         # Set the local directory where data-files are stored.
#         sf.set_data_dir(api_params['api_directory'])
#
#         # Download the data from the SimFin server and load into a Pandas DataFrame.
#         df = sf.load_industries()
#
#         return df
#
#     @staticmethod
#     def pull_share_price_ratios(api_params: Dict) -> pd.DataFrame:
#         # Set your SimFin+ API-key for downloading data.
#         sf.set_api_key(api_params['api_key'])
#
#         # Set the local directory where data-files are stored.
#         sf.set_data_dir(api_params['api_directory'])
#
#         # Download the data from the SimFin server and load into a Pandas DataFrame.
#         df = sf.load_derived_shareprices(variant='daily', market='us')
#
#         return df
