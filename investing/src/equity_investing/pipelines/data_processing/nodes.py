# This is the data_processing nodes file
# Import libraries
import pandas as pd


def join_data(companies: pd.DataFrame, industries: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates raw data set from SimFin 'us-companies',
    'industries', and 'us-shareprices-daily' files.

    Args:
        companies: 'us-companies' data set from SimFin
        industries: 'industries' data set from SimFin
        prices: 'us-shareprices-daily' data set from SimFin

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
                                  'Date', 'Adj. Close', 'Volume', 'Shares Outstanding']]
    # Rename columns
    raw_data_2 = raw_data_2.rename(columns={'Ticker': 'ticker',
                                            'SimFinId': 'simfinid',
                                            'Company Name': 'comp_name',
                                            'Sector': 'sector',
                                            'Industry': 'industry',
                                            'Date': 'date',
                                            'Adj. Close': 'adj_close',
                                            'Volume': 'volume',
                                            'Shares Outstanding': 'shares_out'})
    # Make 'date' as datetime
    raw_data_2.loc[:, 'date'] = pd.to_datetime(raw_data_2['date'])
    # Make 'sector' and 'industry' as categorical type
    raw_data_2.loc[:, 'sector'] = raw_data_2.loc[:, 'sector'].astype('category')
    raw_data_2.loc[:, 'industry'] = raw_data_2.loc[:, 'industry'].astype('category')

    return raw_data_2
