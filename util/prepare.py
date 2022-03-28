################################################################################

#
#
#       prepare.py
#
#       Description: This file contains commonly used preparation functions.
#
#       Fields:
#
#           None
#
#       Functions:
#
#           prepare_zillow_data(df)
#           split_data(df, stratify, random_seed = 24)
#           remove_outliers(df, k, col_list)
#           scale_data(train, validate, test)
#           _fill_missing_values(df)
#           _drop_columns(df)
#           _cast_columns(df)
#
#
################################################################################

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

################################################################################

def prepare_zillow_data(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Returns a prepared zillow dataset with all missing values handled.
        
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the unprepared zillow dataset.
        
        Returns
        -------
        DataFrame: A pandas dataframe containing the prepared zillow dataset.
    '''

    missing_target = df.taxvaluedollarcnt.isnull()
    df = df[~missing_target]
    
    df = _fill_missing_values(df)
    df = _cast_columns(df)

    df.fips = df.fips.astype('int')
    df.fips = df.fips.astype('object')
    df = pd.get_dummies(df, columns = ['fips'])

    df['property_age'] = 2017 - df.yearbuilt
    df = df.drop(columns = 'yearbuilt')

    df['amenities'] = df.has_hot_tub + df.has_pool + (df.fireplace_count > 0)

    df = _drop_columns(df)

    # Rename the columns for readability
    df = df.rename(columns = {
        'bedroomcnt' : 'bedroom_count',
        'bathroomcnt' : 'bathroom_count',
        'calculatedfinishedsquarefeet' : 'square_feet',
        'taxvaluedollarcnt' : 'property_tax_assessed_values'
        'buildingqualitytypeid' : 'building_quality',
        'lotsizesquarefeet' : 'lot_size',
        'fips_6037' : 'fed_code_6037'
    })
    
    return df

################################################################################

def split_data(df: pd.core.frame.DataFrame, random_seed: int = 24, stratify: str = None) -> tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame
]:
    '''
        Accepts a DataFrame and returns train, validate, and test DataFrames.
        Splits are performed randomly.

        Proportion of original dataframe that each return dataframe comprises.
        ---------------
        Train:      56% (70% of 80%)
        Validate:   24% (30% of 80%)
        Test:       20%

        Parameters
        ----------
        df : DataFrame
            A Pandas DataFrame containing prepared data. It is expected that
            the input to this function will already have been prepared and
            tidied so that it will be ready for exploratory analysis.

        stratify : str
            A string value containing the name of the column to be stratified
            in the sklearn train_test_split function. This parameter should
            be the name of a column in the df dataframe.

        random_seed : int, default 24
            An integer value to be used as the random number seed. This parameter
            is passed to the random_state argument in the sklearn train_test_split
            function.

        Returns
        -------
        tuple : A tuple containing three Pandas DataFrames for train, validate
            and test datasets.    
    '''
    test_split = 0.2
    train_validate_split = 0.3

    train_validate, test = train_test_split(
        df,
        test_size = test_split,
        random_state = random_seed,
        stratify = stratify if not stratify else df[stratify]
    )
    train, validate = train_test_split(
        train_validate,
        test_size = train_validate_split,
        random_state = random_seed,
        stratify = stratify if not stratify else train_validate[stratify]
    )
    return train, validate, test

################################################################################

def remove_outliers(df: pd.core.frame.DataFrame, k: float, col_list: list[str]) -> pd.core.frame.DataFrame:
    '''
        Remove outliers from a list of columns in a dataframe 
        and return that dataframe.
        
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing data from which we want to remove
            outliers.
        
        k: float
            A numeric value that indicates how strict our outlier threshold
            should be. Typically 1.5.

        col_list: list[str]
            A list of columns from which we want to remove outliers.
        
        Returns
        -------
        DataFrame: A pandas dataframe with outliers removed.
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

################################################################################

def scale_data(train: pd.core.frame.DataFrame, validate: pd.core.frame.DataFrame, test: pd.core.frame.DataFrame) -> tuple (
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame
):
    '''
        Scale all numeric columns using a MinMaxScaler.
    
        Parameters
        ----------
        train: DataFrame
            The training dataset for a machine learning problem.

        validate: DataFrame
            The out of sample validate dataset for a machine learning problem.

        test: DataFrame
            The out of sample test dataset for a machine leraning problem.
    
        Returns
        -------
        tuple(DataFrame): A tuple of three dataframes with all the numeric 
            columns scaled.
    '''

    scaler = MinMaxScaler()
    numeric_columns = train.select_dtypes('number').columns
    
    train[numeric_columns] = scaler.fit_transform(train[numeric_columns])
    validate[numeric_columns] = scaler.transform(validate[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])
    
    return train, validate, test

################################################################################

def _fill_missing_values(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Fill in all missing values with 0 and return the dataframe.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the zillow dataset.
    
        Returns
        -------
        DataFrame: A pandas dataframe containing the zillow dataset with all 
            missing values filled with 0.
    '''

    avg_square_feet = df.calculatedfinishedsquarefeet.mean()
    df.calculatedfinishedsquarefeet.fillna(avg_square_feet, inplace = True)

    df.yearbuilt.fillna(df.yearbuilt.mode()[0], inplace = True)
    df.fireplacecnt.fillna(0, inplace = True)
    df.hashottuborspa.fillna(0, inplace = True)
    df.poolcnt.fillna(0, inplace = True)

    median_quality = df.buildingqualitytypeid.median()
    df.buildingqualitytypeid.fillna(median_quality, inplace = True)

    median_lotsize = df.lotsizesquarefeet.median()
    df.lotsizesquarefeet.fillna(median_lotsize, inplace = True)

    return df

################################################################################

def _drop_columns(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Drop all columns provided from the dataframe.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the zillow dataset.
    
        Returns
        -------
        DataFrame: A pandas dataframe containing the zillow dataset with the 
            provided columns dropped.
    '''

    df = df.drop(columns = 'parcelid')
    df = df.drop(columns = 'numberofstories')
    df = df.drop(columns = 'basementsqft')
    df = df.drop(columns = 'heatingorsystemdesc')
    df = df.drop(columns = 'airconditioningdesc')
    df = df.drop(columns = 'garagetotalsqft')
    df = df.drop(columns = 'poolsizesum')
    df = df.drop(columns = 'yardbuildingsqft17')
    df = df.drop(columns = 'roomcnt')
    df = df.drop(columns = 'finishedfloor1squarefeet')
    df = df.drop(columns = 'finishedsquarefeet15')
    df = df.drop(columns = 'fips_6059')
    df = df.drop(columns = 'fips_6111')
    df = df.drop(columns = 'fireplacecnt')
    df = df.drop(columns = 'hashottuborspa')
    df = df.drop(columns = 'poolcnt')

    return df

################################################################################

def _cast_columns(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Cast all provided columns to the int type.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the zillow dataset.
    
        Returns
        -------
        DataFrame: A pandas dataframe containing the zillow dataset with the 
            provided columns cast to the int type.
    ''' 

    df.bedroomcnt = df.bedroomcnt.astype('int')
    df.fireplacecnt = df.fireplacecnt.astype('int')
    df.hashottuborspa = df.hashottuborspa.astype('int')
    df.poolcnt = df.poolcnt.astype('int')

    return df