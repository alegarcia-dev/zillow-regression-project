################################################################################
#
#
#
#       acquire.py
#
#       Description: This file contains functions used for acquiring the
#           zillow dataset.
#
#       Fields:
#
#           _zillow_db
#           _zillow_file
#
#       Functions:
#
#           wrangle_zillow()
#           get_zillow_data(use_cache)
#           _get_zillow_sql()
#
#
################################################################################

import os
import pandas as pd

from get_db_url import get_db_url
from prepare import prepare_zillow_data, split_data

################################################################################

_zillow_file = 'zillow.csv'
_zillow_db = 'zillow'

################################################################################

def wrangle_zillow() -> tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame
]:
    '''
        Returns the acquired, prepared, and split zillow dataset.
        
        Returns
        -------
        DataFrame: A pandas dataframe containing the prepared and split zillow 
            dataset.
    '''
    
    return split_data(prepare_zillow_data(get_zillow_data()))

################################################################################

def get_zillow_data(use_cache: bool = True) -> pd.core.frame.DataFrame:
    '''
        Return a dataframe containing data from the zillow dataset.

        If a zillow.csv file containing the data does not already
        exist the data will be cached in that file inside the current
        working directory. Otherwise, the data will be read from the
        .csv file.

        Parameters
        ----------
        use_cache: bool, default True
            If True the dataset will be retrieved from a csv file if one
            exists, otherwise, it will be retrieved from the MySQL database. 
            If False the dataset will be retrieved from the MySQL database
            even if the csv file exists.

        Returns
        -------
        DataFrame: A Pandas DataFrame containing the data from the zillow
            dataset is returned.
    '''

    # If the file is cached, read from the .csv file
    if os.path.exists(_zillow_file) and use_cache:
        return pd.read_csv(_zillow_file)
    
    # Otherwise read from the mysql database
    else:
        df = pd.read_sql(get_zillow_sql(), get_db_url(_zillow_db))
        df.to_csv(_zillow_file, index = False)
        return df

################################################################################

def _get_zillow_sql():
    return """
        SELECT
            bedroomcnt,
            bathroomcnt,
            calculatedfinishedsquarefeet,
            taxvaluedollarcnt,
            yearbuilt,
            fips,
            numberofstories,
            basementsqft,
            fireplacecnt,
            heatingorsystemdesc,
            roomcnt,
            garagetotalsqft,
            hashottuborspa,
            poolcnt,
            poolsizesum,
            yardbuildingsqft17
        FROM properties_2017
        JOIN propertylandusetype
            ON propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
            AND (propertylandusetype.propertylandusedesc IN ('Single Family Residential', 'Inferred Single Family Residential'))
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid);
        """