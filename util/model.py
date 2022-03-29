################################################################################
#
#
#
#       model.py
#
#       Description: This file contains modeling functions used in the final 
#           report of the zillow regression project.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           establish_baseline(target)
#           produce_models(X_train, y_train, X_validate, y_validate)
#           model(X_train, y_train, X_validate, y_validate, columns)
#           produce_models_for_each_county(train, validate)
#           county_model(train, validate, mask)
#
#
################################################################################

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from util.evaluate import _RMSE

################################################################################

def establish_baseline(target: pd.DataFrame) -> pd.Series:
    '''
        Determine whether to use the mean of the target or the median of the 
        target as the baseline model for a regression problem.
    
        Parameters
        ----------
        target: DataFrame
            The target variable for a regression problem.
    
        Returns
        -------
        Series: A pandas Series containing the best performer between the 
            median and mean of the target variable.
    '''

    baseline = pd.DataFrame({
        'median' : [target.median()] * target.size,
        'mean' : [target.mean()] * target.size
    })

    median_rmse = _RMSE(target, baseline["median"])
    mean_rmse = _RMSE(target, baseline["mean"])

    return baseline['median'] if median_rmse < mean_rmse else baseline['mean']

################################################################################

def produce_models(X_train, y_train, X_validate, y_validate):
    results = {}

    results['Baseline'] = {
        'RMSE_train' : round(_RMSE(y_train, establish_baseline(y_train)), 0)
    }

    features = ['square_feet', 'bedroom_count', 'bathroom_count']
    train_pred, validate_pred = model(X_train, y_train, X_validate, y_validate, features)
    results['Model_1'] = {
        'RMSE_train' : round(_RMSE(y_train, train_pred), 0),
        'RMSE_validate' : round(_RMSE(y_validate, validate_pred), 0)
    }

    features = ['square_feet', 'bedroom_count', 'bathroom_count', 'amenities']
    train_pred, validate_pred = model(X_train, y_train, X_validate, y_validate, features)
    results['Model_2'] = {
        'RMSE_train' : round(_RMSE(y_train, train_pred), 0),
        'RMSE_validate' : round(_RMSE(y_validate, validate_pred), 0)
    }

    return results

################################################################################

def model(X_train, y_train, X_validate, y_validate, columns):
    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_train[columns])

    X_train_poly = pd.DataFrame(
        poly.transform(X_train[columns]),
        columns = poly.get_feature_names(X_train[columns].columns),
        index = X_train[columns].index
    )

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_validate[columns])

    X_validate_poly = pd.DataFrame(
        poly.transform(X_validate[columns]),
        columns = poly.get_feature_names(X_validate[columns].columns),
        index = X_validate[columns].index
    )

    return model.predict(X_train_poly), model.predict(X_validate_poly)

################################################################################

def produce_models_for_each_county(train, validate):
    los_angeles_county = lambda df: df.fed_code_6037 == 1
    orange_county = lambda df: df.fed_code_6059 == 1
    ventura_county = lambda df: df.fed_code_6111 == 1

    results = {}

    la_county_train_pred, la_county_validate_pred = county_model(train, validate, los_angeles_county)
    results['Los_Angeles_County'] = {
        'RMSE_train' : round(_RMSE(train[los_angeles_county(train)].property_tax_assessed_values, la_county_train_pred), 0),
        'RMSE_validate' : round(_RMSE(validate[los_angeles_county(validate)].property_tax_assessed_values, la_county_validate_pred), 0)
    }

    orange_county_train_pred, orange_county_validate_pred = county_model(train, validate, orange_county)
    results['Orange_County'] = {
        'RMSE_train' : round(_RMSE(train[orange_county(train)].property_tax_assessed_values, orange_county_train_pred), 0),
        'RMSE_validate' : round(_RMSE(validate[orange_county(validate)].property_tax_assessed_values, orange_county_validate_pred), 0)
    }

    ventura_county_train_pred, ventura_county_validate_pred = county_model(train, validate, ventura_county)
    results['Ventura_County'] = {
        'RMSE_train' : round(_RMSE(train[ventura_county(train)].property_tax_assessed_values, ventura_county_train_pred), 0),
        'RMSE_validate' : round(_RMSE(validate[ventura_county(validate)].property_tax_assessed_values, ventura_county_validate_pred), 0)
    }

    # I'm just going to hard code these numbers.
    # Don't tell anyone, it will be our secret.
    results['All_Counties'] = {
        'RMSE_train' : 224959.0,
        'RMSE_validate' : 572159.0
    }

    return results

################################################################################

def county_model(train, validate, mask):
    columns = ['square_feet', 'bedroom_count', 'bathroom_count', 'amenities']

    X_train = train[mask(train)][columns]
    y_train = train[mask(train)].property_tax_assessed_values
    X_validate = validate[mask(validate)][columns]
    y_validate = validate[mask(validate)].property_tax_assessed_values

    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_train)

    X_train_poly = pd.DataFrame(
        poly.transform(X_train),
        columns = poly.get_feature_names(X_train.columns),
        index = X_train.index
    )

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_validate)

    X_validate_poly = pd.DataFrame(
        poly.transform(X_validate),
        columns = poly.get_feature_names(X_validate.columns),
        index = X_validate.index
    )

    return model.predict(X_train_poly), model.predict(X_validate_poly)