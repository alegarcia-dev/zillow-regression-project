################################################################################
#
#
#
#       explore.py
#
#       Description: This file contains visualization functions used in the final
#           report for the zillow regression project.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           plot_property_value_distribution(df)
#           plot_primary_features_vs_target(df)
#           plot_quality_and_age(df)
#           plot_amenities_and_location(df)
#           statistical_tests(df, variable)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import util.stats_util as stats_util

################################################################################

def plot_property_value_distribution(df: pd.DataFrame) -> None:
    '''
        Plot a boxplot and histogram showing the distribution of values in 
        property values column.
    
        Parameters
        ----------
        df: DataFrame
            The train dataset from the zillow property data.
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 3))
    mean = df.property_tax_assessed_values.mean()

    sns.boxplot(data = df, x = 'property_tax_assessed_values', ax = ax[0])
    ax[0].set_title('The target variable has a lot of outliers.')

    mask = df.property_tax_assessed_values < 1_000_000
    sns.histplot(df[mask].property_tax_assessed_values, bins = 10)
    ax[1].set_title('Properties with tax assessed value less than $1,000,000.')

    plt.show()

################################################################################

def plot_primary_features_vs_target(df: pd.DataFrame) -> None:
    '''
        Plot bedroom count, bathroom count, and square feet versus the 
        property value.
    
        Parameters
        ----------
        df: DataFrame
            The train dataset from the zillow property data.
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 3))
    mean = df.property_tax_assessed_values.mean()
    mask = df.property_tax_assessed_values < 1_000_000

    sns.regplot(data = df[mask], x = 'bedroom_count', y = 'property_tax_assessed_values', ax = ax[0], line_kws = {'color' : 'red'})
    ax[0].axhline(mean, ls='--', color='grey')
    ax[0].ticklabel_format(style = 'plain')
    ax[0].set_title('Property value increases as bedroom count increases.')

    sns.regplot(data = df[mask], x = 'bathroom_count', y = 'property_tax_assessed_values', ax = ax[1], line_kws = {'color' : 'red'})
    ax[1].axhline(mean, ls='--', color='grey')
    ax[1].ticklabel_format(style = 'plain')
    ax[1].set_title('Property value increases as bathroom count increases.')

    sns.regplot(data = df[mask], x = 'square_feet', y = 'property_tax_assessed_values', ax = ax[2], line_kws = {'color' : 'red'})
    ax[2].axhline(mean, ls='--', color='grey')
    ax[2].ticklabel_format(style = 'plain')
    ax[2].set_title('Property value increases as square footage increases.')

    plt.show()

################################################################################

def plot_quality_and_age(df: pd.DataFrame) -> None:
    '''
        Plot building quality and property age versus the property value.
    
        Parameters
        ----------
        df: DataFrame
            The train dataset from the zillow property data.
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 3))
    mean = df.property_tax_assessed_values.mean()
    mask = df.property_tax_assessed_values < 1_000_000

    sns.lineplot(data = df[mask], x = 'property_age', y = 'property_tax_assessed_values', ax = ax[0])
    ax[0].axhline(mean, ls='--', color='grey')
    ax[0].set_title('Property value decreases as age increases.')

    sns.lineplot(data = df[mask], x = 'building_quality', y = 'property_tax_assessed_values', ax = ax[1])
    ax[1].axhline(mean, ls='--', color='grey')
    ax[1].set_title('Property value tends to increase as building quality gets worse?')

    plt.show()

################################################################################

def plot_amenities_and_location(df: pd.DataFrame) -> None:
    '''
        Plot total amenities and location versus the property value.
    
        Parameters
        ----------
        df: DataFrame
            The train dataset from the zillow property data.
    ''' 

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 3))
    mean = df.property_tax_assessed_values.mean()
    mask = df.property_tax_assessed_values < 1_000_000

    sns.lineplot(data = df, x = "amenities", y = "property_tax_assessed_values", ax = ax[0])
    ax[0].axhline(mean, ls = '--', color = 'grey')
    ax[0].ticklabel_format(style = 'plain')
    ax[0].set_title('Property value increases as the total number of amenities increases.')

    sns.boxplot(data = df[mask], x = 'fed_code_6037', y = 'property_tax_assessed_values', ax = ax[1])
    ax[1].axhline(mean, ls = '--', color = 'grey')
    ax[1].set_title('Properties in Los Angeles County have lower value on average.')

    plt.show()

################################################################################

def statistical_tests(df: pd.DataFrame, variable: str) -> None:


    stats_util.correlation_test(df[variable], df.property_tax_assessed_values)