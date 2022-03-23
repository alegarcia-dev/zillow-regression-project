# An Improved Model For Predicting Property Tax Assessed Values

This repository contains all deliverables for the Zillow Regression project including additional files used 
in the process of producing the final deliverables.

**Repository Format**
- README.md: Contains a full outline of the project as well as information regarding the format of the repository 
and instructions for reproducing the results.
- Zillow_Final_Report.ipynb: The final report containing a high level overview of the project including key takeaways, 
final results, and a recommendations.
- notebooks:
    - acquire.ipynb: A detailed and thorough overview of the data acquisition process.
    - prepare.ipynb: A detailed and thorough overview of the data preparation process.
    - explore.ipynb: A detailed and thorough overview of the exploratory analysis process along with key takeaways.
    - model.ipynb: A detailed and thorough overview of the modeling process including key takeaways.
- util:
    - get_db_url.py: Contains function used for accessing the MySQL database.
    - acquire.py: Contains functions used for acquiring the property data.
    - prepare.py: Contains functions used for preparing and tidying the property data.
    - explore.py: Contains functions used for visualizing key findings.
    - model.py: Contains functions used for producing and visualizing ML model results.
---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Project Description](#project-description)
3. [Initial Questions](#initial-questions)
4. [Data Dictionary](#data-dictionary)
5. [Instructions for Reproducing the Results](#instructions-for-reproducing-the-results)
6. [Outline of Project Plan](#outline-of-project-plan)
    1. [Data Acquisition](#data-acquisition)
    2. [Data Preparation](#data-preparation)
    3. [Exploratory Analysis](#exploratory-analysis)
    4. [Modeling](#modeling)
7. [Key Takeaways and Recommendations](#key-takeaways-and-recommendations)

## Project Goals

Identify attributes that can be used to predict property tax assessed values, build a prediction model that is an improvement upon the existing model, 
and offer recommendations on what works or doesn't work.

## Project Description

The Zillow data science team would like to predict property tax assessed values for single family properties that had a transaction in 2017. A model 
already exists, but the Zillow data science team is hoping an outside perspective can lead to an improved model. We will analyze the data available 
and produce an improved prediction model. We will compare the new model to the existing one and offer recommendations on what works or doesn't work 
for acquiring the most accurate predictions.

## Initial Questions

We know that bedroom count, bathroom count, and square footage all have an influence on a property's value. However, we can ask additional questions 
about the data to determine if other attributes may have an impact. Initial analysis of the data was conducted by answering these questions:

- What influence do bedroom count, bathroom count, and square footage have on property value?
- Is there a relationship between the age of a property and its value?
- Is there a relationship between the number of stories a property has and its value?
- Does a basement increase a property's value?
- Does a fireplace increase a property's value?
- Is there a relationship between heating or system type of a property and its value?
- Does location relate to a property's value?
- Does the total room count affect the value of a property?
- Is there a relationship between garage size and property value?
- Does a hot tub or pool increase a property's value?
- Is there a relationship between the size of a property's patio and its value?

## Data Dictionary

| Variable              | Meaning      |
| --------------------- | ------------ |
| bedroom_count         | Number of bedrooms in home |
| bathroom_count        | Number of bathrooms in home including fractional bathrooms |
| square_feet           | Calculated total finished living area of the home |
| property_tax_assessed_values | The total tax assessed value of the parcel |
| year_built            | The Year the principal residence was built |
| fed_code              | Federal Information Processing Standard code |
| basement_square_feet  | Finished living area below or partially below ground level |
| fireplace_count       | Number of fireplaces in a home (if any) |
| garage_square_feet    | Total number of square feet of all garages on lot including an attached garage |
| has_hot_tub           | Does the home have a hot tub or spa |
| has_pool              | Total square footage of all pools on property |
| yardbuildingsqft17    | Patio in yard |
| numberofstories       | Number of stories or levels the home has |
| heatingorsystemdesc   | Type of home heating system |
| propertylandusedesc   | Type of land use the property is zoned for |
| roomcnt               | Total number of rooms in the principal residence |


## Instructions for Reproducing the Results



## Outline of Project Plan
---
### Data Acquisition

In this phase the zillow property data is acquired from the MySQL database. The dataset is quite large so it would be inefficient to pull 
the full dataset from the database. With that in mind we have to look at our initial questions to determine which columns we should select 
and only pull those columns for the database. Additionally we must look at our requirements and ensure that our acquired data adheres to 
our requirements.

- The acquire.ipynb notebook in the notebooks directory contains a reproducible step by step process for acquiring the data with details 
and explanations.

- The acquire.py file in the util directory contains all the data acquisition functions used in the final report notebook.

**Steps Taken:**
1. Create an SQL query that will select only the data that is needed.
2. Create and test function to acquire our data from either the database or .csv file if it exists.
3. Create a wrangle function that will acquire, prepare, and split our data in a single step.

### Data Preparation

In this phase the zillow property data is prepared so that it will be ready for exploration and modeling. The dataset initially contains a large number of missing values and outliers. In the first pass outliers are removed (in a second iteration they can be left in to see how this affects the results). Missing values are handled in various ways depending on the context. Additionally, any columns that are not removed are renamed for readability.

Preparing the data allows us to see through the noise and focus on the data that is useful to the problem at hand.

- The prepare.ipynb notebook in the notebooks directory contains a reproducible step by step process for preparing the data with details and explanations.

- The univariate_analysis.py file in the notebooks directory contains some utility functions used in the the prepare notebook.

- The prepare.py file in the util directory contains all the data preparation functions used in the final report notebook.

**Steps Taken:**
1. Analyze data to determine if there are unusual values or outliers that should be removed.
2. Analyze any columns with missing values and determine the best course of action for handling the missing values.
3. For any columns that are a type that doesn't make sense for the data within cast the column to a more reasonable type.
4. Rename columns for readability.
5. Wrap all steps in a single function and test it to ensure that it works.

### Exploratory Analysis



### Modeling



---
## Key Takeaways and Recommendations



[Back to top](#an-improved-model-for-predicting-property-tax-assessed-values )