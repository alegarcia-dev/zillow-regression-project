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
- Is there a relationship between the construction type of a property and its value?
- Does a basement increase a property's value?
- Does a fireplace increase a property's value?
- Is there a relationship between heating or system type of a property and its value?
- Does location relate to a property's value?
- Is there a relationship between the architectural style of a property and its value?

## Data Dictionary

| Variable              | Meaning      |
| --------------------- | ------------ |
| bedroomcnt            | Number of bedrooms in home |
| bathroomcnt           | Number of bathrooms in home including fractional bathrooms |
| calculatedfinishedsquarefeet | Calculated total finished living area of the home |
| taxvaluedollarcnt     | The total tax assessed value of the parcel |
| yearbuilt             | The Year the principal residence was built |
| fips                  | Federal Information Processing Standard code |
| numberofstories       | Number of stories or levels the home has |
| basementsqft          | Finished living area below or partially below ground level |
| fireplacecnt          | Number of fireplaces in a home (if any) |
| typeconstructiondesc  | What type of construction material was used to construct the home |
| heatingorsystemdesc   | Type of home heating system |
| architecturalstyledesc | Architectural style of the home (i.e. ranch, colonial, split-level, etc…) |
| propertylandusedesc   | Type of land use the property is zoned for |


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



### Exploratory Analysis



### Modeling



---
## Key Takeaways and Recommendations



[Back to top](#an-improved-model-for-predicting-property-tax-assessed-values )