DengAI
==============================

Repository for my submission to the DrivenData competition 'DengAI'; predicting the spread of disease.

### Challenge Description

From the DrivenData website description:

*Dengue fever is a mosquito-borne disease that occurs in tropical and sub-tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash, and muscle and joint pain. In severe cases, dengue fever can cause severe bleeding, low blood pressure, and even death.*

*Because it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation. Although the relationship to climate is complex, a growing number of scientists argue that climate change is likely to produce distributional shifts that will have significant public health implications worldwide.*

*In recent years dengue fever has been spreading. Historically, the disease has been most prevalent in Southeast Asia and the Pacific islands. These days many of the nearly half billion cases per year are occurring in Latin America:*

*Using environmental data collected by various U.S. Federal Government agencies—from the Centers for Disease Control and Prevention to the National Oceanic and Atmospheric Administration in the U.S. Department of Commerce—can you predict the number of dengue fever cases reported each week in San Juan, Puerto Rico and Iquitos, Peru?*

### Dependencies

Out of the box anaconda environment with Keras and Tensorflow.

### Exploratory Data Analysis

In depth exploratory data analysis can be found in [Exploratory_Data_Analysis.ipynb] (https://github.com/chrisgschon/DengAI/blob/master/notebooks/Exploratory_Data_Analysis.ipynb)
Here are the key takeways and figures from this analysis:

- Each record has 20 numerical weather/climate related measurements, 3 time related columns and an categorical for the city (sj for San Juan and iq for Iquitos). 

- There are 1455 total records, 936 from San Juan and 519 from Iquitos, relating to a year and 'weekofyear' set of measurements for each city.





### Feature Engineering

### Models

## Results + Submissions

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- Description of the repo.
    ├── data
    |   ├── features       <- Feature sets constructed by feautre engineering notebook.
    │   ├── processed      <- Processed datasets (slight tweaks to raw data)
    │   └── raw            <- The original, immutable data dump.
    │
    ├── submissions        <- submission csvs 
    │
    ├── notebooks          <- Jupyter notebooks. Contains all the core scripts for analysis.
    │
    ├── reports          
    │   └── figures        <- Library of key figures of the project


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
