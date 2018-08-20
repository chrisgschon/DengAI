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

In depth exploratory data analysis can be found in the [Exploratory_Data_Analysis notebook](https://github.com/chrisgschon/DengAI/blob/master/notebooks/Exploratory_Data_Analysis.ipynb)
Here are the key takeways and figures from this analysis:

- Each record has 20 numerical weather/climate related measurements, 3 time related columns and an categorical for the city (sj for San Juan and iq for Iquitos). 

- There are 1455 total records, 936 from San Juan and 519 from Iquitos, relating to a year and 'weekofyear' set of measurements for each city.

Training data view (first 10 rows and 15 columns..)

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/train_data_view.png)

- The training data also contains the number of total cases of dengue disease recorded in that city. This is misisng for the test data. 

- The number of total cases for each city shows heavy seasonality and 'bursty' nature.

- San Juan typically has higher case rate than Iquitos. 

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/total_cases_time_series.png)


- As is common with many time series problems and to be expected in the 'outbreak' nature of the cases, there is high autocorrelation of total cases, for example, sj autocorrelation plot:

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/total_cases_autocorrelation.png)

- Although temporal seasonality appears to be the key driving factor of the number of cases in each city, there are some non-dismissable correlations between weather factors and case rates, particularly relating to heat,  and humidity. For example, in Iquitos we see the following correlations with the output:

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/correlations_feature_cases_iq.png)


- Naturally, correlations between weather measurements are seen. This suggests that careful feature selection/engineering is be needed to avoid overfitting.

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/feature_correlations.png)



### Feature Engineering

A few strategies have been implemented for building feature sets from the raw data. First of all, the data is split by the two cities and ultimately fit separately (using either the same or differing models). The initial strategies included:

- Selected best measurement features (upon exploratory analysis):
    - reanalysis_specific_humidity_g_per_kg
    - reanalysis_dew_point_temp_k
    - station_avg_temp_c
    - station_min_temp_c
  
- Categorical feautures indicating quarters and *weekofyear*.

- Polynomial fit for case counts by week of year. 
    - As seen below, we can fit 6 degree (SJ) and 2 degree (IQ) polynomials to quite accurately capture the average number of cases by week of year. 
    **SJ**:
    
    ![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/iq_weekly_poly_fit.png)
    
    **IQ**:
    
    ![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/sj_weekly_poly_fit.png)
    
- Polynomial interaction terms for selected and complete sets of features.

- Lookback features that allow a static record to use features from previous timesteps. Incorporating these allowed shallow models to access previous states of features. The feature set with a 'lookback' of 10 has achieved the best submission score so far (with a random forest estimator).


**Futures:**
- **Time series forecasting features.**
- **Create feature(s) that try to capture the 'seasons', combining the weather data in some way to find a 'golden' feature/features that has a strong correlation with the output.**
- **Create an automated pipeline for creating combinations of features, maximising over cross validation score of a Random forest, for example.**

### Models

This section will show the most succesful models in terms of submission score so far. The table below shows the package, features, model architecture and parameterisations used. 

| Model        |    Package |    Feature csv | Parameters      | Submission Score  |
| ------------- |:-------------:|:-------------:|:-------------:| -----:|
| RandoForestRegressor       | SKLearn |   Lookback10            | n_estimators = 1000  | 23.2764   |
| RandoForestRegressor       | SKLearn |  Lookback10             |   n_estimators = 1000, max_features = 10, max_depth = 20 | 23.3341 |
| LSTM NN | Keras      |   Lookback10           | n_hidden = 40, 2 layers (one LSTM one Dense) |   24.3486 |


Highest performing model so far produces predictions shown below:

![alt text](https://github.com/chrisgschon/DengAI/blob/master/reports/figures/feature_correlations.png)

A typical workflow for building and submitting a model's predictions has involved (20/8/18):

- Selecting (or building) a feature set using Feature_Engineering.ipynb.
- If lookback features are desired, use Create_Lookback_Features.ipynb to pipe the chosen feature set into one that allows lookback for a chosen number of time steps
- Exploring benchmark models in Model_Benchmarking.ipynb to see if there is some potential in the feature set.
- Using Make_Submission_SKL or Keras_LSTM_Basic.ipynb to choose and fine tune a model architecture.

**Futures:**
- **Explore modelling mappings of total_cases (e.g. log(1 + total_cases))**
- **Dive deeper into the potential of lookback features.. after creating the automated feature engineering pipeline**.
- **Dive deeper into the neural network in Keras implementation, which has shown some early promise.**


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
