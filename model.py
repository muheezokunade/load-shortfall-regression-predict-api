"""
    Helper functions for the pretrained model to be used within our API.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------
    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  
"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.
    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.
    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

    # fill the missing value in Valencia_pressure with the mean, which is 1012.
    #Train Data
    feature_vector_df.fillna(feature_vector_df.Valencia_pressure.mean(), inplace= True)
    # Connvert both Seville_pressure to numerical columns by scrapping off the 'sp' infrom of it
    feature_vector_df.Seville_pressure = feature_vector_df.Seville_pressure.str[2:]
    # Convert to data type int
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].astype(int) 
    # Connvert both Valencia_wind_deg to numerical columns by scrapping off the 'level_' infrom of it

    feature_vector_df.Valencia_wind_deg = feature_vector_df.Valencia_wind_deg.str[6:]
    # Convert to data type int
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].astype(int)
    """
    We had to convert the time type from an object to a datetime format using the 'astype' method before desampling

    """
    feature_vector_df['Year']  = feature_vector_df['time'].astype('datetime64').dt.year
    feature_vector_df['Month_of_year']  = feature_vector_df['time'].astype('datetime64').dt.month
    feature_vector_df['Week_of_year'] = feature_vector_df['time'].astype('datetime64').dt.weekofyear
    feature_vector_df['Day_of_year']  = feature_vector_df['time'].astype('datetime64').dt.dayofyear
    feature_vector_df['Day_of_month']  = eature_vector_df['time'].astype('datetime64').dt.day
    feature_vector_df['Day_of_week'] = feature_vector_df['time'].astype('datetime64').dt.dayofweek
    feature_vector_df['Hour_of_week'] = ((feature_vector_df['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df['time'].astype('datetime64').dt.hour)
    feature_vector_df['Hour_of_day']  = feature_vector_df['time'].astype('datetime64').dt.hour

    feature_vector_df = feature_vector_df.drop(columns=['Week_of_year','Day_of_year','Hour_of_week','time'])
    feature_vector_df = feature_vector_df[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Seville_temp_max', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.
    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.
    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.
    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.
    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.
    Returns
    -------
    list
        A 1-D python list containing the model prediction.
    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()