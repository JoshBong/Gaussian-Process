#libraries
import os
import pandas as pd
import pickle
import numpy as np
import datamol as dm
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.svm import SVR
from sklearn.model_selection import KFold



DATA = 'smiles-RedPot.csv'
TO_PRED = 'S0red-Quinone-DFT (1).xlsx'



def combine_and_pickle(data_list, pickle_filename):
    """
    Combine data from a list of file paths (CSV, XLSX) or DataFrames into one DataFrame
    and pickle it to a file.

    Parameters:
    data_list (list): A list of file paths (strings), DataFrames, or a mix of both.
    pickle_filename (str): The name of the pickle file to save the combined DataFrame.

    Returns:
    combined_df (pd.DataFrame): The combined DataFrame.
    """

    combined_dfs = []

    for item in data_list:
        try:
            if isinstance(item, str):
                if item.lower().endswith('.csv'):
                    df = pd.read_csv(item)
                elif item.lower().endswith('.xlsx'):
                    df = pd.read_excel(item)
                else:
                    print(f"Warning: Unsupported file format for {item}. Only CSV or XLSX files are supported.")
                    continue  # Skip this item and move to the next one
            elif isinstance(item, pd.DataFrame):
                df = item
            else:
                print(f"Warning: The item must be a file path or a DataFrame. Skipping item: {item}")
                continue  # Skip this item and move to the next one

            combined_dfs.append(df)

        except Exception as e:
            print(f"An error occurred while processing {item}: {e}")
            continue  # Skip this item and move to the next one

    if not combined_dfs:
        raise ValueError("No valid data was provided to combine.")

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Pickle the combined DataFrame to the specified file
    with open(pickle_filename, 'wb') as file:
        pickle.dump(combined_df, file)

    print(f"Data has been successfully combined and pickled to {pickle_filename}.")
    return combined_df

# Example usage:
# data_list = ["data1.csv", "data2.xlsx", pd.DataFrame({'A': [1, 2], 'B': [3, 4]})]
# pickle_filename = "combined_data.pkl"
# combined_data = combine_and_pickle(data_list, pickle_filename)



def prepare_data_for_gpr(data, column_to_drop_nan=None):
    """
    Prepare the data for Gaussian Process Regression (GPR) by loading from a file or DataFrame and dropping rows with NaN values in a specific column.

    Parameters:
    data (str or pd.DataFrame): Either a file path (str) or a DataFrame. If a file path is provided, it can be a pickle, CSV, or Excel file.
    column_to_drop_nan (str): The name of the column to check for NaN values to drop.

    Returns:
    prepared_df (pd.DataFrame): The DataFrame with NaN values dropped from the specified column.
    """

    if isinstance(data, pd.DataFrame):
        # If data is already a DataFrame, handle NaN values directly
        if column_to_drop_nan and column_to_drop_nan in data.columns:
            prepared_df = data.loc[data[column_to_drop_nan].notnull(), :]
        else:
            prepared_df = data
        return prepared_df

    elif isinstance(data, str):
        # If data is a string, determine the file type and load accordingly
        try:
            if data.endswith('.pkl'):
                with open(data, 'rb') as file:
                    prepared_df = pickle.load(file)
            elif data.endswith('.csv'):
                prepared_df = pd.read_csv(data)
            elif data.endswith('.xlsx'):
                prepared_df = pd.read_excel(data)
            else:
                print(f"Error: The file '{data}' has an unsupported extension.")
                return None

            # Drop rows where the specified column has NaN values
            if column_to_drop_nan and column_to_drop_nan in prepared_df.columns:
                prepared_df = prepared_df.loc[prepared_df[column_to_drop_nan].notnull(), :]
            return prepared_df

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        print("Error: The 'data' parameter should be either a DataFrame or a file path string.")
        return None

# Usage example:
# prepared_df = prepare_data_for_gpr('my_data.csv', 'column_to_check_for_nan')



def transfer_smiles_to_descriptor(data, column_name, descriptor_type):
    """
    Transfer SMILES strings in a DataFrame to the designated descriptor type (fingerprints or weighted fingerprints).

    Parameters:
    data (pd.DataFrame or str): Either a DataFrame or a pickle filename (str) containing molecular data.
    column_name (str): The name of the column where SMILES strings are stored.
    descriptor_type (str): The type of descriptor to be generated ("ecfp" for regular or "ecfp-count" for weighted fingerprints).

    Returns:
    data (pd.DataFrame): The DataFrame with an additional column for the generated descriptor.
    """

    # Load data from a pickle file if a filename is provided
    if isinstance(data, str):
        try:
            with open(data, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: The pickle file '{data}' was not found.")
            return None

    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Error: Invalid data type. Provide either a DataFrame or a pickle filename.")
        return None

    # Validate descriptor type
    if descriptor_type not in ["ecfp", "ecfp-count"]:
        raise ValueError("Invalid descriptor_type. Supported types: 'ecfp', 'ecfp-count'.")

    # Prepare the descriptors
    fingerprints = []

    # Use the training data approach for generating fingerprints
    for smiles in data[column_name]:
        mol = dm.to_mol(smiles, sanitize=True)

        # Generate the specified type of fingerprint
        if descriptor_type == "ecfp":
            fp = dm.to_fp(mol)
        elif descriptor_type == "ecfp-count":
            fp = dm.to_fp(mol, count=True)  # Assuming 'count=True' generates weighted fingerprints
        else:
            continue  # If descriptor_type is somehow invalid, skip this iteration

        fingerprints.append(fp)

    # Add the fingerprints to the DataFrame
    data[descriptor_type] = fingerprints

    return data

# Example usage:
# Assuming you have a DataFrame 'df' with a 'smiles' column
# result = transfer_smiles_to_descriptor(df, column_name='smiles', descriptor_type='ecfp')



def ensure_column_2d(dataframe, column_name):
    """
    Ensures that the specified column in a DataFrame is in a 2D format.

    If the column is 1D, it reshapes it to 2D with each element as a separate sample.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to reshape.

    Returns:
    pd.DataFrame: The DataFrame with the specified column reshaped to 2D.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame.")

    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    column_data = dataframe[column_name]

    # Check if the column data is 1D and reshape to 2D if necessary
    if column_data.ndim == 1:
        dataframe[column_name] = column_data.values.reshape(-1, 1)
    elif column_data.ndim != 2:
        raise ValueError("Column data must be either 1D or 2D.")

    return dataframe

# Example usage
# df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [6, 7, 8, 9, 10]})
# updated_df = ensure_column_2d(df, 'feature1')
# use this on the smile strings



def split_data(data, train_percent=0.8, random_state=None):
    """
    Split a DataFrame into two DataFrames for training and testing.

    Parameters:
    data (str, pd.DataFrame): Either a pickle filename (str) or an existing DataFrame.
    train_percent (float): The percentage of data to include in the training DataFrame (default is 80%).
    random_state (int): A seed to ensure reproducibility when shuffling data.

    Returns:
    train_df (pd.DataFrame): The DataFrame for training.
    test_df (pd.DataFrame): The DataFrame for testing.
    """
    # Load data from a pickle file if a filename is provided
    if isinstance(data, str):
        try:
            with open(data, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Invalid data type. Provide either a DataFrame or a pickle filename.")

    # Split the data into training and testing DataFrames
    train_df, test_df = train_test_split(data, train_size=train_percent, random_state=random_state, shuffle=True)

    return train_df, test_df

# Example usage:
# Assuming you have a DataFrame 'df' or a pickled DataFrame filename 'data.pkl'
# train_data, test_data = split_data('data.pkl', train_percent=0.8, random_state=42)
# or
# train_data, test_data = split_data(df, train_percent=0.8, random_state=42)



def gpr_pred(train_data, test_data, column1, column2, column3):
    """
    Perform Gaussian Process Regression (GPR) prediction and visualization.

    Parameters:
    train_data (pd.DataFrame): Training data containing features and actual values.
    test_data (pd.DataFrame): Test data containing features and actual values.
    column1 (str): Name of the column in 'train_data' and 'test_data' containing features.
    column2 (str): Name of the column in 'train_data' containing actual values for training.
    column3 (str): Name of the column in 'test_data' containing actual values for testing.

    Returns:
    None
    """

    # Extract data from DataFrames
    X_train = np.array(train_data[column1].tolist())
    y_train = np.array(train_data[column2].tolist())

    X_test = np.array(test_data[column1].tolist())
    y_actual = test_data[column3].tolist()

    # Define Gaussian Process Regression model with a specific kernel
    kernel = WhiteKernel() + RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=69, normalize_y=True)

    # Train the model with training data
    gpr.fit(X_train, y_train)

    # Make predictions on test data
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Create a DataFrame for actual and predicted values
    df = pd.DataFrame({'y_actual': y_actual, 'y_pred': y_pred, 'std': y_std})

    # Calculate and print evaluation metrics
    mse = round(mean_squared_error(y_actual, y_pred), 3)
    mae = round(mean_absolute_error(y_actual, y_pred), 3)
    r2 = round(r2_score(y_actual, y_pred), 2)

    # Create a scatter plot with Plotly
    fig = px.scatter(df, x='y_actual', y='y_pred', color='std', trendline='ols',
                     labels={"y_actual": "Actual Values", "y_pred": "Predicted Values"},
                     width=500, height=500)

    # Add annotation with evaluation metrics
    fig.add_annotation(text=f"MSE: {mse}, MAE: {mae}, R\u00B2: {r2}",
                       xref="paper", yref="paper", x=0.5, y=0, showarrow=False)

    # Customize the plot
    fig.update_layout(font_family="Arial", font_color="black", title_text="GPR Predictions", title_x=0.5)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    # Save the plot as an image
    fn = 'gpr-results/predictions.png'
    fig.write_image(fn)

    # Save predictions to a CSV file
    df.to_csv('gpr-results/gpr_predictions.csv')

# Example usage:
# gpr_pred(train_data, test_data, 'feature_column', 'target_column_train', 'target_column_test')



def main():

# excluding "Results-001.csv" as it is going to be used to train the data instead
    #data_list = ["Results-002.csv", "Results-003.csv", "Results-004.csv",
    #         "Results-005.csv", "Results-006.csv", "Results-007.csv", "Results-008.csv",
    #         "Results-009.csv", "Results-010.csv"]


    #pickle_filename = "CombinedResults.pkl"
    #combined_data = combine_and_pickle(data_list, pickle_filename)

    df = pd.read_csv(DATA)
    data = prepare_data_for_gpr(df, column_to_drop_nan = 'SMILES_step1')
    data = transfer_smiles_to_descriptor(data, 'SMILES_step1', 'ecfp')
    data = ensure_column_2d(data, 'ecfp')
    #train_data, test_data = split_data(data, train_percent=0.8)
    #gpr_pred(train_data, test_data, 'ecfp', 'Red Pot ( Fc/Fc+)', 'Red Pot ( Fc/Fc+)')
    
    pred_data = pd.read_excel(TO_PRED)
    pred_data = prepare_data_for_gpr(pred_data, column_to_drop_nan = 'SMILES')
    pred_data = transfer_smiles_to_descriptor(pred_data, 'SMILES', 'ecfp')
    pred_data = ensure_column_2d(pred_data, 'ecfp')
    
    gpr_pred(data, pred_data, 'ecfp', 'Red Pot ( Fc/Fc+)', 'DFT-predict(Fc/Fc+)')

main()

