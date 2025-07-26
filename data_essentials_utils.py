# Module-level docstring
"""
data_essentials_utils.py

This utility module provides various functions to assist with common tasks in 
Data Science projects. The functions include package installation, package 
importing with aliases where available, data extraction from multiple formats, 
and several DataFrame analysis and cleaning methods.

Functions:
    install(packages):
        Installs a list of Python packages and summarizes the installation status.

    import_packages(packages):
        Imports a list of Python packages with predefined aliases where available.

    extract(file_path):
        Extracts data from various file formats (CSV, JSON, AST Gzip, Parquet) 
        into a DataFrame.

    df_info(df, df_name="Unnamed_DataFrame"):
        Prints useful information about a DataFrame, including duplicates, missing 
        values, and summary statistics.

    df_clean(df, df_name="Unnamed_DataFrame"):
        Cleans a DataFrame by removing duplicates, fully empty rows, and resetting 
        the index.

    column_unique(df):
        Prints unique values for each column of the DataFrame.

    unique_count(df, column_name):
        Prints the count and percentage of unique values in a specified column.

    unique_count_all(df, column_name):
        Prints the count and percentage of all unique values in a specified column.

    unique_count_list(df, column_name):
        Prints the total number of unique values and an alphabetical list of all 
        unique values in a specified column.

    univariate(df, column):
        Performs univariate analysis on a DataFrame column, including descriptive 
        statistics, skewness, kurtosis, and visualizations.
"""
__all__ = ['install_packages', 'import_packages', 'extract', 'info', 'clean',
           'unique_count_all_columns', 'unique_count_top_10', 'unique_count',
           'unique_count_sorted', 'fill_with_mean', 'univariate', 'bivariate',
           'multivariate']

# Imports
import subprocess
import importlib
import gzip
import json
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def install_packages(packages):
    """
    Installs Python packages using pip and summarizes the installation status.

    Args:
        packages (list of str): List of package names to install.
    """
    installed, already_installed, failed, errors = [], [], [], []

    # Sort the package list to ensure consistent installation order
    packages.sort()

    for package in packages:
        try:
            result = subprocess.run(
                ["pip", "install", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            if "Requirement already satisfied" in result.stdout:
                already_installed.append(package)
            elif "Successfully installed" in result.stdout:
                installed.append(package)
            else:
                failed.append(package)
        except subprocess.CalledProcessError as e:
            errors.append(f"Error installing {package}: {e}")
        except Exception as e:
            errors.append(f"Unexpected error installing {package}: {e}")

    _print_installation_summary(installed, already_installed, failed, errors)


def _print_installation_summary(installed, already_installed, failed, errors):
    """
    Helper function that prints a summary of the installation process,
    categorizing items as installed, already installed, failed, or with errors.

    Parameters:
        installed (list): List of items successfully installed.
        already_installed (list): List of items that were already installed.
        failed (list): List of items that failed to install.
        errors (list): List of error messages related to the installation.
    """
    if installed:
        print(f"Installed: {', '.join(installed)}")

    if already_installed:
        print(f"Already installed: {', '.join(already_installed)}")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    if errors:
        print("\n".join(errors))

    print("")  # Add spacing for better readability


def import_packages(packages):
    """
    Imports a list of Python packages with aliases where defined.

    Args:
        packages (list of str): List of package names to import.

    """
    aliases = {
        "numpy": "np",
        "pandas": "pd",
        "seaborn": "sns",
        "matplotlib.pyplot": "plt",
        "sklearn": "sklearn",
        "tensorflow": "tf",
        "torch": "torch",
        "keras": "keras",
        "statsmodels.api": "sm",
        "plotly.express": "px",
        "plotly.graph_objects": "go",
        "scipy": "sp",
        "xgboost": "xgb",
        "lightgbm": "lgb",
        "nltk": "nltk",
        "spacy": "spacy",
        "requests": "req",
        "beautifulsoup4": "bs4",
        "pytorch_lightning": "pl",
        "dask": "dask",
        "joblib": "joblib",
        "tqdm": "tqdm",
        "pyarrow": "pa",
    }

    for package in packages:
        try:
            if package in aliases:
                globals()[aliases[package]] = importlib.import_module(package)
                print(f"{package} imported as {aliases[package]}")
            else:
                globals()[package] = importlib.import_module(package)
                print(f"{package} imported")
        except ImportError:
            print(f"Error importing {package}: Is it installed?")


def extract(file_path):
    """
    Extracts data from various file formats into a DataFrame.

    Supported formats:
        - CSV
        - JSON (gzip compressed or plain)
        - AST Gzip
        - Parquet

    Args:
        file_path (str): The path to the file.

    Returns:
        pd.DataFrame or None: DataFrame if extraction is successful, None if it fails.
    """
    try:
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            print(f"CSV {file_name} data extraction successful.")
        elif file_path.endswith(".json.gz"):
            with gzip.open(file_path, "rb") as file:
                data = [json.loads(row) for row in file]
            df = pd.DataFrame(data)
            print(f"Gzip JSON {file_name} data extraction successful.")
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path, lines=True)
            print(f"JSON {file_name} data extraction successful.")
        elif file_path.endswith(".ast.gz"):
            with gzip.open(file_path, "rb") as file:
                data = [ast.literal_eval(row.decode("utf-8")) for row in file]
            df = pd.DataFrame(data)
            print(f"AST Gzip {file_name} data extraction successful.")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            print(f"Parquet {file_name} data extraction successful.")
        else:
            print(f"Unsupported file format: {file_path}")
            return None

        return df

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None


def info(df, df_name="Unnamed_DataFrame"):
    """
    Prints useful information about the DataFrame, such as duplicates,
    missing values, and summary statistics.

    Args:
        df (DataFrame): The DataFrame to analyze.
        df_name (str): Optional name of the DataFrame for display.
    """
    line_length = max(64, len(df_name) + 10)

    # Print header with proper formatting
    print(
        f"\n---Info-{df_name.replace(' ', '-')}{'-' * (line_length - len(df_name) - 10)}"
    )

    # Print number of duplicated values
    print(f"---Duplicated values: {df.duplicated().sum()}")

    # Print number of fully empty rows
    print(f"---Fully empty rows: {df.isnull().all(axis=1).sum()}")

    # Display dataframe info
    print("\n---Dataframe info:\n")
    df.info()

    # Print missing values per column
    print(f"\n---Missing values:\n{df.isna().sum()}")

    # Display the first 3 rows of the dataframe
    print(f"\n---Dataframe head:\n{df.head(3).to_string()}")

    # Display dataframe summary statistics
    print(f"\n----Dataframe description:\n{df.describe().to_string()}")


def clean(df, df_name="Unnamed_DataFrame"):
    """
    Cleans the DataFrame by removing duplicates, fully empty rows, and resetting the index.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        df_name (str, optional): Name of the DataFrame for display. Defaults to "Unnamed_DataFrame".
    """
    line_length = max(60, len(df_name) + 10)
    print(
        f"\n---Cleaning-{df_name.replace(' ', '-')}{'-' * (line_length - len(df_name) - 10)}"
    )

    # Remove duplicates
    duplicates_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"{duplicates_before} duplicate rows removed.")

    # Remove missing values
    missing_before = df.isnull().sum().sum()
    df.dropna(how="all", inplace=True)
    print(f"{missing_before} missing values removed.")

    # Reset the index
    df.reset_index(drop=True, inplace=True)
    print("Index reset.")


def unique_count_all_columns(df):
    """
    Prints the unique values in each column of the DataFrame.

    Args:
        df (DataFrame): The DataFrame to analyze.
    """
    for column in df.columns:
        unique_values = df[column].unique()
        unique_counted = len(unique_values)

        if unique_counted == len(df[column]):
            print(f"Column: {column} - All values are different")
        else:
            # Separate numerical and non-numerical values
            numerical_values = [
                value for value in unique_values if isinstance(value, (int, float))
            ]
            non_numerical_values = [
                value for value in unique_values if not isinstance(value, (int, float))
            ]

            # Sort numerical values
            numerical_values.sort()

            # Sort non-numerical values
            non_numerical_values.sort()

            # Combine sorted numerical and non-numerical values
            sorted_values = numerical_values + non_numerical_values

            print(
                f"Column: {column} - Unique values: {', '.join(map(str, sorted_values))}"
            )
        print("-" * 50)


def unique_count_top_10(df, column_name):
    """
    Prints the count and percentage of unique values in a specified column.
    If there are more than 10 unique values, it shows only the top 10 and their cumulative percentage.

    Args:
        df (DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.
    """
    if column_name in df.columns:
        # Count only non-null values for the total
        non_null_count = df[column_name].notnull().sum()
        value_counts = df[column_name].value_counts()
        total_count = non_null_count  # Use non-null count for the total

        # Get the top 10 most frequent values
        top_10_values = value_counts.head(10)

        # Calculate the cumulative percentage of the top 10 values
        cumulative_percentage = (top_10_values.sum() / total_count) * 100

        # Check if the number of unique values exceeds 10
        if len(value_counts) > 10:
            print(
                f"\nUnique values in '{column_name}' (total {total_count}) - Showing only the first 10 values:"
            )
            print(
                f"Cumulative percentage of top 10 values: {cumulative_percentage:.2f}%"
            )
        else:
            print(f"\nUnique values in '{column_name}' (total {total_count}):")

        for value, count in top_10_values.items():
            percentage = (count / total_count) * 100
            print(f"{value}: {count} times ({percentage:.2f}%)")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def unique_count(df, column_name):
    """
    Prints the count and percentage of unique values in a specified column.

    Args:
        df (DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.

    Prints:
        The count and percentage of each unique value in the specified column.
    """
    if column_name in df.columns:
        # Count only non-null values for the total
        non_null_count = df[column_name].notnull().sum()

        value_counts = df[column_name].value_counts()
        total_count = non_null_count  # Use non-null count for the total

        # Sort by count in descending order (most frequent first)
        sorted_values = value_counts.items()

        print(f"\nUnique values in '{column_name}' (total {total_count}):")
        for value, count in sorted_values:
            percentage = (count / total_count) * 100
            print(f"{value}: {count} times ({percentage:.2f}%)")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def unique_count_sorted(df, column_name):
    """
    Prints the total number of unique values and an alphabetical list of all
    unique values in a specified column.

    Parameters:
        df (DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.
    """
    if column_name in df.columns:
        # Get unique values, excluding NaNs
        unique_values = df[column_name].dropna().unique()

        # Check if the column is numeric or not
        if pd.api.types.is_numeric_dtype(df[column_name]):
            # Sort the unique values numerically
            sorted_unique_values = sorted(unique_values)
        else:
            # Sort alphabetically if it's not numeric
            sorted_unique_values = sorted(unique_values, key=str)

        # Print the total number of unique values and the list
        total_unique = len(sorted_unique_values)
        print(f"\nTotal unique values in '{column_name}': {total_unique}")
        print("Unique values in order:")
        for value in sorted_unique_values:
            print(value)
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def fill_with_mean(df, column_name, decimals=None):
    """
    Fills missing values in the specified column with the mean value of that column in place.
    Also stores a binary flag indicating if the column had missing values and was filled.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: str, the name of the column to fill
    - decimals: int, optional, the number of decimal places to round the mean value. Default is None (no rounding).
    
    Returns:
    - str: confirmation message including the mean value used and a statement about whether the column was filled,
      along with the binary flag column name.
    """
    # Create the binary flag column before filling missing values
    flag_column_name = f'{column_name}_missing'
    df[flag_column_name] = df[column_name].isnull().astype(int)
    
    # Check if there are any missing values in the column
    had_missing_values = df[column_name].isnull().sum() > 0
    
    # Calculate the mean value of the column
    mean_value = df[column_name].mean()
    
    if decimals is not None:
        mean_value = f"{mean_value:.{decimals}f}"
    
    # Fill missing values with the mean value
    df[column_name] = df[column_name].fillna(float(mean_value))
    
    # Return a message with the relevant details
    if had_missing_values:
        return f"Column '{column_name}' has been filled with the mean value: {mean_value}. " \
               f"Column: '{flag_column_name}' has been created."
    else:
        return f"Column '{column_name}' did not have missing values, no filling necessary."



def univariate(df, column):
    """
    Perform univariate analysis of a column in a DataFrame.

    Automatically detects if the variable is numeric or categorical.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to analyze.
    """
    print(f"\n=== Univariate Analysis for '{column}' ===")

    # Identify the variable type
    if pd.api.types.is_numeric_dtype(df[column]):
        print("Type: Numeric")

        # Descriptive statistics
        stats = df[column].describe()
        print("\nDescriptive Statistics:")
        print(stats)

        # Skewness and kurtosis
        skewness = df[column].skew()
        kurtosis = df[column].kurt()
        print(f"\nSkewness: {skewness:.2f}")
        print(f"Kurtosis: {kurtosis:.2f}")

        # Visualization
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True, bins=30, color="blue")
        plt.title(f"Histogram of {column}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column], color="orange")
        plt.title(f"Boxplot of {column}")

        plt.tight_layout()
        plt.show()

    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == "object":
        print("Type: Categorical")

        # Frequencies
        freq = df[column].value_counts()
        rel_freq = df[column].value_counts(normalize=True) * 100

        print("\nAbsolute Frequencies:")
        print(freq)
        print("\nRelative Frequencies (%):")
        print(rel_freq)

        # Visualization
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, y=column, palette="viridis", order=freq.index)
        plt.title(f"Count of {column}")
        plt.xlabel("Frequency")
        plt.ylabel(column)
        plt.show()

    else:
        print("Unrecognized type or empty column.")

def bivariate():
    pass

def multivariate():
    pass

globals().update({k: v for k, v in locals().items() if callable(v)})
# This line ensures that all functions are accessible at the module level
# and can be imported directly without needing to reference the module name.