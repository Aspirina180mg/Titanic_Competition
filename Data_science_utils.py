import subprocess
import importlib
import gzip
import json
import ast
import pandas as pd


def install(packages):
    """
    Installs Python packages using pip and summarizes the installation status.

    Args:
        packages (list of str): List of package names to install.
    """
    installed, already_installed, failed, errors = [], [], [], []

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
    """Helper function to print installation summary."""
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
    aliases = {"numpy": "np", "pandas": "pd"}

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
        DataFrame or None if extraction fails.
    """
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            print("CSV data extraction successful.")
        elif file_path.endswith(".json.gz"):
            with gzip.open(file_path, "rb") as file:
                data = [json.loads(row) for row in file]
            df = pd.DataFrame(data)
            print("Gzip JSON data extraction successful.")
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path, lines=True)
            print("JSON data extraction successful.")
        elif file_path.endswith(".ast.gz"):
            with gzip.open(file_path, "rb") as file:
                data = [ast.literal_eval(row.decode("utf-8")) for row in file]
            df = pd.DataFrame(data)
            print("AST Gzip data extraction successful.")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            print("Parquet data extraction successful.")
        else:
            print(f"Unsupported file format: {file_path}")
            return None

        return df

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None


def df_info(df, df_name="Unnamed_DataFrame"):
    """
    Prints useful information about the DataFrame, such as duplicates, missing values, and summary statistics.

    Args:
        df (DataFrame): The DataFrame to analyze.
        df_name (str): Optional name of the DataFrame for display.
    """
    line_length = max(64, len(df_name) + 10)
    print(
        f"\n---Info-{df_name.replace(' ', '-')}{'-' * (line_length - len(df_name) - 10)}"
    )
    print(f"---Duplicated values: {df.duplicated().sum()}")
    print(f"---Fully empty rows: {df.isnull().all(axis=1).sum()}")
    print(f"\n---Dataframe head:\n{df.head(3).to_string()}")
    print("\n---Dataframe info:\n")
    df.info()
    print(f"\n----Dataframe description:\n{df.describe().to_string()}")
    print(f"\n---Missing values:\n{df.isna().sum()}")


def df_clean(df, df_name="Unnamed_DataFrame"):
    """
    Cleans the DataFrame by removing duplicates, fully empty rows, and resetting the index.

    Args:
        df (DataFrame): The DataFrame to clean.
        df_name (str): Optional name of the DataFrame for display.
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


def column_unique(df):
    """
    Prints unique values in each column of the DataFrame.

    Args:
        df (DataFrame): The DataFrame to analyze.
    """
    for column in df.columns:
        unique_values = df[column].unique()
        unique_counted = len(unique_values)

        if unique_counted == len(df[column]):
            print(f"Column: {column} - All values are different")
        else:
            sorted_values = sorted(unique_values, key=str)
            print(
                f"Column: {column} - Unique values: {', '.join(map(str, sorted_values))}"
            )
        print("-" * 50)


def unique_count(df, column_name):  # todo modify docstring
    """
    Prints the count and percentage of unique values in a specified column.

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

        # Check if the number of unique values exceeds 10
        if len(value_counts) > 10:
            print(
                f"\nUnique values in '{column_name}' (total {total_count}) - Showing only the first 10 values:"
            )
        else:
            print(f"\nUnique values in '{column_name}' (total {total_count}):")

        for value, count in top_10_values.items():
            percentage = (count / total_count) * 100
            print(f"{value}: {count} times ({percentage:.2f}%)")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def unique_count_all(df, column_name):  # todo modify docsrting
    """
    Prints the count and percentage of unique values in a specified column.

    Args:
        df (DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.
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


def unique_count_list(df, column_name):
    """
    Prints the total number of unique values and an alphabetical list of all
    unique values in a specified column.

    Args:
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
