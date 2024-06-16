# Catalog-Intelligence-ML

Welcome to the **Catalog-Intelligence-ML**! This repository contains a comprehensive suite of Python scripts and Jupyter Notebooks designed to demonstrate core concepts of **Machine Learning**, **Database Management (SQL)**, **Data Engineering**, **Data Processing Models**, and **Data and Systems Integration**. This project also focuses on developing tools for attribute enrichment and price analysis.

## Overview

In the realm of data science, efficiently managing and processing data is crucial. This suite of scripts is a powerful tool designed to handle real-time data integration and processing tasks using various advanced technologies. Here's a glimpse of what this project offers:

### Key Features

1. **Machine Learning**:
    - This project prepares data, ensuring it is clean and well-structured, making it ready for machine learning models.
    - Implements similarity algorithms to match and prioritize data.
    - Utilizes OpenAI's GPT-4 to generate human-like product descriptions.
        - Processes product names in batches to efficiently generate descriptions.
    - Focuses on developing tools for attribute enrichment and price analysis.

2. **Database Management (SQL)**:
    - Utilizes Snowflake, a cloud-based SQL database, for robust data management.
    - Executes complex queries to extract and manipulate data, ensuring seamless data operations.

3. **Data Engineering**:
    - Performs essential data engineering tasks, including data extraction, transformation, and loading (ETL).
    - Ensures real-time data updates and integration, crucial for maintaining up-to-date datasets for accurate analysis.

4. **Data Processing Models**:
    - Applies sophisticated models for data processing and transformation, ensuring high-quality data for analysis or machine learning applications.
    - Uses TF-IDF vectorization and cosine similarity for data matching.

5. **Data and Systems Integration**:
    - Demonstrates seamless integration of data from various sources into a centralized system.
    - Facilitates continuous data flow and updates, essential for real-time analytics and decision-making processes.
    - Exports processed data to Google Sheets for further use and integration.

## How It Works

The scripts perform the following tasks:

### Main Script (`add_mps_v3.py`)

1. **Database Connection**:
    - Establishes a connection to a Snowflake database using secure credentials.
    
2. **Data Retrieval**:
    - Extracts data from Snowflake tables based on specific criteria and thresholds defined for various countries.

3. **Data Processing and Prioritization**:
    - Processes the extracted data to calculate similarities, GMV coverage, and other metrics.
    - Prioritizes products based on calculated metrics, ensuring the most relevant data is available for analysis.

4. **Export and Integration**:
    - Exports the processed data to Google Sheets for further use and integration.
    - Utilizes custom functions for exporting and cleaning the data in Google Sheets.

### Supporting Scripts

1. **Database Extraction (`database_extract.py`)**:
    - Contains functions to run SQL queries and extract data from the Snowflake database.
    - Ensures data is accurately retrieved and prepared for further processing.

2. **Similarity Calculation (`funcition_run_similarity.py`)**:
    - Implements TF-IDF vectorization and cosine similarity calculations to find matches between data sets.
    - Uses efficient algorithms to process large data sets quickly.

3. **Utility Functions (`functions.py`)**:
    - Includes functions for exporting data to Google Sheets and cleaning up sheets.
    - Manages Google Sheets API authentication and operations.

## Technologies Used

- **Python**: The primary programming language used for scripting.
- **Pandas**: Utilized for data manipulation and analysis.
- **NumPy**: Used for numerical operations.
- **Snowflake Connector**: Facilitates connection and operations with the Snowflake database.
- **Google Sheets API**: Used for exporting and updating data in Google Sheets.
- **Scikit-learn**: Provides tools for TF-IDF vectorization and cosine similarity calculations.
- **Sparse Matrix Operations**: For efficient similarity calculations.
- **OpenAI API**: Used for generating product descriptions with GPT-4.

## Getting Started

To run these scripts, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install pandas numpy snowflake-connector-python google-api-python-client scikit-learn openai