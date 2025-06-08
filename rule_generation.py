import uuid # For generating UUIDs for rule_id
import os
from datetime import datetime, timedelta
import time # For retry logic sleep
from google.api_core.exceptions import GoogleAPIError, NotFound # For catching specific BigQuery API errors
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for parallelism

# --- Google Cloud Client Imports ---
from google.cloud import bigquery
from google.cloud import aiplatform
from google import genai
from google.genai import types
from vertexai.generative_models import GenerativeModel

# --- Other Standard Library Imports (from your environment setup) ---
import pandas as pd
import sqlite3 # Included as per your environment, though not used for BQ operations here
import base64 # Included as per your environment, though not directly called in main logic
from IPython.display import Markdown, display # Included as per your environment, though not directly called in main logic


# --- Configuration ---
# Google Cloud Project ID - uses environment variable or defaults to 'cloud-professional-services'
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "cloud-professional-services")
# Google Cloud Region - uses environment variable or defaults to 'us-central1'
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

# The BigQuery dataset where your *control table* and *results table* are located
CONTROL_DATASET_ID = "sprint"

# The full ID of your control table (project.dataset.table)
# RENAMED: testing_table to control_table
CONTROL_TABLE_FULL_ID = f"{PROJECT_ID}.{CONTROL_DATASET_ID}.control_table"

# The full ID of your results table (project.dataset.table)
# RENAMED: testing_results to results_table
RESULTS_TABLE_FULL_ID = f"{PROJECT_ID}.{CONTROL_DATASET_ID}.summary_table"

# The full ID of the new failed rows sample table
FAILED_ROWS_SAMPLE_TABLE_FULL_ID = f"{PROJECT_ID}.{CONTROL_DATASET_ID}.sample_table"

# --- NEW: Define Referential Integrity Rules Here ---
# Add tuples in the format: (referencing_table_id, referencing_column_name, referenced_table_id, referenced_column_name)
# These table names should be just the table_id, not the full project.dataset.table path.
# Example: If your 'orders' table has a 'customer_id' column that refers to the 'id' column in your 'customers' table.
REFERENTIAL_INTEGRITY_RULES = [
    # ('orders', 'customer_id', 'customers', 'id'),
    # ('order_items', 'product_id', 'products', 'product_id'),
    # Add more rules as needed based on your source_data relationships
]


# --- Initialize BigQuery client ---
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    print(f"BigQuery client initialized for project: {bq_client.project}")
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    # It's critical to have a BQ client, so exit if it fails
    exit("Exiting: BigQuery client initialization failed.")

# --- Initialize Vertex AI client ---
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI client initialized for project: {PROJECT_ID}, location: {LOCATION}")
except Exception as e:
    print(f"Error initializing Vertex AI client: {e}")
    print("Warning: Vertex AI client initialization failed. AI-driven rules may not function.")


# --- AI Prompt Generation and Content Generation Functions ---

def generate_string_pattern_prompt(project_id, dataset_name, table_name, column_name, sample_size=100):
    """
    Generates a prompt for Gemini to create a SQL query based on inconsistent string patterns.
    It fetches a sample of distinct non-null values from the specified column.
    """
    try:
        # Query to get distinct sample values from the specified column
        # Using backticks for table and column names for robustness
        query = f"""
            SELECT DISTINCT `{column_name}`
            FROM `{project_id}.{dataset_name}.{table_name}`
            WHERE `{column_name}` IS NOT NULL
            LIMIT {sample_size}
        """

        # Runs the query to get samples
        query_job = bq_client.query(query)
        results = query_job.result()
        sample_values = [str(row[column_name]) for row in results] # Convert to string to handle various types safely

        if not sample_values:
            return f"No non-null string samples found in '{table_name}.{column_name}'. Cannot generate pattern prompt."

        # Constructing the prompt for Gemini
        prompt = f"""
        Based on these sample string values from the column '{column_name}' in the table '{table_name}':
        {', '.join(sample_values)}

        Analyze these sample values and identify a common pattern or format if one exists.
        Then, write a SQL query using `REGEXP_CONTAINS()` that retrieves all rows from the table
        '{table_name}' where the value in the column '{column_name}' does NOT match this identified pattern.
        If no clear pattern is discernible, generate a query that looks for potential inconsistencies
        (e.g., variations in case, unexpected characters, common data entry errors, or unexpected length).
        Return ONLY the SQL query.
        """
        return prompt

    except Exception as e:
        return f"Error generating prompt for {table_name}.{column_name}: {e}"

def generate_content(prompt):
    """
    Generates content (SQL query or description) using the Gemini model based on the provided prompt.
    Includes robust error handling and markdown stripping.
    """
    try:
        # Ensure genai client is initialized. This might be redundant if aiplatform.init works,
        # but good for explicit clarity.
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        model = "gemini-2.0-flash-001" # Or gemini-1.5-flash-001 if preferred/available
        contents = [
          types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
          )
        ]

        # System instructions to give model context and control output format
        system_instruction = """You are an expert in analyzing tabular data and generating SQL queries to identify data anomalies, specifically for string format validation. Your task is to examine the provided sample string values and generate a SQL query using REGEXP_CONTAINS to find values that deviate from a common pattern or exhibit inconsistencies. When asked to describe a rule, provide a concise, single-sentence description in simple English. Ensure column names in the description are formatted with spaces instead of underscores (e.g., 'region code' instead of 'region_code'). Provide ONLY the SQL query or the description, without any surrounding markdown or extra text."""

        generate_content_config = types.GenerateContentConfig(
            temperature=0.0, # Lower temperature for more deterministic output
            top_p=0.8,
            max_output_tokens=1024,
            safety_settings=[ # Disable safety settings for more direct output
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        if response.candidates and response.candidates[0].content.parts:
            raw_text = response.candidates[0].content.parts[0].text.strip()
            # Remove markdown code block delimiters if present (e.g., ```sql ... ``` or ``` ... ```)
            if raw_text.startswith("```sql") and raw_text.endswith("```"):
                raw_text = raw_text[len("```sql"):-len("```")].strip()
            elif raw_text.startswith("```") and raw_text.endswith("```"):
                raw_text = raw_text[len("```"):-len("```")].strip()
            return raw_text
        else:
            return "No content generated."

    except Exception as e:
        return f"Error generating content: {e}"


# --- Helper Function to Insert Rules into control_table ---
def insert_rule(rule_data: dict, max_retries=5, initial_delay=1):
    """
    Inserts a single rule definition into the BigQuery control table with retry logic.
    """
    for attempt in range(max_retries):
        try:
            # Re-fetch table in case of transient issues or recent recreation
            table = bq_client.get_table(CONTROL_TABLE_FULL_ID)

            # Prepare the row to insert. The order of values must match the table schema.
            rows_to_insert = [
                (
                    rule_data['rule_id'],
                    rule_data['source_project_id'],
                    rule_data['source_dataset_id'],
                    rule_data['source_table_id'],
                    rule_data['metric_column'],
                    rule_data['rule_generation_timestamp'],
                    rule_data['rule_sql'],
                    rule_data['rule_family'],
                    rule_data['rule_description']
                )
            ]

            errors = bq_client.insert_rows(table, rows_to_insert)

            if errors:
                print(f"Attempt {attempt + 1}/{max_retries}: Encountered errors while inserting rule with ID '{rule_data['rule_id']}': {errors}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt)) # Exponential backoff
                    print(f"Retrying insertion for rule with ID '{rule_data['rule_id']}'...")
                else:
                    print(f"Failed to insert rule with ID '{rule_data['rule_id']}' after {max_retries} attempts.")
            else:
                print(f"Successfully inserted rule with ID: {rule_data['rule_id']}")
                return True # Success, return True
        except GoogleAPIError as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Google API Error during insertion for rule with ID '{rule_data['rule_id']}': {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))
                print(f"Retrying insertion for rule with ID '{rule_data['rule_id']}'...")
            else:
                print(f"Failed to insert rule with ID '{rule_data['rule_id']}' after {max_retries} attempts due to API error.")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Unexpected Error during insertion for rule with ID '{rule_data['rule_id']}': {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))
                print(f"Retrying insertion for rule with ID '{rule_data['rule_id']}'...")
            else:
                print(f"Failed to insert rule with ID '{rule_data['rule_id']}' after {max_retries} attempts due to unexpected error.")
    print(f"Rule with ID '{rule_data['rule_id']}' was not inserted after {max_retries} attempts.")
    return False


# --- New Helper Function: Ensure BigQuery Dataset Exists ---
def ensure_dataset_exists(dataset_id: str, project_id: str, location: str, max_retries=5, initial_delay=1):
    """
    Ensures that a BigQuery dataset exists. Creates it if it doesn't.
    """
    full_dataset_id = f"{project_id}.{dataset_id}"
    print(f"Ensuring BigQuery dataset '{full_dataset_id}' exists...")
    for attempt in range(max_retries):
        try:
            bq_client.get_dataset(dataset_id)
            print(f"Dataset '{full_dataset_id}' found.")
            return True
        except NotFound:
            print(f"Dataset '{full_dataset_id}' not found. Attempting to create...")
            dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
            dataset.location = location
            try:
                bq_client.create_dataset(dataset, timeout=30)
                print(f"Successfully created dataset '{full_dataset_id}'.")
                return True
            except GoogleAPIError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Error creating dataset '{full_dataset_id}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                else:
                    print(f"Failed to create dataset '{full_dataset_id}' after {max_retries} attempts.")
                    return False
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Unexpected error creating dataset '{full_dataset_id}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                else:
                    print(f"Failed to create dataset '{full_dataset_id}' after {max_retries} attempts due to unexpected error.")
                return False
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Error checking for dataset '{full_dataset_id}': {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))
            else:
                print(f"Failed to ensure dataset '{full_dataset_id}' exists after {max_retries} attempts.")
                return False
    return False # Should not reach here


def create_results_table():
    """
    Checks for the existence of the results table and creates it or replaces it if found.
    Returns True if successful, False otherwise.
    """
    print(f"Checking for results table: '{RESULTS_TABLE_FULL_ID}'...")
    try:
        # Using get_table to check existence, but the SQL will handle replace logic
        bq_client.get_table(RESULTS_TABLE_FULL_ID)
        print(f"Results table '{RESULTS_TABLE_FULL_ID}' exists. It will be replaced as requested.")
    except NotFound:
        print(f"Results table '{RESULTS_TABLE_FULL_ID}' not found. Attempting to create it.")
    except Exception as e:
        print(f"An unexpected error occurred while checking results table existence: {e}")
        # Even on error, we attempt CREATE OR REPLACE, as it might resolve some issues.

    create_table_sql = f"""
    CREATE OR REPLACE TABLE `{RESULTS_TABLE_FULL_ID}` (
        result_id STRING NOT NULL OPTIONS(description="Unique identifier for each rule execution result (UUID)"),
        rule_id STRING NOT NULL OPTIONS(description="ID of the data quality rule that was executed"),
        source_dataset_id STRING NOT NULL OPTIONS(description="BigQuery dataset ID of the source data checked"),
        source_table_id STRING NOT NULL OPTIONS(description="Name of the source table checked"),
        metric_column STRING NOT NULL OPTIONS(description="The column name to which the data quality rule was applied"),
        rule_family STRING NOT NULL OPTIONS(description="Category of the executed rule"),
        execution_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when this rule execution completed"),
        execution_duration_seconds FLOAT64 OPTIONS(description="Duration of the rule execution in seconds"),
        status STRING NOT NULL OPTIONS(description="Outcome of the execution (PASS, FAIL, ERROR)"),
        failed_rows INT64 OPTIONS(description="Number of rows that failed the data quality check"),
        error_message STRING OPTIONS(description="Error message if execution failed")
    )
    PARTITION BY
        DATE(execution_timestamp)
    OPTIONS(
        description="Table to store results and metadata of data quality rule executions."
    );
    """
    try:
        query_job = bq_client.query(create_table_sql)
        query_job.result()  # Wait for the table creation/replacement to complete
        print(f"Successfully created or replaced results table: '{RESULTS_TABLE_FULL_ID}'.")
        return True
    except Exception as e:
        print(f"Error creating or replacing results table '{RESULTS_TABLE_FULL_ID}': {e}")
        return False

def create_failed_rows_sample_table():
    """
    Checks for the existence of the failed rows sample table and creates it or replaces it if found.
    If it exists, it does nothing, preserving existing data.
    Returns True if successful, False otherwise.
    """
    print(f"Checking for failed rows sample table: '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}'...")
    try:
        # Try to get the table. If it exists, we're good.
        bq_client.get_table(FAILED_ROWS_SAMPLE_TABLE_FULL_ID)
        print(f"Failed rows sample table '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}' already exists. Preserving existing data.")
        return True
    except NotFound:
        print(f"Failed rows sample table '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}' not found. Attempting to create it.")
        create_table_sql = f"""
        CREATE TABLE `{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}` (
            failed_row_entry_id STRING NOT NULL OPTIONS(description="Unique identifier for each failed row entry (UUID)"),
            rule_id STRING NOT NULL OPTIONS(description="ID of the data quality rule that this failed row belongs to"),
            source_id STRING OPTIONS(description="The unique ID of the failed row from the source table's first column"),
            source_project_id STRING NOT NULL OPTIONS(description="BigQuery project ID of the source data"),
            source_dataset_id STRING NOT NULL OPTIONS(description="BigQuery dataset ID of the source data"),
            source_table_id STRING NOT NULL OPTIONS(description="Name of the source table where the row failed"),
            execution_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when the rule execution that identified this failed row completed"),
            metric_column STRING NOT NULL OPTIONS(description="The column name to which the data quality rule was applied")
        )
        PARTITION BY
            DATE(execution_timestamp)
        OPTIONS(
            description="Table to store sample of actual rows that failed data quality checks, including source_id."
        );
        """
        try:
            query_job = bq_client.query(create_table_sql)
            query_job.result()  # Wait for the table creation to complete
            print(f"Successfully created failed rows sample table: '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}'.")
            return True
        except Exception as e:
            print(f"Error creating failed rows sample table '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}': {e}")
            return False
    except Exception as e:
        print(f"An unexpected error occurred while checking failed rows sample table existence: {e}")
        return False

# --- Worker function for processing a single column/rule type (now without ARIMA) ---
def process_column_for_rules(target_dataset_id, table_name, col_row, columns_df):
    """
    Processes a single column to generate and insert various types of data quality rules,
    excluding ARIMA anomaly detection which is now handled at the table level.
    This function will be run in parallel.
    """
    col_name = col_row['column_name']
    data_type = col_row['data_type']
    is_nullable = col_row['is_nullable'] == 'YES'
    full_table_id_quoted = f"`{PROJECT_ID}.{target_dataset_id}.{table_name}`"
    rule_insertions = [] # Collect all rule data to be inserted

    # --- Null Check Rule ---
    rule_family = "Null Check"
    cleaned_col_name = col_name.replace('_', ' ')
    if not is_nullable:
        rule_sql = f"SELECT COUNT(*) AS failed_count FROM {full_table_id_quoted} WHERE `{col_name}` IS NULL"
        rule_description = f"Checks for unexpected NULL values in the NOT NULL column '{cleaned_col_name}' in table '{table_name}'."
    else:
        rule_sql = f"SELECT COUNT(*) AS failed_count FROM {full_table_id_quoted} WHERE `{col_name}` IS NULL"
        rule_description = f"Counts NULL values in the nullable column '{cleaned_col_name}' in table '{table_name}'."
    rule_insertions.append({
        'rule_id': str(uuid.uuid4()),
        'source_project_id': PROJECT_ID,
        'source_dataset_id': target_dataset_id,
        'source_table_id': table_name,
        'metric_column': col_name,
        'rule_generation_timestamp': datetime.now().isoformat(),
        'rule_sql': " ".join(rule_sql.split()).strip(),
        'rule_family': rule_family,
        'rule_description': rule_description
    })

    # --- Uniqueness Check (only for the first column, assuming PK) ---
    # This check should ideally only run for the primary key.
    # For parallelism, it's safer to check if this column is indeed the first column (assumed PK).
    first_column_name = columns_df.iloc[0]['column_name'] if not columns_df.empty else None
    if col_name == first_column_name:
        print(f"Generating uniqueness check rule for {table_name}.{col_name} (assumed PK)...")
        uniqueness_sql = f"""
            SELECT COUNT(*) AS failed_count
            FROM (
                SELECT `{col_name}`
                FROM {full_table_id_quoted}
                WHERE `{col_name}` IS NOT NULL
                GROUP BY `{col_name}`
                HAVING COUNT(*) > 1
            )
        """
        cleaned_first_col_name = col_name.replace('_', ' ')
        uniqueness_description = f"Checks for duplicate values in the primary key column '{cleaned_first_col_name}' of table '{table_name}'."
        rule_insertions.append({
            'rule_id': str(uuid.uuid4()),
            'source_project_id': PROJECT_ID,
            'source_dataset_id': target_dataset_id,
            'source_table_id': table_name,
            'metric_column': col_name,
            'rule_generation_timestamp': datetime.now().isoformat(),
            'rule_sql': " ".join(uniqueness_sql.split()).strip(),
            'rule_family': "Uniqueness",
            'rule_description': uniqueness_description
        })

    # --- String Formatting (AI-generated regex) for STRING columns ---
    if data_type == 'STRING':
        print(f"Generating AI prompt for string pattern in {table_name}.{col_name}...")
        string_pattern_prompt = generate_string_pattern_prompt(
            project_id=PROJECT_ID,
            dataset_name=target_dataset_id,
            table_name=table_name,
            column_name=col_name
        )

        ai_generated_sql = ""
        ai_generated_description = ""
        rule_family_str_format = "String Formatting"

        if "Error generating prompt" in string_pattern_prompt or "No non-null string samples found" in string_pattern_prompt:
            print(f"Skipping AI-driven formatting for {table_name}.{col_name} due to prompt generation issue: {string_pattern_prompt}")
            ai_generated_sql = f"SELECT COUNT(*) AS failed_count FROM {full_table_id_quoted} WHERE `{col_name}` IS NOT NULL AND LENGTH(`{col_name}`) = 0"
            ai_generated_description = f"Checks for empty strings in '{cleaned_col_name}' in table '{table_name}' due to AI pattern generation failure."
        else:
            ai_generated_sql = generate_content(string_pattern_prompt)
            if "No content generated" in ai_generated_sql or "Error generating content" in ai_generated_sql:
                print(f"{table_name}.{col_name}: {ai_generated_sql}. Falling back to generic check.")
                ai_generated_sql = f"SELECT COUNT(*) AS failed_count FROM {full_table_id_quoted} WHERE `{col_name}` IS NOT NULL AND LENGTH(`{col_name}`) = 0"
                ai_generated_description = f"Checks for empty strings in '{cleaned_col_name}' in table '{table_name}' due to AI SQL generation failure."
            else:
                ai_generated_sql = ai_generated_sql.strip()
                ai_generated_sql = " ".join(ai_generated_sql.split())
                ai_generated_sql = ai_generated_sql.rstrip(';')
                ai_generated_sql = ai_generated_sql.replace(f"FROM {table_name}", f"FROM {full_table_id_quoted}")
                ai_generated_sql = ai_generated_sql.replace(f" {col_name}", f" `{col_name}`")
                ai_generated_sql = ai_generated_sql.replace(f"WHERE {col_name}", f"WHERE `{col_name}`")
                ai_generated_sql = ai_generated_sql.replace(f"REGEXP_CONTAINS({col_name},", f"REGEXP_CONTAINS(`{col_name}`,")

                if not ai_generated_sql.lower().startswith("select count(*)"):
                    ai_generated_sql = f"SELECT COUNT(*) AS failed_count FROM ({ai_generated_sql})"

                description_prompt = f"""
                Here's a SQL rule generated using REGEXP_CONTAINS for the column `{col_name}`.
                Please describe in one concise sentence what this rule is checking for in simple English.
                Refer to the column as '{cleaned_col_name}' in the description.
                """
                ai_generated_description = generate_content(description_prompt)
                if "No content generated" in ai_generated_description or "Error generating content" in ai_generated_description:
                    ai_generated_description = f"Checks for formatting inconsistencies in '{cleaned_col_name}' in table '{table_name}' using AI-generated regex."

        rule_insertions.append({
            'rule_id': str(uuid.uuid4()),
            'source_project_id': PROJECT_ID,
            'source_dataset_id': target_dataset_id,
            'source_table_id': table_name,
            'metric_column': col_name,
            'rule_generation_timestamp': datetime.now().isoformat(),
            'rule_sql': ai_generated_sql,
            'rule_family': rule_family_str_format,
            'rule_description': ai_generated_description
        })
    return rule_insertions # Return the list of rules to be inserted

# --- Main Logic to Populate Data Quality Rules ---
def populate_data_quality_rules(target_dataset_id: str):
    """
    Discovers columns in all base tables within the specified BigQuery dataset,
    generates data quality rules (AI-driven String Formatting, Anomaly Detection,
    Null Checks, Uniqueness, and Referential Integrity), and inserts them into
    the control table with UUID rule_ids.

    Args:
        target_dataset_id (str): The ID of the dataset to scan for source data tables.
    """
    print(f"Starting population of data quality rules for dataset: {PROJECT_ID}.{target_dataset_id}")

    # --- Ensure Control Table Schema is Correct (ALWAYS REPLACED) ---
    print(f"Ensuring control table: '{CONTROL_TABLE_FULL_ID}' has the correct schema...")
    create_control_table_sql = f"""
CREATE OR REPLACE TABLE `{CONTROL_TABLE_FULL_ID}` (
    rule_id STRING NOT NULL OPTIONS(description="Unique identifier for each data quality rule (UUID)"),
    source_project_id STRING NOT NULL OPTIONS(description="BigQuery project ID of the source data"),
    source_dataset_id STRING NOT NULL OPTIONS(description="BigQuery dataset ID of the source data"),
    source_table_id STRING NOT NULL OPTIONS(description="Name of the source table being checked"),
    metric_column STRING NOT NULL OPTIONS(description="The column name to which the data quality rule is applied"),
    rule_generation_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when this rule definition was created or last updated"),
    rule_sql STRING NOT NULL OPTIONS(description="The BigQuery SQL query string to perform the data quality check"),
    rule_family STRING NOT NULL OPTIONS(description="Categorization of the rule (e.g., 'String Formatting', 'Anomaly Detection', 'Null Check')"),
    rule_description STRING OPTIONS(description="Detailed human-readable description of the rule's purpose and logic")
)
PARTITION BY
    DATE(rule_generation_timestamp)
OPTIONS(
    description="Control table for defining and managing data quality rules."
);
"""
    try:
        query_job = bq_client.query(create_control_table_sql)
        query_job.result() # Wait for the table creation/replacement to complete
        print(f"Successfully created or replaced control table: '{CONTROL_TABLE_FULL_ID}' with the latest schema.")
        print("Waiting a few seconds for control table to stabilize after creation/replacement...")
        time.sleep(5) # Give BigQuery time to propagate the new table
    except Exception as e:
        print(f"Error creating or replacing control table '{CONTROL_TABLE_FULL_ID}': {e}")
        exit("Exiting: Failed to create or replace control table with correct schema.")


    # Discover Tables in the target dataset
    tables_query = f"""
        SELECT table_name
        FROM `{PROJECT_ID}.{target_dataset_id}.INFORMATION_SCHEMA.TABLES`
        WHERE table_type = 'BASE TABLE'
    """
    tables_df = bq_client.query(tables_query).to_dataframe()

    if tables_df.empty:
        print(f"No base tables found in dataset {PROJECT_ID}.{target_dataset_id}. Exiting rule population.")
        return

    all_rules_to_insert = [] # List to collect all rules generated from all threads

    # Use ThreadPoolExecutor for parallel processing of columns for non-ARIMA rules
    MAX_WORKERS_COLUMN_PROCESSING = 5 # You can increase this, but be mindful of AI API rate limits

    for _, table_row in tables_df.iterrows():
        table_name = table_row['table_name']
        print(f"\n--- Discovering columns for table: {table_name} ---")

        columns_query = f"""
            SELECT column_name, data_type, is_nullable
            FROM `{PROJECT_ID}.{target_dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        try:
            columns_df = bq_client.query(columns_query).to_dataframe()
            if columns_df.empty:
                print(f"No columns found for table {table_name}. Skipping rule generation.")
                continue

            # --- Generate rules for individual columns (Null, Uniqueness, String Formatting) in parallel ---
            column_futures = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_COLUMN_PROCESSING) as executor:
                for _, col_row in columns_df.iterrows():
                    column_futures.append(executor.submit(process_column_for_rules, target_dataset_id, table_name, col_row, columns_df))

                for future in as_completed(column_futures):
                    try:
                        rules = future.result()
                        all_rules_to_insert.extend(rules)
                    except Exception as exc:
                        print(f"Column rule generation task generated an exception: {exc}")

            # --- NEW: Generate ONE ARIMA Model and Anomaly Detection Rule per table ---
            numeric_columns = columns_df[columns_df['data_type'].isin(['INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC'])]
            timestamp_columns = columns_df[columns_df['data_type'].isin(['TIMESTAMP', 'DATETIME', 'DATE'])]

            if not numeric_columns.empty and not timestamp_columns.empty:
                metric_col_for_arima = numeric_columns['column_name'].iloc[0]
                timestamp_col_for_arima = timestamp_columns['column_name'].iloc[0]
                full_table_id_quoted = f"`{PROJECT_ID}.{target_dataset_id}.{table_name}`"

                # Heuristic to pick a more suitable time-series metric if available
                preferred_time_series_metrics = ['sale', 'spend', 'cases', 'deaths', 'total', 'count', 'value', 'amount', 'sum']
                found_preferred_metric = False
                for pref_metric_part in preferred_time_series_metrics:
                    for _, num_col_row in numeric_columns.iterrows():
                        if pref_metric_part in num_col_row['column_name'].lower():
                            metric_col_for_arima = num_col_row['column_name']
                            found_preferred_metric = True
                            print(f"Selected preferred metric for ARIMA in table '{table_name}': '{metric_col_for_arima}'")
                            break
                    if found_preferred_metric:
                        break

                if not found_preferred_metric:
                    print(f"Warning: No preferred time-series metric found for table '{table_name}'. Using '{metric_col_for_arima}' (first numeric column).")

                print(f"Attempting to set up ARIMA for table '{table_name}': metric='{metric_col_for_arima}', timestamp='{timestamp_col_for_arima}'")

                # Define ARIMA model parameters and naming convention
                bqml_model_name = f"dq_arima_{table_name.replace('-', '_')}_{metric_col_for_arima.replace('-', '_')}"
                full_bqml_model_path = f"`{PROJECT_ID}.{CONTROL_DATASET_ID}.{bqml_model_name}`"

                # --- Dynamically determine training date range ---
                training_start_date = None
                training_end_date = None

                try:
                    max_timestamp_query = f"SELECT MAX(`{timestamp_col_for_arima}`) FROM {full_table_id_quoted} WHERE `{timestamp_col_for_arima}` IS NOT NULL"
                    max_timestamp_job = bq_client.query(max_timestamp_query)
                    max_timestamp_result = max_timestamp_job.result().to_dataframe()

                    if not max_timestamp_result.empty and max_timestamp_result.iloc[0, 0] is not None:
                        latest_data_datetime_or_date = max_timestamp_result.iloc[0, 0]
                        if isinstance(latest_data_datetime_or_date, datetime):
                            latest_data_date = latest_data_datetime_or_date.date()
                        elif isinstance(latest_data_datetime_or_date, datetime.date):
                            latest_data_date = latest_data_datetime_or_date
                        else:
                            raise TypeError(f"Unexpected type for timestamp: {type(latest_data_datetime_or_date)}")

                        training_end_date = latest_data_date
                        training_start_date = latest_data_date - timedelta(days=90)
                        print(f"Dynamically determined ARIMA training date range for '{table_name}': {training_start_date} to {training_end_date}")
                    else:
                        print(f"Warning: Could not determine max timestamp for {table_name}.{timestamp_col_for_arima}. This might lead to 'Input data doesn't contain any rows' error if data is sparse.")
                        training_end_date = datetime.now().date() - timedelta(days=7)
                        training_start_date = training_end_date - timedelta(days=90)
                        print(f"ARIMA model training date range (default fallback) for '{table_name}': {training_start_date} to {training_end_date}")
                except Exception as e:
                    print(f"{table_name}.{timestamp_col_for_arima}: {e}. Falling back to default relative date range.")
                    training_end_date = datetime.now().date() - timedelta(days=7)
                    training_start_date = training_end_date - timedelta(days=90)
                    print(f"ARIMA model training date range (default fallback) for '{table_name}': {training_start_date} to {training_end_date}")

                if training_start_date and training_end_date:
                    create_model_sql = f"""
                        CREATE OR REPLACE MODEL {full_bqml_model_path}
                        OPTIONS(
                          MODEL_TYPE='ARIMA_PLUS',
                          TIME_SERIES_TIMESTAMP_COL='{timestamp_col_for_arima}',
                          TIME_SERIES_DATA_COL='{metric_col_for_arima}',
                          AUTO_ARIMA_MAX_ORDER=5,
                          DATA_FREQUENCY='AUTO_FREQUENCY'
                        ) AS
                        SELECT
                          `{timestamp_col_for_arima}`,
                          `{metric_col_for_arima}`
                        FROM {full_table_id_quoted}
                        WHERE DATE(`{timestamp_col_for_arima}`) BETWEEN '{training_start_date}' AND '{training_end_date}'
                        AND `{metric_col_for_arima}` IS NOT NULL
                        ORDER BY `{timestamp_col_for_arima}` ASC
                    """
                    create_model_sql = " ".join(create_model_sql.split()).strip()

                    print(f"Training ARIMA model for {table_name}.{metric_col_for_arima}...")
                    try:
                        train_job = bq_client.query(create_model_sql)
                        train_job.result()
                        print(f"Successfully trained BQML model: {full_bqml_model_path}")

                        anomaly_threshold = 0.99
                        anomaly_detection_sql = f"""
                            WITH RecentData AS (
                              SELECT
                                `{timestamp_col_for_arima}`,
                                `{metric_col_for_arima}`
                              FROM
                                {full_table_id_quoted}
                              WHERE
                                DATE(`{timestamp_col_for_arima}`) >= CURRENT_DATE()
                            ),
                            AnomalyResults AS (
                              SELECT
                                *
                              FROM
                                ML.DETECT_ANOMALIES(
                                  MODEL {full_bqml_model_path},
                                  STRUCT({anomaly_threshold} AS anomaly_prob_threshold),
                                  TABLE RecentData
                                )
                            )
                            SELECT
                              COUNT(*) AS failed_count
                            FROM
                              AnomalyResults
                            WHERE
                              is_anomaly = TRUE;
                        """
                        anomaly_detection_sql = " ".join(anomaly_detection_sql.split()).strip().rstrip(';')

                        cleaned_metric_col_name = metric_col_for_arima.replace('_', ' ')
                        cleaned_timestamp_col_name = timestamp_col_for_arima.replace('_', ' ')
                        description_prompt = f"""
                        Provide a concise description for a data quality rule.
                        This rule uses an ARIMA time series model to detect anomalies in the '{cleaned_metric_col_name}' column of table '{table_name}'.
                        The model analyzes historical '{cleaned_metric_col_name}' values against the '{cleaned_timestamp_col_name}' to learn and predict expected patterns, accounting for underlying trends and any repeating seasonal behaviors.
                        It flags values as anomalous if they significantly deviate from this learned prediction, indicating unusual activity.
                        Start directly with the description without introductory phrases like 'This rule checks for...'.
                        """
                        ai_generated_description = generate_content(description_prompt)
                        if "No content generated" in ai_generated_description or "Error generating content" in ai_generated_description:
                            ai_generated_description = (
                                f"Detects anomalies in '{metric_col_for_arima}' by training an ARIMA time series model on historical data from '{timestamp_col_for_arima}'. "
                                f"The model learns trends, seasonality, and autocorrelation in the data to predict expected values, flagging significant deviations as anomalies."
                            )

                        all_rules_to_insert.append({
                            'rule_id': str(uuid.uuid4()),
                            'source_project_id': PROJECT_ID,
                            'source_dataset_id': target_dataset_id,
                            'source_table_id': table_name,
                            'metric_column': metric_col_for_arima, # This is the specific metric column chosen for the table's ARIMA model
                            'rule_generation_timestamp': datetime.now().isoformat(),
                            'rule_sql': anomaly_detection_sql,
                            'rule_family': "Anomaly Detection",
                            'rule_description': ai_generated_description
                        })

                    except GoogleAPIError as e:
                        print(f"Error training BQML model or getting results for {table_name}.{metric_col_for_arima}: {e}")
                        if "Input data doesn't contain any rows" in str(e):
                            print("This error often means the training data range or filter resulted in no rows, or the frequency is too low.")
                        print("Skipping ARIMA model and anomaly rule generation for this table.")
                    except Exception as e:
                        print(f"An unexpected error occurred during ARIMA model training or rule generation for {table_name}.{metric_col_for_arima}: {e}")
                        print("Skipping ARIMA model and anomaly rule generation for this table.")
                else:
                    print(f"Skipping ARIMA model training due to invalid or un-determinable date range for '{table_name}'.")
            else:
                print(f"Skipping Anomaly Detection for table '{table_name}': No suitable numeric and/or timestamp columns found.")

        except Exception as e:
            print(f"Error querying columns for table {table_name}: {e}")
            print("Please ensure the table exists and you have permissions to access its INFORMATION_SCHEMA.")
            continue

    # --- NEW: Generate Referential Integrity Rules (can be done sequentially or in a separate thread if many) ---
    print("\n--- Generating Referential Integrity Rules ---")
    if not REFERENTIAL_INTEGRITY_RULES:
        print("No referential integrity rules defined in REFERENTIAL_INTEGRITY_RULES. Skipping.")
    else:
        for ref_rule in REFERENTIAL_INTEGRITY_RULES:
            try:
                referencing_table, referencing_column, referenced_table, referenced_column = ref_rule

                # Check if both tables exist in the target dataset before creating the rule
                try:
                    bq_client.get_table(f"{PROJECT_ID}.{target_dataset_id}.{referencing_table}")
                    bq_client.get_table(f"{PROJECT_ID}.{target_dataset_id}.{referenced_table}")
                except NotFound:
                    print(f"Skipping referential integrity rule for '{referencing_table}.{referencing_column}' -> '{referenced_table}.{referenced_column}': One or both tables not found in dataset '{target_dataset_id}'.")
                    continue
                except Exception as e:
                    print(f"Error checking existence of tables for RI rule '{referencing_table}.{referencing_column}' -> '{referenced_table}.{referenced_column}': {e}. Skipping.")
                    continue

                full_referencing_table_id_quoted = f"`{PROJECT_ID}.{target_dataset_id}.{referencing_table}`"
                full_referenced_table_id_quoted = f"`{PROJECT_ID}.{target_dataset_id}.{referenced_table}`"

                ri_sql = f"""
                    SELECT COUNT(T1.`{referencing_column}`) AS failed_count
                    FROM {full_referencing_table_id_quoted} AS T1
                    LEFT JOIN {full_referenced_table_id_quoted} AS T2
                    ON T1.`{referencing_column}` = T2.`{referenced_column}`
                    WHERE T1.`{referencing_column}` IS NOT NULL AND T2.`{referenced_column}` IS NULL
                """
                ri_sql = " ".join(ri_sql.split()).strip()

                cleaned_referencing_col = referencing_column.replace('_', ' ')
                cleaned_referenced_col = referenced_column.replace('_', ' ')

                ri_description = (
                    f"Checks for referential integrity: ensures all values in '{cleaned_referencing_col}' of table '{referencing_table}' "
                    f"have a matching value in '{cleaned_referenced_col}' of table '{referenced_table}'."
                )

                all_rules_to_insert.append({
                    'rule_id': str(uuid.uuid4()),
                    'source_project_id': PROJECT_ID,
                    'source_dataset_id': target_dataset_id,
                    'source_table_id': referencing_table,
                    'metric_column': referencing_column,
                    'rule_generation_timestamp': datetime.now().isoformat(),
                    'rule_sql': ri_sql,
                    'rule_family': "Referential Integrity",
                    'rule_description': ri_description
                })
            except Exception as e:
                print(f"Error generating referential integrity rule for {ref_rule}: {e}")

    # Now, insert all collected rules into the control table in a single batch insert (or batched inserts)
    if all_rules_to_insert:
        print(f"\nAttempting to insert {len(all_rules_to_insert)} generated rules into '{CONTROL_TABLE_FULL_ID}'...")
        # BigQuery's insert_rows can handle a list of rows, so one large insert is efficient.
        # However, for extremely large numbers of rules, you might want to batch this.
        # For simplicity, we'll do one big insert for now.
        insert_rule_data_into_control_table(all_rules_to_insert)
    else:
        print("No rules were generated to insert into the control table.")


    print("\n--- Data quality rule population complete. ---")

def insert_rule_data_into_control_table(rules_data: list, max_retries=5, initial_delay=1):
    """
    Inserts a list of rule definitions into the BigQuery control table with retry logic.
    This function is designed to handle batch inserts.
    """
    if not rules_data:
        print("No rules to insert.")
        return

    print(f"Inserting {len(rules_data)} rules into '{CONTROL_TABLE_FULL_ID}'...")
    for attempt in range(max_retries):
        try:
            table = bq_client.get_table(CONTROL_TABLE_FULL_ID)
            # Each item in rules_data is a dict, convert to tuple matching schema order
            rows_to_insert = [
                (
                    rule['rule_id'],
                    rule['source_project_id'],
                    rule['source_dataset_id'],
                    rule['source_table_id'],
                    rule['metric_column'],
                    rule['rule_generation_timestamp'],
                    rule['rule_sql'],
                    rule['rule_family'],
                    rule['rule_description']
                ) for rule in rules_data
            ]
            errors = bq_client.insert_rows(table, rows_to_insert)

            if errors:
                print(f"Attempt {attempt + 1}/{max_retries}: Encountered errors while inserting rules: {errors}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                    print("Retrying batch insertion...")
                else:
                    print(f"Failed to insert rules after {max_retries} attempts.")
            else:
                print(f"Successfully inserted {len(rules_data)} rules.")
                return True
        except GoogleAPIError as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Google API Error during batch insertion: {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))
                print("Retrying batch insertion...")
            else:
                print(f"Failed to insert rules after {max_retries} attempts due to API error.")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Unexpected Error during batch insertion: {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))
                print("Retrying batch insertion...")
            else:
                print(f"Failed to insert rules after {max_retries} attempts due to unexpected error.")
    return False

# --- NEW: RuleExecutor Class as Sub-agent ---
class RuleExecutor:
    def __init__(self, bq_client, project_id, control_dataset_id, failed_rows_sample_table_full_id, referential_integrity_rules):
        self.bq_client = bq_client
        self.project_id = project_id
        self.control_dataset_id = control_dataset_id
        self.failed_rows_sample_table_full_id = failed_rows_sample_table_full_id
        self.referential_integrity_rules = referential_integrity_rules

    def _get_first_column_name(self, source_project_id, source_dataset_id, source_table_id):
        """Helper to dynamically get the first column name for source_id."""
        first_column_query = f"""
            SELECT column_name
            FROM `{source_project_id}.{source_dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{source_table_id}'
            ORDER BY ordinal_position
            LIMIT 1
        """
        try:
            first_col_job = self.bq_client.query(first_column_query)
            first_col_results = first_col_job.result()
            for row in first_col_results:
                return row['column_name']
            return None
        except Exception as e:
            print(f"WARNING: Error getting first column name for '{source_table_id}': {e}")
            return None

    def execute_rule(self, rule: dict) -> tuple:
        """
        Executes a single data quality rule and returns its result and failed row samples.
        Returns (result_dict, failed_row_samples_list).
        """
        rule_id = rule['rule_id']
        source_project_id = rule['source_project_id']
        source_dataset_id = rule['source_dataset_id']
        source_table_id = rule['source_table_id']
        metric_column = rule['metric_column']
        rule_sql = rule['rule_sql']
        rule_family = rule['rule_family']
        log_rule_identifier = f"{rule_family} - {source_table_id}.{metric_column} (ID: {rule_id})"

        execution_start_time = time.time()
        failed_rows_count = 0
        status = "ERROR"
        error_message = None
        current_execution_timestamp = datetime.now().isoformat()
        failed_row_samples_for_this_rule = []

        try:
            job = self.bq_client.query(rule_sql)
            query_results = job.result()
            execution_end_time = time.time()
            execution_duration_seconds = execution_end_time - execution_start_time

            for row in query_results:
                failed_rows_count = row['failed_count']
                break

            status = "FAIL" if failed_rows_count > 0 else "PASS"
            print(f"Executing rule: '{log_rule_identifier}' on table '{source_project_id}.{source_dataset_id}.{source_table_id}'...")
            print(f"  Result: {status}, Failed Rows: {failed_rows_count}, Duration: {execution_duration_seconds:.2f}s")

            # Capture Failed Rows Sample if status is FAIL
            if status == "FAIL":
                first_col_name = self._get_first_column_name(source_project_id, source_dataset_id, source_table_id)
                if first_col_name:
                    sample_selection_query = self._build_sample_query(
                        rule_family, rule_sql, source_project_id, source_dataset_id,
                        source_table_id, metric_column, first_col_name
                    )

                    if sample_selection_query:
                        print(f"  Fetching sample failed rows for rule '{log_rule_identifier}'...")
                        try:
                            sample_failed_rows_job = self.bq_client.query(sample_selection_query)
                            sample_failed_rows_iterator = sample_failed_rows_job.result()
                            for failed_row in sample_failed_rows_iterator:
                                source_id_val = failed_row.get('source_id_val')
                                failed_row_samples_for_this_rule.append(
                                    (
                                        str(uuid.uuid4()),
                                        rule_id,
                                        str(source_id_val) if source_id_val is not None else None,
                                        source_project_id,
                                        source_dataset_id,
                                        source_table_id,
                                        current_execution_timestamp,
                                        metric_column
                                    )
                                )
                        except Exception as e:
                            print(f"  WARNING: Could not fetch sample failed rows for rule '{log_rule_identifier}' due to query execution error: {e}")
                else:
                    print(f"  INFO: Skipping sample capture for rule '{log_rule_identifier}' as first column (source_id) could not be determined.")

        except GoogleAPIError as e:
            execution_end_time = time.time()
            execution_duration_seconds = execution_end_time - execution_start_time
            error_message = f"BigQuery API Error: {e}"
            print(f"  Error executing rule '{log_rule_identifier}': {error_message}")
        except Exception as e:
            execution_end_time = time.time()
            execution_duration_seconds = execution_end_time - execution_start_time
            error_message = f"Unexpected Error: {e}"
            print(f"  Error executing rule '{log_rule_identifier}': {error_message}")

        result_data = {
            'result_id': str(uuid.uuid4()),
            'rule_id': rule_id,
            'source_dataset_id': source_dataset_id,
            'source_table_id': source_table_id,
            'metric_column': metric_column,
            'rule_family': rule_family,
            'execution_timestamp': current_execution_timestamp,
            'execution_duration_seconds': execution_duration_seconds,
            'status': status,
            'failed_rows': failed_rows_count if status == "FAIL" else None,
            'error_message': error_message
        }
        return result_data, failed_row_samples_for_this_rule

    def _build_sample_query(self, rule_family, rule_sql, source_project_id, source_dataset_id, source_table_id, metric_column, first_col_name):
        """Helper to build the appropriate sample selection query based on rule family."""
        sample_selection_query = None
        full_table_id_quoted = f"`{source_project_id}.{source_dataset_id}.{source_table_id}`"

        if rule_family == "Null Check":
            sample_selection_query = f"""
                SELECT `{first_col_name}` AS source_id_val
                FROM {full_table_id_quoted}
                WHERE `{metric_column}` IS NULL
                LIMIT 10
            """
        elif rule_family == "Uniqueness":
            sample_selection_query = f"""
                SELECT `{first_col_name}` AS source_id_val
                FROM {full_table_id_quoted}
                WHERE `{metric_column}` IN (
                    SELECT `{metric_column}`
                    FROM {full_table_id_quoted}
                    WHERE `{metric_column}` IS NOT NULL
                    GROUP BY `{metric_column}`
                    HAVING COUNT(*) > 1
                )
                LIMIT 10
            """
        elif rule_family == "Referential Integrity":
            # To reconstruct the RI sample query, we need the referenced table/column.
            found_ri_config = None
            for ri_config in self.referential_integrity_rules:
                if ri_config[0] == source_table_id and ri_config[1] == metric_column:
                    found_ri_config = ri_config
                    break
            if found_ri_config:
                _, _, referenced_table_id_ri, referenced_column_ri = found_ri_config
                full_referenced_table_id_ri_quoted = f"`{source_project_id}.{source_dataset_id}.{referenced_table_id_ri}`"
                sample_selection_query = f"""
                    SELECT T1.`{first_col_name}` AS source_id_val
                    FROM {full_table_id_quoted} AS T1
                    LEFT JOIN {full_referenced_table_id_ri_quoted} AS T2
                    ON T1.`{metric_column}` = T2.`{referenced_column_ri}`
                    WHERE T1.`{metric_column}` IS NOT NULL AND T2.`{referenced_column_ri}` IS NULL
                    LIMIT 10
                """
            else:
                print(f"  WARNING: Could not find RI config for {source_table_id}.{metric_column}. Skipping sample capture for RI.")

        elif rule_family == "String Formatting":
            # The AI-generated SQL is typically `SELECT COUNT(*) FROM (actual_sql_that_returns_failed_rows)`.
            # We need to extract `actual_sql_that_returns_failed_rows`.
            inner_query_start_idx = rule_sql.lower().find("from (")
            inner_query_end_idx = rule_sql.rfind(")")

            if inner_query_start_idx != -1 and inner_query_end_idx != -1:
                inner_query = rule_sql[inner_query_start_idx + len("from ("):inner_query_end_idx].strip()
                sample_selection_query = f"""
                    SELECT `{first_col_name}` AS source_id_val
                    FROM ({inner_query}) AS subquery_for_sample
                    LIMIT 10
                """
            else:
                print(f"  WARNING: String Formatting rule SQL structure unexpected for {source_table_id}.{metric_column}. Cannot extract inner query for sample rows. Skipping sample capture.")

        elif rule_family == "Anomaly Detection":
            # Re-run the ML.DETECT_ANOMALIES to get the anomalous timestamps, then select the first column.
            # The metric_column here is the specific one chosen for the ARIMA model for this table.
            table_columns_df = self.bq_client.query(f"""
                SELECT column_name, data_type
                FROM `{source_project_id}.{source_dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{source_table_id}' AND data_type IN ('TIMESTAMP', 'DATETIME', 'DATE')
                ORDER BY ordinal_position LIMIT 1
            """).to_dataframe()

            if not table_columns_df.empty:
                timestamp_col_for_arima_sample = table_columns_df.iloc[0]['column_name']
                # The model name should correspond to the one generated for this table and its chosen metric_column
                bqml_model_name_for_sample = f"dq_arima_{source_table_id.replace('-', '_')}_{metric_column.replace('-', '_')}"
                full_bqml_model_path_for_sample = f"`{self.project_id}.{self.control_dataset_id}.{bqml_model_name_for_sample}`"

                sample_selection_query = f"""
                    SELECT T1.`{first_col_name}` AS source_id_val
                    FROM {full_table_id_quoted} AS T1
                    WHERE T1.`{timestamp_col_for_arima_sample}` IN (
                        SELECT `{timestamp_col_for_arima_sample}`
                        FROM ML.DETECT_ANOMALIES(
                            MODEL {full_bqml_model_path_for_sample},
                            STRUCT(0.99 AS anomaly_prob_threshold),
                            TABLE (
                                SELECT
                                    `{timestamp_col_for_arima_sample}`,
                                    `{metric_column}`
                                FROM
                                    {full_table_id_quoted}
                                WHERE
                                    DATE(`{timestamp_col_for_arima_sample}`) >= CURRENT_DATE()
                            )
                        )
                        WHERE is_anomaly = TRUE
                    )
                    LIMIT 10
                """
            else:
                print(f"  WARNING: Could not find a timestamp column for anomaly detection sample for {source_table_id}.{metric_column}. Skipping sample capture.")
        else:
            print(f"  INFO: Rule family '{rule_family}' not specifically handled for sample capture. Skipping.")

        return sample_selection_query


# --- New Function to Execute Data Quality Rules ---
def execute_data_quality_rules():
    """
    Fetches all defined data quality rules from the control table,
    executes each rule's SQL using a RuleExecutor sub-agent, and records the results
    in the results table and failed rows sample table.
    """
    print(f"\n--- Starting execution of data quality rules from '{CONTROL_TABLE_FULL_ID}' ---")

    rule_executor = RuleExecutor(bq_client, PROJECT_ID, CONTROL_DATASET_ID, FAILED_ROWS_SAMPLE_TABLE_FULL_ID, REFERENTIAL_INTEGRITY_RULES)

    # Fetch all rules from the control table
    rules_query = f"""
        SELECT
            rule_id,
            source_project_id,
            source_dataset_id,
            source_table_id,
            metric_column,
            rule_generation_timestamp,
            rule_sql,
            rule_family,
            rule_description
        FROM `{CONTROL_TABLE_FULL_ID}`
    """
    try:
        rules_df = bq_client.query(rules_query).to_dataframe()
    except Exception as e:
        print(f"Error fetching rules from control table '{CONTROL_TABLE_FULL_ID}': {e}")
        return

    if rules_df.empty:
        print("No data quality rules found in the control table to execute.")
        return

    all_results_to_insert = []
    all_failed_row_samples_to_insert = []

    MAX_EXECUTION_WORKERS = 10 # Adjust based on BigQuery concurrency limits and CPU cores

    with ThreadPoolExecutor(max_workers=MAX_EXECUTION_WORKERS) as executor:
        execution_futures = [executor.submit(rule_executor.execute_rule, rule.to_dict()) for _, rule in rules_df.iterrows()]

        for future in as_completed(execution_futures):
            try:
                result_data, failed_row_samples = future.result()
                # Only append to main results if it's a FAIL or ERROR status
                if result_data['status'] == "FAIL" or result_data['status'] == "ERROR":
                    all_results_to_insert.append(tuple(result_data.values())) # Convert dict to tuple for insertion
                all_failed_row_samples_to_insert.extend(failed_row_samples)
            except Exception as exc:
                print(f"Rule execution task generated an exception: {exc}")

    # Insert all collected results into the main results table
    if all_results_to_insert:
        print(f"\nInserting {len(all_results_to_insert)} results into '{RESULTS_TABLE_FULL_ID}'...")
        insert_success = False
        max_retries = 5
        initial_delay = 2 # Start with a slightly longer delay for inserts

        for attempt in range(max_retries):
            try:
                results_table = bq_client.get_table(RESULTS_TABLE_FULL_ID)
                errors = bq_client.insert_rows(results_table, all_results_to_insert)
                if errors:
                    print(f"Attempt {attempt + 1}/{max_retries}: Encountered errors while inserting results: {errors}")
                    if attempt < max_retries - 1:
                        time.sleep(initial_delay * (2 ** attempt))
                        print("Retrying results insertion...")
                    else:
                        print(f"Failed to insert results after {max_retries} attempts.")
                else:
                    print("Successfully inserted all rule execution results.")
                    insert_success = True
                    break
            except GoogleAPIError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Google API Error during results insertion: {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                    print("Retrying results insertion...")
                else:
                    print(f"Failed to insert results after {max_retries} attempts due to API error.")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Unexpected Error during results insertion: {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                    print("Retrying results insertion...")
                else:
                    print(f"Failed to insert results after {max_retries} attempts due to unexpected error.")

        if not insert_success:
            print("WARNING: Not all results were inserted into the main results table due to repeated errors.")
    else:
        print("No results to insert into the main results table (all rules passed or encountered errors not recorded).")

    # --- Insert collected failed row samples into the new table ---
    if all_failed_row_samples_to_insert:
        print(f"\nInserting {len(all_failed_row_samples_to_insert)} sample failed rows into '{FAILED_ROWS_SAMPLE_TABLE_FULL_ID}'...")
        insert_success = False
        max_retries = 5
        initial_delay = 2

        for attempt in range(max_retries):
            try:
                failed_rows_sample_table = bq_client.get_table(FAILED_ROWS_SAMPLE_TABLE_FULL_ID)
                errors = bq_client.insert_rows(failed_rows_sample_table, all_failed_row_samples_to_insert)
                if errors:
                    print(f"Attempt {attempt + 1}/{max_retries}: Encountered errors while inserting failed row samples: {errors}")
                    if attempt < max_retries - 1:
                        time.sleep(initial_delay * (2 ** attempt))
                        print("Retrying failed row samples insertion...")
                    else:
                        print(f"Failed to insert failed row samples after {max_retries} attempts.")
                else:
                    print("Successfully inserted all sample failed rows.")
                    insert_success = True
                    break
            except GoogleAPIError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Google API Error during failed row samples insertion: {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                    print("Retrying failed row samples insertion...")
                else:
                    print(f"Failed to insert failed row samples after {max_retries} attempts due to API error.")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Unexpected Error during failed row samples insertion: {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                    print("Retrying failed row samples insertion...")
                else:
                    print(f"Failed to insert failed row samples after {max_retries} attempts due to unexpected error.")

        if not insert_success:
            print("WARNING: Not all failed row samples were inserted due to repeated errors.")
    else:
        print("No failed rows to sample and insert.")

    print("\n--- Data quality rule execution complete. ---")


# --- Execute the population and execution functions ---
if __name__ == "__main__":
    # --- CRITICAL: Ensure the BigQuery dataset exists first ---
    if not ensure_dataset_exists(CONTROL_DATASET_ID, PROJECT_ID, LOCATION):
        print(f"FATAL: Could not ensure dataset '{CONTROL_DATASET_ID}' exists. Exiting.")
        exit(1) # Exit if dataset cannot be ensured

    # Create threads for results and failed rows sample table creation
    results_thread = threading.Thread(target=create_results_table)
    failed_rows_thread = threading.Thread(target=create_failed_rows_sample_table)

    # Start the threads
    results_thread.start()
    failed_rows_thread.start()

    # Wait for both threads to complete
    results_thread.join()
    failed_rows_thread.join()

    # Check if table creation was successful
    # Note: The `create_results_table` and `create_failed_rows_sample_table` functions
    # already print their success/failure. You might want to capture their return
    # values if you need to perform additional conditional logic here.
    # For now, we'll re-call them to just verify their state after the threads complete.
    # A more robust check might be to modify the functions to return a boolean and
    # check that boolean here.
    if not create_results_table() or not create_failed_rows_sample_table():
        exit("Exiting: Failed to create or replace results or failed rows sample table.")


    # Add a short sleep here!
    print("Waiting a few seconds for results and failed rows sample tables to stabilize after creation/replacement...")
    time.sleep(5) # Wait for 5 seconds

    # Example usage: Call with the dataset you want to scan for rule generation
    # To run on 'source_data' dataset:
    populate_data_quality_rules(target_dataset_id="source_data")

    # Now, execute the generated rules and populate the results table
    execute_data_quality_rules()