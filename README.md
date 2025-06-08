# Data Quality Diagnosis AI

## Build for Everyone

## Table of Contents

* [Overview](#overview)
* [Team](#team)
* [Experience at Google](#experience-at-google)
* [Project Timeline](#project-timeline)
* [What is Data Quality?](#what-is-data-quality)
* [Our Purpose](#our-purpose)
* [Project Overview and Approach](#project-overview-and-approach)
* [Designs and Architecture](#designs-and-architecture)
* [Use Case and User Persona (Walmart Example)](#use-case-and-user-persona-walmart-example)
* [Rule Generation and Anomaly Detection Flow](#rule-generation-and-anomaly-detection-flow)
* [Vertex AI Integration](#vertex-ai-integration)
* [Rule Generation Details](#rule-generation-details)
* [String Formatting Rules](#string-formatting-rules)
* [Anomaly Detection (ARIMA)](#anomaly-detection-arima)
* [Additional Rules for Complexity](#additional-rules-for-complexity)
* [Control Table](#control-table)
* [Rule Execution](#rule-execution)
* [Optimizations with Execution](#optimizations-with-execution)
* [Agentic AI Testing](#agentic-ai-testing)
* [Dashboards](#dashboards)
* [Operations for the User (CRUD UI)](#operations-for-the-user-crud-ui)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Configuration](#configuration)
    * [Running the Project](#running-the-project)
* [Host & Mentors](#host--mentors)
* [License](#license)
* [Contact](#contact)

## Overview

This project, "Data Quality Diagnosis AI," developed by Austin's Google x Break Through Tech Sprinternship Cohort, aims to build a powerful data quality assurance tool. It leverages Google Cloud's BigQuery and Vertex AI (specifically the Gemini model) to proactively identify and address data quality issues, ensuring the accuracy, consistency, and reliability of data for Google Cloud customers.

## Team

*   **Rayna DeJesus** 
*   **Ritika Banepalli** 
*   **Akshitaa Balasai** 
*   **Maanasvi Kotturi** 
*   **Odette Saenz** 

## Experience at Google

Our experience at Google was multi-faceted, encompassing:

*   **Challenge Project**: Sprinterns built a scalable AI-powered data quality management system, leveraging Google BigQuery and Gemini models.
*   **Skills Workshops**: Each day, Sprinterns engaged in at least one workshop ranging from Googlers' Day in the Life, Women in Tech Panels, Interview Prep, etc! 
*   **Bonding**: Sprinterns and Google mentors connected after work and had fun getting to know each other! 
*   **Workplace**: Sprinterns explored the workplace and engaged with all of Google's amazing amenities such as food, office tours, pop-up kitchens, etc! 

## Project Timeline

  The project followed a structured timeline over three weeks:

*   **Week One**: Introduction to Google, Training, and Project Planning.
*   **Week Two**: Initial Implementations, Revisions, and Build Heavy.
*   **Week Three**: Finalizing the Build and Presentation Preparation.

## What is Data Quality?

  Data quality is crucial because "Over 60% of data scientists' time is spent cleaning and validating data instead of building insights. Poor data quality leads to flawed business decisions and increased operational costs."    Data quality ensures that data is **accurate, consistent, and reliable**.   This is essential for making informed decisions, especially when working with large datasets in BigQuery.   Examples of ensuring data quality include checking for correct string formatting, missing values, and unusual patterns.

## Our Purpose

  We aim to provide an AI-powered solution that enhances data quality for Google Cloud customers who rely on BigQuery for critical data analysis.   The goal is to empower these customers to proactively identify and address data quality issues, ensuring the accuracy and trustworthiness of their BigQuery data.   This initiative will give our customers more confidence to empower their data-driven decisions.

## Project Overview and Approach

  The core question addressed by this project is: "How might we leverage AI to develop a data quality guardian for BigQuery that automatically identifies and flags potential data integrity issues, ensuring reliable data for our customers?" 

  Our approach includes:
1.    Researching data quality rule families.
2.    Utilizing AI to find patterns and generate SQL queries to the rule families.
3.    Developing the mechanism to execute the queries and store results.
4.    Implementing alerting with 3rd party integrations.
5.    Creating a dashboard to visualize the results of the executed data quality rules.
6.    Developing a CRUD UI for users to manage data quality rules.

## Designs and Architecture

  The design incorporates several key components:

*   **AI (Gemini/Chatbot)**: Responsible for generating SQL queries based on identified patterns.
*   **Control Table**: Stores all SQL rules derived from the generative AI.
*   **Result Table**: Stores failed rows when SQL queries are applied.
*   **Looker Dashboard**: Used for analytics and monitoring.
*   **UI Agent**: Generates rules and interacts with the control table.
*   **Data Owner/Platform Admin**: Users who interact with the system.
*   **Storage**: Data flows from POS to BQ data warehouse, then to Looker Admin and Analytics.

## Use Case and User Persona (Walmart Example)

  To illustrate the project's utility, consider the Walmart use case:

* **Use Case**: Walmart tracks products with store registers (point of sale) systems. The POS information goes into Google Cloud.   Walmart uses this data to tell suppliers what products to send.
*   **Data Owner**:
    *   **Goals**: Ensure stock data is accurate so Walmart can order the right amount of products.
    *   **Challenges**: Data may be documented incorrectly, which makes it difficult to know exactly what Walmart needs to order.
    *   **Our Project Offers**: Our UI tool offers CRUD operations for the data.
*   **Platform Admin**:
    *   **Goals**: Ensure revenue data is correct from Walmart's POS system.
    * **Challenges**: Data may have sudden jumps or drops that aren't real.   Inaccurate demand can lead to inaccurate supply, which impacts Walmart.
    *   **Our Project Offers**: Our dashboard tracks revenue anomalies and other data quality issues.

## Rule Generation and Anomaly Detection Flow

  The data quality process involves two main flows:

**Data Detection for Rule Families:**
1.    **Rule Generation** 
2.    **Storing** 
3.    **Execution** 
4.    **Alerting** 
5.    **Visualization** 

**ML Model for Anomalies:**
1.    **Find a Metric to Measure** 
2.    **Model Training and Prediction** 
3.    **Apply Forecasting to the Model** 
4.    **Insert Model into Storing** 

## Vertex AI Integration

  Vertex AI is strategically integrated to seamlessly incorporate Generative AI into the project workflow.

*   **Automated Prompting Engine**: Leveraged Python to automate and streamline prompt generation.
*   **Primary Use Case**: Focused on generating custom rules tailored to individual customer needs and client-specific data.
*   **Enhanced Output Control & Precision**:
    *   Fine-tuned AI responses by adjusting temperature.
    *   Provided example inputs and outputs (few-shot prompting) for improved accuracy.
*   **Core Model**: Utilized the `gemini-flash-2.0` model.

## Rule Generation Details

  The rule generation is powered by Vertex AI so it works for any dataset.

  **Challenges encountered while working with AI**:
*   Can't handle complex tasks with large amounts of data at once.
*   Required prompt engineering to output valid SQL queries.

  **Generated Rule Families**:
*   **Level 1**: Generated 2 rule families: String Formatting and Anomaly Detection.
*   **Level 2**: Generated 5 rule families: Level 1 + Null Check, Referential Integrity, and Uniqueness.

## String Formatting Rules

  The process for string formatting rules involves:
1.    Selects each string column in the tables.
2.    Gemini analyzes each column.
3.    Outputs SQL Queries for each.

  **Example Usage**:
*   Emails must be in `user@example.ext` format.
*   Standardized address formatting.
*   Names must be in First Last format.
*   No special characters in Names column.

## Anomaly Detection (ARIMA)

  Anomaly detection uses the following flow:
1.    **Column Discovery**: Query BigQuery metadata to find a timestamp column for time-series analysis.
2.    **Run Anomaly Detection**: Use `ML.DETECT_ANOMALIES` with ARIMA model on recent data to find outliers.
3.    **Query Affected Rows**: Pull up to 10 rows from the original table where those anomalies occurred.
4.    **Model Identification**: Construct the name of pre-trained ARIMA model for the metric and table.
5.    **Extract Anomalous Timestamps**: Identify which timestamps contain anomalies based on the model's output.
6.    **Store & Alert**: Store these samples for review and trigger alerts if rule execution fails.

## Additional Rules for Complexity

  The project also incorporates rules for more complex data quality checks:

*   **Null Check**: Ensures critical fields are not missing in the data.
*   **Primary Key (Uniqueness)**: Ensures each record in the table is uniquely identifiable to maintain trust and consistency.
*   **Referential Integrity**: Ensures relationships between tables are valid -i.e., foreign key values must exist in the referenced (parent) table.

## Control Table

  The Control Table is central to managing data quality rules.   It stores various metadata about each rule:

| Field name                  | Type      |
| :-------------------------- | :-------- |
| `rule_id`                   | `STRING`  |
| `source_project_id`         | `STRING`  |
| `source_dataset_id`         | `STRING`  |
| `source_table_id`           | `STRING`  |
| `metric_column`             | `STRING`  |
| `rule_generation_timestamp` | `TIMESTAMP` |
| `rule_sql`                  | `STRING`  |
| `rule_family`               | `STRING`  |
| `rule_description`          | `STRING`  |

## Rule Execution

  The rule execution process involves:
*   Runs through all queries in the data.
*   Applied to appropriate datasets and metrics.
*   Results are stored in a results table.
*   Looker tables for better visualization.
*   Summaries generated for email alerts.

**Example Alert**:
  `Alert: Data Quality Columns Failed` 
  `source_data is failing 41 data quality columns` 

  This is an automated alert from BigQuery, with data generated from the `cloud-professional-services.sprint.summary_table`.

## Optimizations with Execution

  Optimizations for rule execution include:

*   **Parallelization**: Tasks that can be performed concurrently rather than one after another [ : 48]  , optimizing Time, Application Scalability, and Resource Utilization.   Implementation uses threads in the system that generate different rules in parallel.
* **5 Rule Families**:
    *   **String Formatting**: Checks if text data adheres to expected patterns (email, phone #).
    *   **Anomaly Detection**: Identifies data points that deviate from historical patterns.
    *   **Null Checks**: Verifies all required fields contain values, ensuring no critical information is missing.
    *   **Uniqueness**: Confirms distinct values for primary keys / unique id for each row.
    *   **Referential Integrity**: Validates relationships between data across different tables are correctly maintained.
*   **Sub Agents**: Sub-agents are independent components designed to handle specialized tasks [ : 50]  , allowing easier integration of new features.   There can be specialized agents for taking care of specific rule family generations (i.e one for string formatting, another for null checks, etc).

## Agentic AI Testing

  A Data Analysis Agent was developed using Google's Agent Development Kit.   Key Functionalities:

*   More user friendly way to interact with source data and control table.
*   Translates natural language into SQL queries.
*   Accesses data from source data and control table, runs queries and provides explanations.

## Dashboards

  The project includes dashboards for visualizing data quality metrics:

*   **Summary Table Dashboard**: Displays tables, data breakdown, rule family division, and metric vs. failed rows.
*   **Sample Data Insights**: Provides implemented rules and rule family breakdown.

## Operations for the User (CRUD UI)

  A very simple UI for the user to alter and execute rules through CRUD operations.

*   **Create Rule**: Allows users to define new rules.
*   **Read Rules**: Enables users to view existing rules.
*   **Update Rule**: Provides functionality to modify existing rules.
*   **Delete Rule**: Allows users to remove rules.

## Getting Started

To get a copy of the project up and running on your local machine for development and testing purposes, follow these steps.

### Prerequisites

Before you begin, ensure you have the following:

* **Google Cloud Project**: A Google Cloud project with billing enabled.
* **BigQuery API Enabled**: Ensure the BigQuery API is enabled for your project.
* **Vertex AI API Enabled**: Ensure the Vertex AI API is enabled for your project.
* **Python 3.9+**:
* **`gcloud` CLI**: Authenticated to your Google Cloud Project.
* **Service Account**: A service account with the following roles:
    * `BigQuery Data Editor` (for `sprint` dataset and tables within it)
    * `BigQuery Job User` (to run queries)
    * `BigQuery Metadata Viewer` (to discover tables and columns)
    * `Vertex AI User` (to use the Gemini model)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-github-username/data-quality-diagnosis-ai.git](https://github.com/your-github-username/data-quality-diagnosis-ai.git)
    cd data-quality-diagnosis-ai
    ```

2.  **Set up a Python virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages**:
    ```bash
    pip install google-cloud-bigquery google-cloud-aiplatform google-generativeai pandas
    ```

### Configuration

Set your Google Cloud Project ID and Region as environment variables. It is highly recommended to set these for the script to pick them up automatically.

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1" # Or your desired region, e.g., 'us-west1'
