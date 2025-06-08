# Data Quality Diagnosis AI

## Build for Everyone

<p align="center">
  <img src="https://lh3.googleusercontent.com/pw/APedPNAE4z-b-Q-jR5k_S3tY7s0_J_2oZ4x6v5y2k8Z8f1j4q2R0z5Q5a-C_8y0O0x_Xw=w1200" alt="Team Cloud Nine" width="700"/>
  <br>
  <em>(Image of Team Cloud Nine from the presentation, Page 1)</em>
</p>

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

[cite_start]Meet Team Cloud Nine [cite: 2][cite_start], the Austin's Google x Break Through Tech Sprinternship Cohort behind this project:

* [cite_start]**Rayna DeJesus** 
* [cite_start]**Ritika Banepalli** 
* [cite_start]**Akshitaa Balasai** 
* [cite_start]**Maanasvi Kotturi** 
* [cite_start]**Odette Saenz** 

## Experience at Google

Our experience at Google was multi-faceted, encompassing:

* [cite_start]**Challenge Project**: Sprinterns built a scalable AI-powered data quality management system, leveraging Google BigQuery and Gemini models.
* [cite_start]**Skills Workshops**: Each day, Sprinterns engaged in at least one workshop ranging from Googlers' Day in the Life, Women in Tech Panels, Interview Prep, etc! 
* [cite_start]**Bonding**: Sprinterns and Google mentors connected after work and had fun getting to know each other! 
* [cite_start]**Workplace**: Sprinterns explored the workplace and engaged with all of Google's amazing amenities such as food, office tours, pop-up kitchens, etc! 

## Project Timeline

[cite_start]The project followed a structured timeline over three weeks:

* [cite_start]**Week One**: Introduction to Google, Training, and Project Planning.
* [cite_start]**Week Two**: Initial Implementations, Revisions, and Build Heavy.
* [cite_start]**Week Three**: Finalizing the Build and Presentation Preparation.

## What is Data Quality?

[cite_start]Data quality is crucial because "Over 60% of data scientists' time is spent cleaning and validating data instead of building insights. Poor data quality leads to flawed business decisions and increased operational costs."  [cite_start]Data quality ensures that data is **accurate, consistent, and reliable**. [cite_start]This is essential for making informed decisions, especially when working with large datasets in BigQuery. [cite_start]Examples of ensuring data quality include checking for correct string formatting, missing values, and unusual patterns.

## Our Purpose

[cite_start]We aim to provide an AI-powered solution that enhances data quality for Google Cloud customers who rely on BigQuery for critical data analysis. [cite_start]The goal is to empower these customers to proactively identify and address data quality issues, ensuring the accuracy and trustworthiness of their BigQuery data. [cite_start]This initiative will give our customers more confidence to empower their data-driven decisions.

## Project Overview and Approach

[cite_start]The core question addressed by this project is: "How might we leverage AI to develop a data quality guardian for BigQuery that automatically identifies and flags potential data integrity issues, ensuring reliable data for our customers?" 

[cite_start]Our approach includes:
1.  [cite_start]Researching data quality rule families.
2.  [cite_start]Utilizing AI to find patterns and generate SQL queries to the rule families.
3.  [cite_start]Developing the mechanism to execute the queries and store results.
4.  [cite_start]Implementing alerting with 3rd party integrations.
5.  [cite_start]Creating a dashboard to visualize the results of the executed data quality rules.
6.  [cite_start]Developing a CRUD UI for users to manage data quality rules.

## Designs and Architecture

[cite_start]The design incorporates several key components:

* [cite_start]**AI (Gemini/Chatbot)**: Responsible for generating SQL queries based on identified patterns.
* [cite_start]**Control Table**: Stores all SQL rules derived from the generative AI.
* [cite_start]**Result Table**: Stores failed rows when SQL queries are applied.
* [cite_start]**Looker Dashboard**: Used for analytics and monitoring.
* [cite_start]**UI Agent**: Generates rules and interacts with the control table.
* [cite_start]**Data Owner/Platform Admin**: Users who interact with the system.
* [cite_start]**Storage**: Data flows from POS to BQ data warehouse, then to Looker Admin and Analytics.

## Use Case and User Persona (Walmart Example)

[cite_start]To illustrate the project's utility, consider the Walmart use case:

* **Use Case**: Walmart tracks products with store registers (point of sale) systems. The POS information goes into Google Cloud. [cite_start]Walmart uses this data to tell suppliers what products to send.
* [cite_start]**Data Owner**:
    * [cite_start]**Goals**: Ensure stock data is accurate so Walmart can order the right amount of products.
    * [cite_start]**Challenges**: Data may be documented incorrectly, which makes it difficult to know exactly what Walmart needs to order.
    * [cite_start]**Our Project Offers**: Our UI tool offers CRUD operations for the data.
* [cite_start]**Platform Admin**:
    * [cite_start]**Goals**: Ensure revenue data is correct from Walmart's POS system.
    * **Challenges**: Data may have sudden jumps or drops that aren't real. [cite_start]Inaccurate demand can lead to inaccurate supply, which impacts Walmart.
    * [cite_start]**Our Project Offers**: Our dashboard tracks revenue anomalies and other data quality issues.

## Rule Generation and Anomaly Detection Flow

[cite_start]The data quality process involves two main flows:

**Data Detection for Rule Families:**
1.  [cite_start]**Rule Generation** 
2.  [cite_start]**Storing** 
3.  [cite_start]**Execution** 
4.  [cite_start]**Alerting** 
5.  [cite_start]**Visualization** 

**ML Model for Anomalies:**
1.  [cite_start]**Find a Metric to Measure** 
2.  [cite_start]**Model Training and Prediction** 
3.  [cite_start]**Apply Forecasting to the Model** 
4.  [cite_start]**Insert Model into Storing** 

## Vertex AI Integration

[cite_start]Vertex AI is strategically integrated to seamlessly incorporate Generative AI into the project workflow.

* [cite_start]**Automated Prompting Engine**: Leveraged Python to automate and streamline prompt generation.
* [cite_start]**Primary Use Case**: Focused on generating custom rules tailored to individual customer needs and client-specific data.
* [cite_start]**Enhanced Output Control & Precision**:
    * [cite_start]Fine-tuned AI responses by adjusting temperature.
    * [cite_start]Provided example inputs and outputs (few-shot prompting) for improved accuracy.
* [cite_start]**Core Model**: Utilized the `gemini-flash-2.0` model.

## Rule Generation Details

[cite_start]The rule generation is powered by Vertex AI so it works for any dataset.

[cite_start]**Challenges encountered while working with AI**:
* [cite_start]Can't handle complex tasks with large amounts of data at once.
* [cite_start]Required prompt engineering to output valid SQL queries.

[cite_start]**Generated Rule Families**:
* [cite_start]**Level 1**: Generated 2 rule families: String Formatting and Anomaly Detection.
* [cite_start]**Level 2**: Generated 5 rule families: Level 1 + Null Check, Referential Integrity, and Uniqueness.

## String Formatting Rules

[cite_start]The process for string formatting rules involves:
1.  [cite_start]Selects each string column in the tables.
2.  [cite_start]Gemini analyzes each column.
3.  [cite_start]Outputs SQL Queries for each.

[cite_start]**Example Usage**:
* [cite_start]Emails must be in `user@example.ext` format.
* [cite_start]Standardized address formatting.
* [cite_start]Names must be in First Last format.
* [cite_start]No special characters in Names column.

## Anomaly Detection (ARIMA)

[cite_start]Anomaly detection uses the following flow:
1.  [cite_start]**Column Discovery**: Query BigQuery metadata to find a timestamp column for time-series analysis.
2.  [cite_start]**Run Anomaly Detection**: Use `ML.DETECT_ANOMALIES` with ARIMA model on recent data to find outliers.
3.  [cite_start]**Query Affected Rows**: Pull up to 10 rows from the original table where those anomalies occurred.
4.  [cite_start]**Model Identification**: Construct the name of pre-trained ARIMA model for the metric and table.
5.  [cite_start]**Extract Anomalous Timestamps**: Identify which timestamps contain anomalies based on the model's output.
6.  [cite_start]**Store & Alert**: Store these samples for review and trigger alerts if rule execution fails.

## Additional Rules for Complexity

[cite_start]The project also incorporates rules for more complex data quality checks:

* [cite_start]**Null Check**: Ensures critical fields are not missing in the data.
* [cite_start]**Primary Key (Uniqueness)**: Ensures each record in the table is uniquely identifiable to maintain trust and consistency.
* [cite_start]**Referential Integrity**: Ensures relationships between tables are valid -i.e., foreign key values must exist in the referenced (parent) table.

## Control Table

[cite_start]The Control Table is central to managing data quality rules. [cite_start]It stores various metadata about each rule:

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

[cite_start]The rule execution process involves:
* [cite_start]Runs through all queries in the data.
* [cite_start]Applied to appropriate datasets and metrics.
* [cite_start]Results are stored in a results table.
* [cite_start]Looker tables for better visualization.
* [cite_start]Summaries generated for email alerts.

**Example Alert**:
[cite_start]`Alert: Data Quality Columns Failed` 
[cite_start]`source_data is failing 41 data quality columns` 

[cite_start]This is an automated alert from BigQuery, with data generated from the `cloud-professional-services.sprint.summary_table`.

## Optimizations with Execution

[cite_start]Optimizations for rule execution include:

* [cite_start]**Parallelization**: Tasks that can be performed concurrently rather than one after another [cite: 48][cite_start], optimizing Time, Application Scalability, and Resource Utilization. [cite_start]Implementation uses threads in the system that generate different rules in parallel.
* **5 Rule Families**:
    * [cite_start]**String Formatting**: Checks if text data adheres to expected patterns (email, phone #).
    * [cite_start]**Anomaly Detection**: Identifies data points that deviate from historical patterns.
    * [cite_start]**Null Checks**: Verifies all required fields contain values, ensuring no critical information is missing.
    * [cite_start]**Uniqueness**: Confirms distinct values for primary keys / unique id for each row.
    * [cite_start]**Referential Integrity**: Validates relationships between data across different tables are correctly maintained.
* [cite_start]**Sub Agents**: Sub-agents are independent components designed to handle specialized tasks [cite: 50][cite_start], allowing easier integration of new features. [cite_start]There can be specialized agents for taking care of specific rule family generations (i.e one for string formatting, another for null checks, etc).

## Agentic AI Testing

[cite_start]A Data Analysis Agent was developed using Google's Agent Development Kit. [cite_start]Key Functionalities:

* [cite_start]More user friendly way to interact with source data and control table.
* [cite_start]Translates natural language into SQL queries.
* [cite_start]Accesses data from source data and control table, runs queries and provides explanations.

## Dashboards

[cite_start]The project includes dashboards for visualizing data quality metrics:

* [cite_start]**Summary Table Dashboard**: Displays tables, data breakdown, rule family division, and metric vs. failed rows.
* [cite_start]**Sample Data Insights**: Provides implemented rules and rule family breakdown.

## Operations for the User (CRUD UI)

[cite_start]A very simple UI for the user to alter and execute rules through CRUD operations.

* [cite_start]**Create Rule**: Allows users to define new rules.
* [cite_start]**Read Rules**: Enables users to view existing rules.
* [cite_start]**Update Rule**: Provides functionality to modify existing rules.
* [cite_start]**Delete Rule**: Allows users to remove rules.

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
