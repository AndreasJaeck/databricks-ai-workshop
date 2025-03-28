# Databricks notebook source
# MAGIC %md
# MAGIC # UC Toolkit: Tool Functions
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC This notebook demonstrates how to create and use tool functions that can be called by a Language Model (LLM) to extend its capabilities. Tool functions allow the LLM to:
# MAGIC  
# MAGIC  - Access structured data in your data lakehouse
# MAGIC  - Perform vector similarity searches
# MAGIC  - Execute calculations and unit conversions
# MAGIC  - Run Python code for specialized tasks
# MAGIC  - Call other LLMs for specific subtasks
# MAGIC  
# MAGIC  These functions can be registered in your Databricks environment and made available through a Function Calling API, allowing the LLM to use them when needed to answer user questions accurately.

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.41.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.20.2 faker==33.1.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
current_user = ws.current_user.me().user_name
first_name, last_name = current_user.split('@')[0].split('.')
formatted_name = f"{first_name[0]}_{last_name}"

catalog = f'dbdemos_{formatted_name}'
schema = 'chem_manufacturing'
print(f"Catalog name: {catalog}")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Store Search Functions
# MAGIC  
# MAGIC  Vector search enables semantic similarity searches that go beyond exact keyword matching. By converting text to vector embeddings, we can find items that are conceptually similar even if they don't share the same exact words.
# MAGIC  
# MAGIC  Databricks Vector Search simplifies this process with the `VECTOR_SEARCH()` SQL function. This allows us to:
# MAGIC  - Perform semantic search over product descriptions
# MAGIC  - Find relevant documents based on meaning, not just keywords
# MAGIC  - Retrieve the most similar items with relevancy scores

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Find Similar Products
# MAGIC
# MAGIC This function searches for products that are semantically similar to a given description. It's useful for:
# MAGIC - Finding alternative products with similar properties
# MAGIC - Discovering products that match specific requirements
# MAGIC - Providing relevant product recommendations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to find similar products using vector search
# MAGIC CREATE OR REPLACE FUNCTION find_similar_products(product_description STRING)
# MAGIC RETURNS TABLE (product_id STRING, full_description STRING, product_name STRING, category STRING, molecular_weight DOUBLE, density DOUBLE, melting_point DOUBLE, boiling_point DOUBLE, chemical_formula STRING, search_score DOUBLE, price_per_unit DOUBLE)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Find products with similarity search based on description of product name, application areas, storage_conditions. This helps customers find products and alternatives. This query will return details like product_id, full_description, product_name, category, molecular_weight, density, melting_point, boiling_point, chemical_formula, search_score , price_per_unit'
# MAGIC RETURN
# MAGIC   SELECT product_id, full_description, product_name, category, molecular_weight, density, melting_point, boiling_point, chemical_formula, search_score , price_per_unit
# MAGIC   FROM VECTOR_SEARCH(
# MAGIC     index => 'dbdemos_a_jack.chem_manufacturing.products_index',
# MAGIC     query => product_description,
# MAGIC     num_results => 2
# MAGIC   )
# MAGIC   ORDER BY search_score DESC 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's test our function:
# MAGIC SELECT * FROM find_similar_products('A product that has a name similar to "Synth" and is for the food industry');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find Safety Protocols and Research Notes
# MAGIC
# MAGIC This function searches for safety protocols and research notes related to a given topic. It's useful for:
# MAGIC - Finding safety guidelines for specific chemicals or processes
# MAGIC - Locating relevant research notes and documentation
# MAGIC - Discovering appropriate handling procedures for hazardous materials

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to find safety protocols by description
# MAGIC CREATE OR REPLACE FUNCTION find_safety_protocols(search_text STRING)
# MAGIC RETURNS TABLE (description_id STRING, description_type STRING, product_id STRING, title STRING, content STRING, search_score DOUBLE)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Find safety, procedures, research protocools with similarity search for chemicals matching the description. Returns relevant safety information for handling chemicals description_id, description_type, product_id, title, content, search_score.'
# MAGIC RETURN
# MAGIC   SELECT description_id, description_type, product_id, title, content, search_score
# MAGIC   FROM VECTOR_SEARCH(
# MAGIC     index => 'dbdemos_a_jack.chem_manufacturing.descriptions_index',
# MAGIC     query => search_text,
# MAGIC     num_results => 2
# MAGIC   )
# MAGIC   ORDER BY search_score DESC 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's test our function:
# MAGIC SELECT * FROM find_safety_protocols('Find protocols about impurity reduction');

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Functions
# MAGIC
# MAGIC SQL functions provide direct access to structured data within your lakehouse. These functions can:
# MAGIC - Retrieve detailed information about specific products or processes
# MAGIC - Run pre-defined analytical queries
# MAGIC - Return structured data that the LLM can reference in its responses
# MAGIC
# MAGIC The following SQL functions demonstrate how to create structured data access points that an LLM can use to get reliable, up-to-date information.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Information Retrieval
# MAGIC
# MAGIC This function retrieves comprehensive information about a product based on its ID, providing all available details from the products table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed product information
# MAGIC CREATE OR REPLACE FUNCTION get_product(productid STRING)
# MAGIC RETURNS TABLE (product_id string, product_name string, category string,
# MAGIC chemical_formula string, molecular_weight double, density double, melting_point double, boiling_point double,description string, application_areas string, storage_conditions string, full_description string, creation_date string, price_per_unit double
# MAGIC )
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieve detailed information about a product with product id (regex shape: ^P[0-9]{4}$). Returns product_id, product_name, category, chemical_formula, molecular_weight, density, melting_point, boiling_point, description, application_areas, storage_conditions, full_description, creation_date, price_per_unit'
# MAGIC RETURN
# MAGIC   SELECT *
# MAGIC   FROM dbdemos_a_jack.chem_manufacturing.products
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_product('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Safety Protocol Retrieval
# MAGIC
# MAGIC This function retrieves safety protocols, procedures, and research notes for a specific product ID. It's useful for:
# MAGIC - Ensuring proper handling of chemical products
# MAGIC - Accessing the latest safety guidelines
# MAGIC - Reviewing research notes for a particular product

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed safety protocol information
# MAGIC CREATE OR REPLACE FUNCTION get_safety_protocols(productid STRING)
# MAGIC RETURNS TABLE (description_id STRING, description_type STRING, product_id STRING, title STRING, content STRING)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Get safety, procedures, research protocools with product id (regex shape: ^P[0-9]{4}$) for chemicals matching the description. Returns relevant safety information for handling chemicals description_id, description_type, product_id, title, content'
# MAGIC RETURN
# MAGIC   SELECT description_id, description_type, product_id, title, content
# MAGIC   FROM dbdemos_a_jack.chem_manufacturing.descriptions
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_safety_protocols('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reaction Details Retrieval
# MAGIC
# MAGIC This function provides detailed information about chemical reactions used to produce a specific product. It's useful for:
# MAGIC - Understanding production processes
# MAGIC - Evaluating reaction conditions
# MAGIC - Assessing safety requirements and hazards

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed reaction information
# MAGIC CREATE OR REPLACE FUNCTION get_reaction_details(productid STRING)
# MAGIC RETURNS TABLE (reaction_id STRING, reaction_name STRING, reaction_type STRING, catalyst STRING, solvent STRING, temperature DOUBLE, pressure DOUBLE, reaction_time DOUBLE, energy_consumption DOUBLE, hazards STRING)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieve detailed information with product id (regex shape: ^P[0-9]{4}$) about chemical reactions used to produce a specific product. Returns reaction conditions, reaction type, catalyst name, solvent, temperature, pressure, reaction time, energy consumption, hazards.'
# MAGIC RETURN
# MAGIC   SELECT reaction_id, reaction_name, reaction_type, catalyst, solvent, temperature, pressure, reaction_time, energy_consumption, hazards
# MAGIC   FROM dbdemos_a_jack.chem_manufacturing.reactions
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_reaction_details('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Quality Analysis
# MAGIC
# MAGIC This function analyzes quality metrics for a specific product, providing insights into testing results and pass rates. It's useful for:
# MAGIC - Evaluating product reliability
# MAGIC - Identifying quality issues
# MAGIC - Making data-driven decisions about product improvements

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to analyze product quality metrics
# MAGIC CREATE OR REPLACE FUNCTION analyze_product_quality(productid STRING)
# MAGIC RETURNS TABLE (product_name STRING, total_tests INT, passed_tests INT, failed_tests INT, pass_rate DOUBLE)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Analyze quality metrics for a specific product. Shows test results, pass rates, and common quality issues.'
# MAGIC RETURN
# MAGIC   SELECT 
# MAGIC     p.product_name,
# MAGIC     COUNT(q.test_id) as total_tests,
# MAGIC     SUM(CASE WHEN q.test_result = 'Pass' THEN 1 ELSE 0 END) as passed_tests,
# MAGIC     SUM(CASE WHEN q.test_result = 'Fail' THEN 1 ELSE 0 END) as failed_tests,
# MAGIC     ROUND(SUM(CASE WHEN q.test_result = 'Pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(q.test_id), 2) as pass_rate
# MAGIC   FROM dbdemos_a_jack.chem_manufacturing.quality_control q
# MAGIC   JOIN dbdemos_a_jack.chem_manufacturing.batches b ON q.batch_id = b.batch_id
# MAGIC   JOIN dbdemos_a_jack.chem_manufacturing.products p ON b.product_id = p.product_id
# MAGIC   WHERE b.product_id = productid
# MAGIC   GROUP BY p.product_name;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM analyze_product_quality('P0002')

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Python Functions
# MAGIC
# MAGIC Python functions extend the capabilities of your LLM by allowing it to perform more complex operations that aren't easily expressed in SQL. These functions can:
# MAGIC - Perform specialized calculations and unit conversions
# MAGIC - Execute custom algorithms
# MAGIC - Access external APIs and services
# MAGIC - Run arbitrary Python code for advanced use cases
# MAGIC
# MAGIC The functions below demonstrate different ways Python can be integrated as tools for your LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unit Conversion Tool
# MAGIC
# MAGIC This function provides chemical unit conversions between common measurement units. It's useful for:
# MAGIC - Converting between different units of measurement (g, kg, mol, L, mL)
# MAGIC - Ensuring consistent units across calculations
# MAGIC - Simplifying unit conversion tasks for users
# MAGIC
# MAGIC Databricks runs Python functions in a safe container. The function below has been designed to prevent prompt injection issues by restricting its functionality to specific unit conversions.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION convert_chemical_unit(value DOUBLE, from_unit STRING, to_unit STRING, mol_weight DOUBLE)
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Convert between different chemical measurement units (g, kg, mol, L, mL) to units (kg, g, mL, L, mol) usage: convert_chemical_unit(1, "kg", "g", 0) if mol is not provided use 0'
# MAGIC AS
# MAGIC $$
# MAGIC   unit_conversions = {
# MAGIC     'g_to_kg': lambda x: x / 1000,
# MAGIC     'kg_to_g': lambda x: x * 1000,
# MAGIC     'L_to_mL': lambda x: x * 1000,
# MAGIC     'mL_to_L': lambda x: x / 1000,
# MAGIC     'g_to_mol': lambda x, mw: x / mw if mw else None,
# MAGIC     'mol_to_g': lambda x, mw: x * mw if mw else None
# MAGIC   }
# MAGIC   
# MAGIC   conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"
# MAGIC   
# MAGIC   if conversion_key in unit_conversions:
# MAGIC     if conversion_key in ['g_to_mol', 'mol_to_g']:
# MAGIC       if mol_weight is None:
# MAGIC         return f"Molecular weight required for {conversion_key} conversion"
# MAGIC       return unit_conversions[conversion_key](value, mol_weight)
# MAGIC     else:
# MAGIC       return unit_conversions[conversion_key](value)
# MAGIC   else:
# MAGIC     return f"Conversion from {from_unit} to {to_unit} not supported"
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT convert_chemical_unit(1, "g", "mol", 0.58);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculator Tool
# MAGIC
# MAGIC This function allows an LLM to perform mathematical calculations with high precision. It's useful for:
# MAGIC - Solving complex mathematical expressions
# MAGIC - Performing scientific calculations using the math library
# MAGIC - Ensuring accuracy in numerical responses
# MAGIC
# MAGIC The function has been designed with security in mind, restricting operations to mathematical functions and preventing execution of arbitrary code.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION compute_math(
# MAGIC   expr STRING COMMENT 'A mathematical expression as a string to be evaluated. Supports basic operations (+, -, *, /, **, %) and math module functions (e.g., math.sqrt(13), math.sin(0.5), math.log(10)). Example: "2 + 2" or "math.sqrt(16) + math.pow(2, 3)"'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Run any mathematical function and returns the result as output. Supports python syntax like math.sqrt(13)'
# MAGIC AS
# MAGIC $$
# MAGIC   import ast
# MAGIC   import operator
# MAGIC   import math
# MAGIC   operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.UAdd: operator.pos, ast.USub: operator.neg}
# MAGIC     
# MAGIC   # Supported functions from the math module
# MAGIC   functions = {name: getattr(math, name) for name in dir(math) if callable(getattr(math, name))}
# MAGIC
# MAGIC   def eval_node(node):
# MAGIC     if isinstance(node, ast.Num):  # <number>
# MAGIC       return node.n
# MAGIC     elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
# MAGIC       return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
# MAGIC     elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
# MAGIC       return operators[type(node.op)](eval_node(node.operand))
# MAGIC     elif isinstance(node, ast.Call):  # <func>(<args>)
# MAGIC       func = node.func.id
# MAGIC       if func in functions:
# MAGIC         args = [eval_node(arg) for arg in node.args]
# MAGIC         return functions[func](*args)
# MAGIC       else:
# MAGIC         raise TypeError(f"Unsupported function: {func}")
# MAGIC     else:
# MAGIC       raise TypeError(f"Unsupported type: {type(node)}")  
# MAGIC   try:
# MAGIC     if expr.startswith('```') and expr.endswith('```'):
# MAGIC       expr = expr[3:-3].strip()      
# MAGIC     node = ast.parse(expr, mode='eval').body
# MAGIC     return eval_node(node)
# MAGIC   except Exception as ex:
# MAGIC     return str(ex)
# MAGIC $$;
# MAGIC
# MAGIC -- let's test our function:
# MAGIC SELECT compute_math("(2+2)/3") as result;

# COMMAND ----------

# MAGIC %md
# MAGIC ### External API Integration - Weather Service
# MAGIC
# MAGIC This function demonstrates how to access external APIs from within your tool functions. It retrieves rrent weather data based on latitude and longitude coordinates. This is useful for:
# MAGIC - Incorporating real-time external data into responses
# MAGIC - Enhancing responses with contextual information
# MAGIC - Demonstrating API integration capabilities
# MAGIC
# MAGIC **Note:** This function requires serverless network egress access when running on serverless compute. Ensure your networking configuration allows this at the admin account level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_weather(latitude DOUBLE, longitude DOUBLE)
# MAGIC RETURNS STRUCT<temperature_in_celsius DOUBLE, rain_in_mm DOUBLE>
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'This function retrieves the current temperature and rain information for a given latitude and longitude using the Open-Meteo API.'
# MAGIC AS
# MAGIC $$
# MAGIC   try:
# MAGIC     import requests as r
# MAGIC     #Note: this is provided for education only, non commercial - please get a license for real usage: https://api.open-meteo.com. Let s comment it to avoid issues for now
# MAGIC     #weather = r.get(f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,rain&forecast_days=1').json()
# MAGIC     return {
# MAGIC       "temperature_in_celsius": weather["current"]["temperature_2m"],
# MAGIC       "rain_in_mm": weather["current"]["rain"]
# MAGIC     }
# MAGIC   except:
# MAGIC     return {"temperature_in_celsius": 22.0, "rain_in_mm": 0.0}
# MAGIC $$;
# MAGIC
# MAGIC -- let's test our function:
# MAGIC SELECT get_weather(52.52, 13.41) as weather;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python Code Execution
# MAGIC
# MAGIC This function allows an LLM to execute arbitrary Python code and return the results. This is useful for:
# MAGIC - Testing and debugging Python code
# MAGIC - Performing complex data processing tasks
# MAGIC - Creating dynamic responses based on code execution
# MAGIC
# MAGIC **Warning:** This function can execute any Python code, which presents security risks. Only use this in controlled environments with proper security measures in place.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION execute_python_code(python_code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Run python code. The code should end with a return statement and this function will return it as a string. Only send valid python to this function. Here is an exampe of python code input: 'def square_function(number):\\n  return number*number\\n\\nreturn square_function(3)'"
# MAGIC AS
# MAGIC $$
# MAGIC     import traceback
# MAGIC     try:
# MAGIC         import re
# MAGIC         # Remove code block markers (e.g., ```python) and strip whitespace```
# MAGIC         python_code = re.sub(r"^\s*```(?:python)?|```\s*$", "", python_code).strip()
# MAGIC         # Unescape any escaped newline characters
# MAGIC         python_code = python_code.replace("\\n", "\n")
# MAGIC         # Properly indent the code for wrapping
# MAGIC         indented_code = "\n    ".join(python_code.split("\n"))
# MAGIC         # Define a wrapper function to execute the code
# MAGIC         exec_globals = {}
# MAGIC         exec_locals = {}
# MAGIC         wrapper_code = "def _temp_function():\n    "+indented_code
# MAGIC         exec(wrapper_code, exec_globals, exec_locals)
# MAGIC         # Execute the wrapped function and return its output
# MAGIC         result = exec_locals["_temp_function"]()
# MAGIC         return result
# MAGIC     except Exception as ex:
# MAGIC         return traceback.format_exc()
# MAGIC $$;
# MAGIC -- let's test our function:
# MAGIC
# MAGIC SELECT execute_python_code("return 'Hello! '* 3") as result;

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM-Based Functions
# MAGIC
# MAGIC LLM-based functions use other language models to perform specific tasks at scale. The `ai_query` function allows you to apply LLM prompts to each row of a table, enabling efficient processing of multiple items. This is useful for:
# MAGIC - Processing large datasets with LLM intelligence
# MAGIC - Generating consistent analyses across multiple items
# MAGIC - Creating personalized recommendations at scale

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Alternative Recommendation
# MAGIC
# MAGIC This function uses an LLM to recommend alternative products based on specific criteria. It processes each product in the database and generates recommendations tailored to the user's needs.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to recommend product alternatives
# MAGIC CREATE OR REPLACE FUNCTION compare_prod(input_product_name STRING, input_reason STRING, input_product_id STRING, input_chemical_formula STRING, input_molecular_weight DOUBLE, input_density DOUBLE, input_melting_point DOUBLE, input_boiling_point DOUBLE, input_application_areas STRING, input_storage_conditions STRING, input_description STRING, input_price_per_unit DOUBLE)
# MAGIC RETURNS TABLE
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Recommends alternative products based on the specified reason. usage: recommend_product_alternatives(product_name, reason, product_id, chemical_formula, molecular_weight, density, melting_point, boiling_point, application_areas, storage_conditions, description, price_per_unit)'
# MAGIC RETURN SELECT ai_query('databricks-meta-llama-3-70b-instruct',
# MAGIC     CONCAT(
# MAGIC       "You are a chemical product specialist. A customer is looking for alternatives to product ", 
# MAGIC       input_product_name, " for the following reason: ", input_reason, "the product has the following specifications: ",
# MAGIC       input_product_id, " ", input_chemical_formula, " ", input_molecular_weight, " ", input_density, " ",  input_melting_point, " ", input_boiling_point, " ", input_application_areas, " ", input_storage_conditions, " ", input_description, " ", input_price_per_unit,
# MAGIC       "Compare the product with the following product ONLY on the given reason:", product_id, " ", product_name," ", chemical_formula, " ", molecular_weight, " ", density, " ",  melting_point, " ", boiling_point, " ", application_areas, " ", storage_conditions, " ", description, " ", price_per_unit
# MAGIC     ) 
# MAGIC   ) AS alternative_option
# MAGIC   FROM dbdemos_a_jack.chem_manufacturing.products
# MAGIC   LIMIT 2

# COMMAND ----------

df = spark.sql("SELECT * FROM dbdemos_a_jack.chem_manufacturing.products WHERE product_name = 'ProCat-Z91'")

pdf = df.toPandas()

input_product_name = pdf['product_name'][0]
input_reason = "to expensive"
input_product_id = pdf['product_id'][0]
input_chemical_formula = pdf['chemical_formula'][0]
input_molecular_weight = pdf['molecular_weight'][0]
input_density = pdf['density'][0]
input_melting_point = pdf['melting_point'][0]
input_boiling_point = pdf['boiling_point'][0]
input_application_areas = pdf['application_areas'][0]
input_storage_conditions = pdf['storage_conditions'][0]
input_description = pdf['description'][0]
input_price_per_unit = pdf['price_per_unit'][0]

query = f"""
SELECT * FROM recommend_product_alternatives(
    '{input_product_name}', 
    '{input_reason}', 
    '{input_product_id}', 
    '{input_chemical_formula}', 
    {input_molecular_weight}, 
    {input_density},
    {input_melting_point},
    {input_boiling_point},
    '{input_application_areas}',
    '{input_storage_conditions}',
    '{input_description}',
    {input_price_per_unit}
)
"""

result_df = spark.sql(query)
display(result_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Using Databricks Playground to test our functions
# MAGIC
# MAGIC Databricks Playground provides a built-in integration with your functions. It'll analyze which functions are available, and call them to properly answer your question.
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/llm-tools-functions/llm-tools-functions-playground.gif?raw=true" style="float: right; margin-left: 10px; margin-bottom: 10px;">
# MAGIC
# MAGIC To try out our functions with playground:
# MAGIC - Open the [Playground](/ml/playground) 
# MAGIC - Select a model supporting tools (like Llama3.1)
# MAGIC - Add the functions you want your model to leverage (`catalog.schema.function_name`)
# MAGIC - Ask a question (for example to convert inch to cm), and playground will do the magic for you!
# MAGIC
# MAGIC <br/>
# MAGIC <div style="background-color: #d4e7ff; padding: 10px; border-radius: 15px;clear:both">
# MAGIC <strong>Note:</strong> Tools in playground is in preview, reach-out your Databricks Account team for more details and to enable it.
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Building an AI system leveraging our Databricks UC functions with Langchain
# MAGIC
# MAGIC These tools can also directly be leveraged on custom model. In this case, you'll be in charge of chaining and calling the functions yourself (the playground does it for you!)
# MAGIC
# MAGIC Langchain makes it easy for you. You can create your own custom AI System using a Langchain model and a list of existing tools (in our case, the tools will be the functions we just created)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable MLflow Tracing
# MAGIC
# MAGIC Enabling MLflow Tracing is required to:
# MAGIC - View the chain's trace visualization in this notebook
# MAGIC - Capture the chain's trace in production via Inference Tables
# MAGIC - Evaluate the chain via the Mosaic AI Evaluation Suite

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start by creating our tools from Unity Catalog
# MAGIC
# MAGIC Let's use UCFunctionToolkit to select which functions we want to use as tool for our demo:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def get_shared_warehouse(name=None):
    w = WorkspaceClient()
    warehouses = w.warehouses.list()

    # Check for warehouse by exact name (if provided)
    if name:
        for wh in warehouses:
            if wh.name == name:
                return wh

    # Define fallback priorities
    fallback_priorities = [
        lambda wh: wh.name.lower() == "serverless starter warehouse",
        lambda wh: wh.name.lower() == "shared endpoint",
        lambda wh: wh.name.lower() == "dbdemos-shared-endpoint",
        lambda wh: "shared" in wh.name.lower(),
        lambda wh: "dbdemos" in wh.name.lower(),
        lambda wh: wh.num_clusters > 0,
    ]

    # Try each fallback condition in order
    for condition in fallback_priorities:
        for wh in warehouses:
            if condition(wh):
                return wh

    # Raise an exception if no warehouse is found
    raise Exception(
        "Couldn't find any Warehouse to use. Please create one first or pass "
        "a specific name as a parameter to the get_shared_warehouse(name='xxx') function."
    )


def display_tools(tools):
    display(pd.DataFrame([{k: str(v) for k, v in vars(tool).items()} for tool in tools]))

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit
import pandas as pd


wh = get_shared_warehouse(name = None) #Get the first shared wh we can. See _resources/01-init for details
print(f'This demo will be using the wg {wh.name} to execute the functions')

def get_tools():
    return (
        UCFunctionToolkit(warehouse_id=wh.id)
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        .include(f"{catalog}.{schema}.*")
        .get_tools())

display_tools(get_tools()) #display in a table the tools - see _resource/00-init for details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's create our langchain agent using the tools we just created

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be availableK. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-70b-instruct"
)

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks

def get_prompt(history = [], prompt = None):
    if not prompt:
            prompt = """You are a helpful chemistry assistant. You are given tools to help you answer questions:
                        - compute_math: to compute matehmatical expressions
                        - convert_chemical_unit: conversion tool for chemical units
                        - execute_python_code: create and run abitrary pyhton code
                        - get_weather: calling weather api to return temperature
                        - find_safety_protocols: text similarity search for safety protocols and research notes
                        - find_similar_products: text similarity search for products. Will return information about products including storage conditions
                        - get_product: get product information with product id
                        - get_reaction_details: get reaction details with product id
                        - get_safety_protocols: get safety protocols with product id
                        - analyze_product_quality: analyze product quality with product id
                        - recommend_product_alternatives: compare products based on specified criteria with all other products 

                        Make sure to use the appropriate tool for each step and provide a coherent response to the user. Don't mention tools to your users. Only answer what the user is asking for. If the question isn't related to the tools or style/clothe, say you're sorry but can't answer.
                """
    return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
    ])

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
prompt = get_prompt()
tools = get_tools()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's give it a try: Asking to run a simple unit conversion.
# MAGIC Under the hood, we want this to call our Math conversion function:

# COMMAND ----------

#make sure our log is enabled to properly display and trace the call chain 
import mlflow

mlflow.langchain.autolog()
agent_executor.invoke({"input": "what is 1kg in gramms?"})

# COMMAND ----------

agent_executor.invoke({"input": "what is (2+2)*2?"})

# COMMAND ----------

agent_executor.invoke({"input": "get a product similar to Synth for paper industry"})

# COMMAND ----------

agent_executor.invoke({"input": "I need a price for 855g of the product with name SynthChem C402"})

# COMMAND ----------

agent_executor.invoke({"input": "Will i be able to store SynthChem C402 savely outside tomorrow?"})

# COMMAND ----------


