# Multi-Tool Agent Workshop

This workshop guides you through building a gen AI agent with specialized tool functions for a chemical manufacturing use case. You'll learn how to design tool functions, and use them with LLMs to create intelligent agents that can answer domain-specific questions.

## Prerequisites

Before starting the workshop, ensure you have:

- **Databricks Environment**: 
  - DBR 16.0 LTS (or newer)
  - Personal compute with single node
  - Permissions to create catalogs and schemas
  - Access to Databricks AI Playground
  - Ability to create and use serving endpoints
  - Access to Serverless DWH

- **Python Knowledge**:
  - Basic understanding of Python and SQL
  - Familiarity with Databricks notebooks

## Workshop Structure

The workshop consists of two main notebooks:

1. `01-data-create.ipynb`: Creates synthetic chemical manufacturing data
2. `02-tool-functions.py`: Builds tool functions and agents that interact with the data

## Step 1: Setup Environment

1. Create a cluster with the following configuration:
   - Databricks Runtime: 16.0 LTS ML (or newer)
   - Node type: Single node (recommended: at least 4 cores, 16 GB RAM)

2. Mount the the following notebooks to your cluster. 


## Step 2: Generate Synthetic Data

Run the `01-data-create.ipynb` notebook to:

1. Set up a personalized catalog and schema in Unity Catalog
2. Generate synthetic chemical manufacturing data:
   - Products table: Chemical products with details like formula, properties, applications
   - Batches table: Production batch information
   - Quality Control table: Test results and QC metrics
   - Inventory table: Stock levels of raw materials and finished products
   - Reactions table: Chemical reaction details for manufacturing processes
   - Descriptions table: Text descriptions, safety protocols, and research notes

3. Enable Change Data Feed (CDF) for tables that will be used with vector search

**Note**: This notebook automatically creates a catalog named `dbdemos_[initial]_[lastname]` and a schema called `chem_manufacturing`. The data generation takes a few minutes to complete.

## Step 3: Create Tool Functions

Run the `02-tool-functions.py` notebook to create specialized functions that your AI agent can use:

1. **Vector Search Functions**:
   - `find_similar_products`: Search for products semantically similar to a description
   - `find_safety_protocols`: Find safety protocols based on needs or descriptions

2. **SQL Functions**:
   - `get_product`: Retrieve detailed information about a specific product
   - `get_safety_protocols`: Get safety protocols for a specific product
   - `get_reaction_details`: Get chemical reaction information
   - `analyze_product_quality`: Get quality metrics for a product

3. **Python Functions**:
   - `convert_chemical_unit`: Convert between chemical units
   - `compute_math`: Perform mathematical calculations
   - `get_weather`: Get weather data for a location
   - `execute_python_code`: Execute arbitrary Python code (for controlled environments)

4. **LLM-Based Functions**:
   - `alternative_prod`: Recommend product alternatives based on specific criteria

## Step 4: Test Functions in Playground

1. Open the [Databricks AI Playground](/ml/playground)
2. Select a model that supports tool functions (e.g., Llama 3)
3. Add your functions using the fully qualified name: `catalog.schema.function_name`
4. Ask questions to test your functions, such as:
- Find product with similar name to "EcoSolv".
- What is 2+2?
- Compare product id with other products on price

## Step 5: Build a Langchain Agent

The notebook shows how to:

1. Create a Langchain agent that uses your functions as tools
2. Set up MLflow tracing to monitor your agent's behavior
3. Define a system prompt for your agent to follow
4. Execute complex queries that require multiple tool calls

## Example Queries to Try

- "What is 1kg in grams?"
- "What is (2+2)*2?"
- "Get a product similar to SynthChem for the paper industry"
- "I need a price for 855g of the product with name SynthChem C402"
- "Will I be able to store SynthChem C402 safely outside tomorrow?"
- "Find alternatives to product P0001 that are less expensive"
- "What are the safety protocols for handling catalysts?"

## Troubleshooting

- **Vector Search Issues**: Make sure the vector index exists before calling search functions. Also make sure when wrapping vector index with SQL Tool to pass the Vector Index to the Agent MlFlow logging. 
- **Function Permissions**: Ensure your user has permissions to create functions in the Unity Catalog and can execute on resources. 
- **Serverless Warehouse**: If you encounter issues with running SQL functions, verify that you have access to a serverless SQL warehouse
- **LLM Access**: If agent calls fail, check that your workspace has access to the required LLM models and serving endpoints

## Next Steps

After completing the workshop, you can:

1. Add your own custom functions to the agent
2. Create a web app using the agent API
3. Connect to real data sources instead of synthetic data
4. Deploy the agent to production using Databricks Model Serving

## Resources

- [Databricks Documentation on Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Function Calling Documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-tool)
- [Langchain Documentation](https://python.langchain.com/docs/get_started/introduction)

