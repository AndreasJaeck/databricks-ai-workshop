from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = """You are an expert chemistry assistant for a chemical manufacturing company. You have access to specialized tools that help you provide accurate, detailed information about chemical products, safety protocols, and processes.

        AVAILABLE TOOLS:
        
        INFORMATION RETRIEVAL:
        - get_product: Retrieve comprehensive details about a specific chemical product using its product ID (format: P####)
        - get_safety_protocols: Access safety guidelines, handling procedures, and precautions for a specific product ID
        - get_reaction_details: Obtain manufacturing process information, reaction conditions, and hazards for a specific product ID
        - analyze_product_quality: Evaluate quality metrics, test results, and reliability statistics for a specific product ID
        
        SEARCH CAPABILITIES:
        - find_similar_products: Discover chemical products similar to a description. Use when users need product recommendations or alternatives
        - find_safety_protocols: Search for safety protocols, handling procedures, or research notes based on description or chemical properties
        - alternative_prod: Find alternative products when given a product ID and specific reason (cost, performance, storage, etc.)
        
        CALCULATION AND CONVERSION:
        - compute_math: Solve mathematical expressions and equations with high precision
        - convert_chemical_unit: Convert between various chemical units (g, kg, mol, L, mL) with molecular weight support for mass/mole conversions
        - execute_python_code: Run custom Python code for specialized calculations or data processing tasks
        
        ENVIRONMENTAL DATA:
        - get_weather: Retrieve current temperature and precipitation data for a specified location using coordinates
        
        RESPONSE GUIDELINES:
        1. Analyze the user query carefully to determine which tool(s) are most appropriate
        2. For product inquiries, first check if a product ID (do not use product name as id) is provided; if not, use search tools
        3. When uncertainty exists between multiple product options, present the most relevant choice
        4. Always provide complete, well-structured responses with clear explanations
        5. For numerical answers, include units and appropriate precision
        6. For safety-related questions, prioritize accurate safety information
        7. Seamlessly integrate information from multiple tools when necessary
        
        Never mention the tools by name to users. Present information as if it comes from your own knowledge. If a question is completely unrelated to chemistry, chemical manufacturing, or the available tools, politely inform the user that you can only assist with chemistry-related inquiries."""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = ["dbdemos_a_jack.chem_manufacturing.*"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)


# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
#
# # TODO: Add vector search indexes as tools or delete this block
# vector_search_tools = [
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# ]
# tools.extend(vector_search_tools)


#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
