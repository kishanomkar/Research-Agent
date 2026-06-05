# Building a Powerful Research Agent with Watsonx.ai

## Introduction

In today's digital era, the ability to efficiently research complex topics is invaluable. Imagine an AI-powered assistant that not only gathers information but also synthesizes it into a structured report while iterating on its work based on feedback. This is the power of a research agent, and in this post, we'll explore how to build one using IBM's cutting-edge **Watsonx.ai** platform.

## What is Watsonx.ai?

Watsonx.ai is IBM's enterprise-grade AI studio designed to help developers build, train, and deploy AI models. It simplifies the AI lifecycle, from data preparation to model deployment and monitoring, and provides a selection of foundation models for various AI applications.

## Why Build a Research Agent?

A research agent automates the process of gathering, analyzing, and summarizing information. This provides several key benefits:

- **Time Savings**: Automates hours of manual research.
- **Improved Accuracy**: Reduces human error in information gathering.
- **Enhanced Productivity**: Frees up time for higher-level analysis and decision-making.
- **Insight Generation**: Helps uncover hidden patterns and connections in data.

## Core Components of the Research Agent

To build our research agent, we'll use the following technologies:

- **Watsonx.ai LLMs**: We’ll use the `meta-llama/llama-3-70b-instruct` model for planning, writing, reviewing, and critiquing research.
- **LangChain & LangGraph**: These libraries enable structured workflow management for our agent, facilitating state transitions and prompt engineering.
- **Tavily**: This search engine provides real-time information retrieval to enhance our research.

## Setting Up the Environment

Before implementing our research agent, we need to set up our development environment properly. Follow these steps to ensure smooth execution:

### Step 1: Install Python 3.12
Ensure you have Python 3.12 installed on your system. Check your version with:
```sh
python3 --version
```
If needed, download it from Python’s official website or install it using a package manager like `pyenv`.

### Step 2: Create a Virtual Environment
To maintain an isolated environment for the project, create a virtual environment:
```sh
python3.12 -m venv .venv
```

### Step 3: Activate the Virtual Environment
- On macOS/Linux:
  ```sh
  source .venv/bin/activate
  ```
- On Windows:
  ```sh
  .venv\Scripts\activate
  ```

### Step 4: Install Dependencies
Create a `requirements.txt` file and add the following dependencies:
```txt
ipykernel
ipython
langsmith
python-dotenv
ibm_watsonx_ai
tavily-python
langchain==0.2.0
langchain-anthropic==0.1.15
langchain-community==0.2.1
langchain-core==0.2.35
langchain-experimental==0.0.59
langchain-ibm==0.1.7
langchain-openai==0.1.8
langchain-text-splitters==0.2.0
langgraph==0.2.14
```
Then install dependencies with:
```sh
pip install -r requirements.txt
```

### Step 5: Configure API Keys
Create a `.env` file in the project directory to store API keys securely:
```txt
WATSONX_API_KEY=your-watsonx-api-key
PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
TAVILY_API_KEY=your-tavily-api-key
```

### Step 6: Launch Jupyter Notebook
Ensure Jupyter Notebook is installed:
```sh
pip install jupyter
```
Then start Jupyter:
```sh
jupyter notebook
```

Okay, I've expanded the blog post section, incorporating the full code and providing more detailed explanations in a blogger-friendly style.

## Building the Research Agent

Let's dive into the core of our agent's construction. We'll break down the code into manageable pieces, explaining each part's role in the overall workflow.

### Defining the Agent's Workflow

Our research agent follows a structured workflow:

1.  **Initial Planning**: The agent generates a high-level research plan.
2.  **Research**: It uses Tavily to gather relevant content from the web.
3.  **Draft Generation**: The LLM synthesizes the gathered information into a first draft.
4.  **Review**: The LLM takes on the role of a reviewer, providing feedback on the draft's quality.
5.  **Critique**: The LLM then critiques the draft, highlighting areas for improvement and suggesting changes.
6.  **Editor Simulation**: Finally, the LLM simulates an editor, deciding whether to accept, reject, or send the draft back for revision.
7.  **Iteration**: If needed, the agent revises the draft based on the feedback and repeats steps 3-6 until the editor is satisfied or the maximum number of revisions is reached.

This iterative process ensures that the final research report is well-structured, informative, and meets a high standard of quality.

### Environment Setup and Configuration

Before we start building, we need to set up our environment and configure the necessary tools:

```python
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from IPython.display import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_ibm import WatsonxLLM
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient
import uuid

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# --- Configuration and Environment Setup ---

load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Watsonx.ai configuration
watsonx_api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("PROJECT_ID")
url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com") # Default URL

# Tavily configuration
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not watsonx_api_key:
    raise ValueError("Please set the WATSONX_API_KEY in your .env file.")
if not project_id:
    raise ValueError("Please set the PROJECT_ID in your .env file.")
if not tavily_api_key:
    raise ValueError("Please set the TAVILY_API_KEY in your .env file.")
```

**Explanation:**

*   We import the required libraries: `os`, `logging`, `dataclasses`, `typing`, `dotenv`, `IPython.display`, `langchain_core`, `langchain_ibm`, `langgraph`, `tavily`, and `uuid`.
*   We load environment variables from a `.env` file using `load_dotenv()`. This is where you should store your API keys for Watsonx.ai and Tavily.
*   We set up basic logging to monitor the agent's execution.
*   We define variables to store the API key, project ID, and URL for Watsonx.ai, as well as the API key for Tavily.
*   We include error handling to ensure that all necessary API keys are set.

### Initializing Watsonx.ai LLM

Now, let's initialize the core of our agent, the Watsonx.ai Large Language Model (LLM):

```python
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.7,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}

model = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url=url,
    apikey=watsonx_api_key,
    project_id=project_id,
    params=parameters,
)
```

**Explanation:**

*   We define `parameters` to control the LLM's generation process. Here's a breakdown:
    *   `DECODING_METHOD: DecodingMethods.SAMPLE`: Uses sampling for more creative output.
    *   `MAX_NEW_TOKENS: 500`: Limits the response length to 500 tokens.
    *   `MIN_NEW_TOKENS: 1`: Ensures at least one token is generated.
    *   `TEMPERATURE: 0.7`: Controls the randomness of the output (0.7 is a good balance).
    *   `TOP_K: 50`: Considers the top 50 most likely tokens at each step.
    *   `TOP_P: 1`: Considers all the tokens (no probability threshold).
*   We initialize the `WatsonxLLM` object with the following:
    *   `model_id="meta-llama/llama-3-70b-instruct"`: Specifies the LLM model we'll be using.
    *   `url`, `apikey`, `project_id`: Credentials and endpoint for your Watsonx.ai instance.
    *   `params`: The generation parameters we defined above.

### Defining the Agent's State

To manage the agent's workflow effectively, we need to define its state, which represents all the information the agent needs to track during the research process:

```python
class AgentState(TypedDict):
    """
    A dictionary representing the state of the research agent.

    Attributes:
        task (str): The description of the task to be performed.
        plan (str): The research plan generated for the task.
        draft (str): The current draft of the research report.
        critique (str): The critique received for the draft.
        content (List[str]): A list of content gathered during research.
        editor_comment (str): Comments from the editor
        revision_number (int): The current revision number of the draft.
        max_revisions (int): The maximum number of revisions allowed.
        finalized_state (bool): Indicates whether the report is finalized.
    """

    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    editor_comment: str
    revision_number: int
    max_revisions: int
    finalized_state: bool
```

**Explanation:**

*   We define a class `AgentState` using `TypedDict` to create a dictionary with specific keys and their expected data types.
*   The state includes:
    *   `task`: The research topic.
    *   `plan`: The generated research plan.
    *   `draft`: The current draft of the report.
    *   `critique`: Feedback from the critique stage.
    *   `content`: Information gathered from Tavily.
    *   `editor_comment`: Feedback from the editor.
    *   `revision_number`: Tracks the current revision iteration.
    *   `max_revisions`: The maximum allowed revisions.
    *   `finalized_state`: Indicates if the report is finalized (accepted or rejected).

### Defining Prompts

Prompts are essential for guiding the LLM's behavior. Let's define the prompts we'll use for each stage of the workflow:

```python
@dataclass
class ResearchPlanPrompt:
    system_template: str = """
    You are an expert writer tasked with creating a high-level outline for a research report.
    Write such an outline for the user-provided topic. Include relevant notes or instructions for each section.
    The style of the research report should be geared towards the educated public. It should be detailed enough to provide
    a good level of understanding of the topic, but not unnecessarily dense. Think of it more like a whitepaper to be consumed
    by a business leader rather than an academic journal article.
    """

@dataclass
class GenerationPrompt:
    system_template: str = """
    You are a researcher tasked with creating a research report draft based on a user-provided plan.
    Use the provided content to create a draft of the research report.
    The style of the research report should be geared towards the educated public. It should be detailed enough to provide
    a good level of understanding of the topic, but not unnecessarily dense. Think of it more like a whitepaper to be consumed
    by a business leader rather than an academic journal article.
    """

@dataclass
class ReviewPrompt:
    system_template: str = """
    You are a reviewer tasked with reviewing a research report draft based on a user-provided plan.
    Review the provided draft and provide feedback on how to improve it.
    """

@dataclass
class CritiquePrompt:
    system_template: str = """
    You are a critic tasked with critiquing a research report draft based on a user-provided plan.
    Review the provided draft and provide a critique of the draft.
    """

@dataclass
class EditorPrompt:
    system_template: str = """
    You are an editor tasked with reviewing a research report draft based on a user-provided plan.
    Review the provided draft and provide a critique of the draft.
    Based on the critique, decide whether the draft should be accepted, rejected, or sent for revision.
    If the draft is accepted, set the finalized_state to True.
    If the draft is rejected, set the finalized_state to True.
    If the draft is sent for revision, increment the revision_number by 1.
    """

@dataclass
class RejectPrompt:
    system_template: str = """
    The research report draft has been rejected.
    """

@dataclass
class AcceptPrompt:
    system_template: str = """
    The research report draft has been accepted.
    """
```
**Explanation:**
- Each prompt is defined as a dataclass with a `system_template` attribute.
- The system template provides instructions to the LLM, shaping its role and the desired output for each stage (planning, generation, review, critique, editing, acceptance, rejection).

### Creating Agent Nodes

Nodes represent the individual actions or steps within our agent's workflow. Let's define the `AgentNodes` class:

```python
class AgentNodes:
    def __init__(self, model: WatsonxLLM, searcher: TavilyClient):
        self.model = model
        self.searcher = searcher

    def plan_node(self, state: AgentState) -> Dict[str, str]:
        """
        Generate a research plan based on the current state.
        """
        try:
            messages = [
                SystemMessage(content=ResearchPlanPrompt.system_template),
                HumanMessage(content=state["task"]),
            ]
            response = self.model.invoke(messages)
            return {"plan": response}
        except Exception as e:
            logger.error(f"Error in plan_node: {e}")
            return {"plan": "Error generating research plan."}

    def research_plan_node(self, state: AgentState) -> Dict[str, List[str]]:
        """
        Perform research based on the generated plan.
        """
        try:
            plan = state["plan"]
            queries = plan.split("\n")
            content = []
            for query in queries:
                search_result = self.searcher.search(query=query, search_depth="advanced")
                if search_result and search_result.results:
                    for result in search_result.results:
                        content.append(result.content)
                else:
                    logger.warning(f"No search results found for query: {query}")

            return {"content": content}
        except Exception as e:
            logger.error(f"Error in research_plan_node: {e}")
            return {"content": []}

    def generation_node(self, state: AgentState) -> Dict[str, str]:
        """
        Generate a draft based on the research content and plan.
        """
        try:
            messages = [
                SystemMessage(content=GenerationPrompt.system_template),
                HumanMessage(
                    content=f"Plan: {state['plan']}\n\nContent: {state['content']}"
                ),
            ]
            response = self.model.invoke(messages)
            return {"draft": response}
        except Exception as e:
            logger.error(f"Error in generation_node: {e}")
            return {"draft": "Error generating draft."}

    def review_node(self, state: AgentState) -> Dict[str, str]:
        """
        Review the generated draft.
        """
        try:
            messages = [
                SystemMessage(content=ReviewPrompt.system_template),
                HumanMessage(
                    content=f"Plan: {state['plan']}\n\nDraft: {state['draft']}"
                ),
            ]
            response = self.model.invoke(messages)
            return {"review": response}
        except Exception as e:
            logger.error(f"Error in review_node: {e}")
            return {"review": "Error reviewing draft."}

    def research_critique_node(self, state: AgentState) -> Dict[str, str]:
        """
        Critique the generated draft and research content.
        """
        try:
            messages = [
                SystemMessage(content=CritiquePrompt.system_template),
                HumanMessage(
                    content=f"Plan: {state['plan']}\n\nDraft: {state['draft']}\n\nContent: {state['content']}"
                ),
            ]
            response = self.model.invoke(messages)
            return {"critique": response}
        except Exception as e:
            logger.error(f"Error in research_critique_node: {e}")
            return {"critique": "Error critiquing draft."}

    def editor_node(self, state: AgentState) -> Dict[str, str]:
        """
        Simulate an editor reviewing and making decisions about the draft.
        """
        try:
            messages = [
                SystemMessage(content=EditorPrompt.system_template),
                HumanMessage(
                    content=f"Plan: {state['plan']}\n\nDraft: {state['draft']}\n\nCritique: {state['critique']}"
                ),
            ]
            response = self.model.invoke(messages)

            # Determine the next state based on the editor's response
            if "accept" in response.lower():
                finalized_state = True
                next_state = "accepted"
            elif "reject" in response.lower():
                finalized_state = True
                next_state = "rejected"
            else:
                finalized_state = False
                next_state = "to_review"
                state["revision_number"] += 1

            state["finalized_state"] = finalized_state
            state["editor_comment"] = response

            return {
                "editor_comment": response,
                "finalized_state": finalized_state,
                "next_state": next_state
            }

        except Exception as e:
            logger.error(f"Error in editor_node: {e}")
            return {
                "editor_comment": "Error in editor review.",
                "finalized_state": False,  # Default to False on error
                "next_state": "to_review"  # Default to sending for review on error
            }

    def reject_node(self, state: AgentState) -> Dict[str, str]:
        """
        Handle the rejection of the draft.
        """
        try:
            messages = [
                SystemMessage(content=RejectPrompt.system_template),
            ]
            response = self.model.invoke(messages)
            return {"reject_message": response}
        except Exception as e:
            logger.error(f"Error in reject_node: {e}")
            return {"reject_message": "Draft rejected."}

    def accept_node(self, state: AgentState) -> Dict[str, str]:
        """
        Handle the acceptance of the draft.
        """
        try:
            messages = [
                SystemMessage(content=AcceptPrompt.system_template),
            ]
            response = self.model.invoke(messages)
            return {"accept_message": response}
        except Exception as e:
            logger.error(f"Error in accept_node: {e}")
            return {"accept_message": "Draft accepted."}
```
**Explanation:**

*   `AgentNodes` class: Encapsulates all the node functions. It takes the Watsonx LLM (`model`) and the Tavily search client (`searcher`) as input during initialization.
*   `plan_node()`: Generates a research plan based on the `task` in the `state`.
*   `research_plan_node()`: Uses Tavily to perform web searches based on the generated plan and gathers content.
*   `generation_node()`: Creates a draft report using the plan and the gathered content.
*   `review_node()`: Reviews the draft and provides feedback.
*   `research_critique_node()`:** Critiques the draft and the research content, suggesting improvements. It uses the `CritiquePrompt` to guide the LLM.
*   `editor_node()`:** Simulates an editor who reviews the draft and the critique. It makes a decision to "accept," "reject," or send the draft back for revision ("to_review"). It updates the `finalized_state` and `revision_number` accordingly.
*   `reject_node()`:** Handles the case where the draft is rejected.
*   `accept_node()`:** Handles the case where the draft is accepted.
*   Each node function uses `try-except` blocks to handle potential errors during LLM invocation.

### Defining Agent Edges

Edges determine the flow of execution between nodes. We define an `AgentEdges` class to manage the transitions:

```python
class AgentEdges:
    def should_continue(self, state: AgentState) -> str:
        """
        Determine whether the research process should continue based on the current state.
        """
        next_state = state.get("next_state", "to_review")

        if next_state == "accepted":
            return "accepted"
        elif next_state == "rejected":
            return "rejected"
        elif state["revision_number"] > state["max_revisions"]:
            logger.info("Revision number > max allowed revisions")
            return "rejected"
        else:
            return "to_review"
```

**Explanation:**

*   `AgentEdges` class: Contains the logic for conditional transitions.
*   `should_continue()`: This function determines the next node to execute based on the current `state`.
    *   It checks the `next_state` determined by the `editor_node`.
    *   If `next_state` is "accepted" or "rejected" or if the `revision_number` exceeds `max_revisions`, it returns the corresponding state, ending the workflow.
    *   Otherwise, it returns "to_review", continuing the iteration.

### Constructing the Research Agent with LangGraph

Now, we'll use LangGraph to assemble the nodes and edges into a state graph, creating our research agent:

```python
searcher = TavilyClient(api_key=tavily_api_key)
nodes = AgentNodes(model, searcher)
edges = AgentEdges()

agent = StateGraph(AgentState)

# Add nodes
agent.add_node("initial_plan", nodes.plan_node)
agent.add_node("do_research", nodes.research_plan_node)
agent.add_node("write", nodes.generation_node)
agent.add_node("review", nodes.review_node)
agent.add_node("research_revise", nodes.research_critique_node)
agent.add_node("editor", nodes.editor_node)
agent.add_node("reject", nodes.reject_node)
agent.add_node("accept", nodes.accept_node)


# Add edges
agent.set_entry_point("initial_plan")
agent.add_edge("initial_plan", "do_research")
agent.add_edge("do_research", "write")
agent.add_edge("write", "editor")
agent.add_conditional_edges(
    "editor",
    edges.should_continue,
    {
        "accepted": "accept",
        "rejected": "reject",
        "to_review": "review"
    },
)
agent.add_edge("review", "research_revise")
agent.add_edge("research_revise", "write")
agent.add_edge("reject", END)
agent.add_edge("accept", END)
```

**Explanation:**

*   We initialize the `TavilyClient`, `AgentNodes`, and `AgentEdges`.
*   `agent = StateGraph(AgentState)`: We create a `StateGraph` object, using `AgentState` to define the structure of the state.
*   `agent.add_node(...)`: We add each node to the graph, associating the node name (e.g., "initial_plan") with the corresponding function from `AgentNodes`.
*   `agent.set_entry_point("initial_plan")`: We set the "initial_plan" node as the starting point of the workflow.
*   `agent.add_edge(...)`: We define the standard flow using directed edges: "initial_plan" -> "do_research" -> "write" -> "editor".
*   `agent.add_conditional_edges(...)`: We add conditional edges from the "editor" node:
    *   Based on the `should_continue` function's output, the flow goes to "accept", "reject", or "review".
*   We add edges from "review" back to "research_revise" and then "write" to continue the cycle.
*   `agent.add_edge("reject", END)` and `agent.add_edge("accept", END)`: We connect the "reject" and "accept" nodes to the `END` node, signifying the termination of the workflow.

### Compiling and Running the Agent

Finally, we compile the graph and define a function to execute the research task:

```python
# Compile the graph
checkpointer = MemorySaver()
in_memory_store = InMemoryStore()
graph = agent.compile(checkpointer=checkpointer, store=in_memory_store)

# Visualize the graph
try:
    Image(graph.get_graph().draw_png())
except:
    print("Unable to display graph architecture. Ensure that you have installed the required dependencies.")

def execute_research_task(
    task_description: str, max_revisions: int = 3
) -> List[Dict]:
    """
    Executes the research task workflow.

    Args:
        task_description (str): The description of the research task.
        max_revisions (int): The maximum number of revisions allowed.

    Returns:
        List[Dict]: A list of dictionaries representing the updates from each step of the workflow.
    """

    results = []

    user_id = "1" # You can make this dynamic if needed
    config = {"configurable": {"thread_id": "1", "user_id": user_id}}
    namespace = (user_id, "memories")

    try:
        for i, update in enumerate(
            graph.stream(
                {
                    "task": task_description,
                    "max_revisions": max_revisions,
                    "revision_number": 0,
                    "finalized_state": False
                },
                config,
                stream_mode="updates",
            )
        ):
            print(f"--- Update {i+1} ---")
            print(update)
            memory_id = str(uuid.uuid4())
            in_memory_store.put(namespace, memory_id, {"memory": update})
            results.append(update)
        print("--- Workflow Complete ---")
    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {e}")
    return results
```

**Explanation:**

*   We are using checkpointer for checkpointing and in\_memory\_store for storing intermediate results.
*   We provide an option to visualize the graph using `graph.get_graph().draw_png()`.
*   `execute_research_task()` function:
    *   Takes the `task_description` and `max_revisions` as input.
    *   Initializes an empty list `results` to store the updates from each step.
    *   Uses `graph.stream()` to execute the workflow, passing the initial state and configuration.
        *   `stream_mode="updates"` ensures we get updates after each node execution.
    *   Iterates through the updates, prints them, stores them using `in_memory_store`, and appends them to the `results` list.
    *   Handles potential errors during execution.
    *   Returns the `results` list.

### Running the Agent

```python
task_description = "What are the key trends in LLM research and application that you see in 2024"
execute_research_task(task_description, max_revisions=1)
```
**Explanation:**
* We define the task description
* We run the agent by calling `execute_research_task` passing the `task_description` and `max_revisions`

## Conclusion

Building a research agent using Watsonx.ai, LangChain, LangGraph, and Tavily provides a robust foundation for AI-assisted research. This agent can be customized further with additional features such as:

Fine-tuning models for domain-specific research

Enhancing evaluation metrics for quality assessment

Integrating with knowledge graphs and visualization tools

By leveraging AI, you can streamline research workflows and unlock new insights faster than ever. Happy building!