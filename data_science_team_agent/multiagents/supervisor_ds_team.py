"""Supervisor agent for coordinating data science team workflow."""

from collections.abc import Sequence
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # type: ignore[import]
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser  # type: ignore[import]
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from langgraph.graph.message import add_messages  # type: ignore[import]

from data_science_team_agent.templates import BaseAgent

TEAM_MAX_MESSAGES = 20
TEAM_MAX_MESSAGE_CHARS = 2000


def _is_agent_output_report_message(m: BaseMessage) -> bool:
    if not isinstance(m, AIMessage):
        return False
    content = getattr(m, "content", None)
    if not isinstance(content, str) or not content:
        return False
    s = content.lstrip()
    if not s.startswith("{"):
        return False
    head = s[:1200]
    return '"report_title"' in head and ("Agent Outputs" in head or "Agent Output Summary" in head)


def _supervisor_merge_messages(
    left: Sequence[BaseMessage] | None,
    right: Sequence[BaseMessage] | None,
) -> list[BaseMessage]:
    merged = add_messages(left or [], right or [])

    cleaned: list[BaseMessage] = []
    for m in merged:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("tool", "function"):
            continue
        if _is_agent_output_report_message(m):
            continue

        content = getattr(m, "content", "")
        message_id = getattr(m, "id", None)

        if isinstance(content, str) and len(content) > TEAM_MAX_MESSAGE_CHARS:
            content = content[:TEAM_MAX_MESSAGE_CHARS] + "\n...[truncated]..."

        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
            continue

        if isinstance(m, AIMessage):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
        elif isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=content or "", id=message_id))
        elif isinstance(m, SystemMessage):
            cleaned.append(SystemMessage(content=content or "", id=message_id))
        else:
            cleaned.append(m)

    return cleaned[-TEAM_MAX_MESSAGES:]


class SupervisorDSTeam(BaseAgent):
    """Supervisor agent for coordinating data science team workflow."""

    def __init__(self, model, agents, checkpointer=None):
        """Initialize the supervisor agent.

        Args:
            model: The language model to use
            agents: List of agents to coordinate
            checkpointer: Optional checkpointer for state management

        """
        self._params = {
            "model": model,
            "agents": agents,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_supervisor_ds_team(**self._params)

    def invoke_agent(self, user_instructions, data=None, max_retries=3, retry_count=0, **kwargs):
        """Invoke the supervisor agent with user instructions.

        Args:
            user_instructions: The user's instructions
            data: Optional data to process
            max_retries: Maximum number of retries
            retry_count: Current retry count
            **kwargs: Additional keyword arguments

        Returns:
            None

        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)],
                "user_instructions": user_instructions,
                "data": data.to_dict() if data is not None else None,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None


def make_supervisor_ds_team(model, agents, checkpointer=None):
    """Create a supervisor data science team workflow graph.

    Args:
        model: The language model to use
        agents: List of agents to coordinate
        checkpointer: Optional checkpointer for state management

    Returns:
        Compiled workflow graph

    """

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], _supervisor_merge_messages]
        user_instructions: str
        data: dict | None
        current_agent: str
        next_action: str
        max_retries: int
        retry_count: int

    # Router prompt
    router_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a supervisor managing a data science team.
        Available agents: {agents}

        Based on the user's request, select the most appropriate agent and action.

        Return JSON with 'agent' and 'action' keys.
        Examples:
        - For data cleaning: {{"agent": "data_cleaning", "action": "clean_data"}}
        - For visualization: {{"agent": "data_visualization", "action": "create_plot"}}
        - For analysis: {{"agent": "pandas_analyst", "action": "analyze_data"}}
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def route_to_agent(state: GraphState):
        print("    * ROUTE TO AGENT")

        agent_list = list(agents.keys())

        router = router_prompt | model | JsonOutputFunctionsParser()

        try:
            decision = router.invoke({"agents": agent_list, "messages": state["messages"]})

            agent_name = decision.get("agent", "data_cleaning")
            action = decision.get("action", "process")

        except Exception as e:
            print(f"Router error: {e}")
            return {"current_agent": "data_cleaning", "next_action": "process"}
        else:
            return {"current_agent": agent_name, "next_action": action}

    def execute_agent_task(state: GraphState):
        print(f"    * EXECUTE {state['current_agent']} TASK")

        agent_name = state["current_agent"]

        if agent_name in agents:
            agent = agents[agent_name]

            try:
                if hasattr(agent, "invoke_agent"):
                    if state.get("data"):
                        agent.invoke_agent(
                            user_instructions=state["user_instructions"], data_raw=pd.DataFrame(state["data"])
                        )
                    else:
                        agent.invoke_agent(user_instructions=state["user_instructions"])
                else:
                    # Handle tool-based agents
                    agent.invoke({"user_instructions": state["user_instructions"], "data": state.get("data", {})})

                response = agent.get_response()
            except Exception as e:
                return {"agent_response": {"error": str(e)}}
            else:
                return {"agent_response": response}

        return {"agent_response": {"error": f"Agent {agent_name} not found"}}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("execute_agent_task", execute_agent_task)
    workflow.add_edge(START, "route_to_agent")
    workflow.add_edge("route_to_agent", "execute_agent_task")
    workflow.add_edge("execute_agent_task", END)

    return workflow.compile()
