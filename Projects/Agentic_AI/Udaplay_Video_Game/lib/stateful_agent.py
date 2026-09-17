from typing import TypedDict, List, Optional
import json

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run, Resource
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory, LongTermMemory, MemoryFragment


class StatefulAgentState(TypedDict):
    user_query: str
    instructions: str
    messages: List[dict]
    current_tool_calls: Optional[List[ToolCall]]
    total_tokens: int
    session_id: str
    owner: str            # identifies the person for long-term memory, separate
                           # from session_id (a session is one conversation; an
                           # owner persists across many sessions)
    memory_context: str    # relevant long-term memories recalled for this query


class StatefulAgent:
    """
    Agent variant addressing two things the base Agent (lib/agents.py) doesn't do:

    1. Each tool is its own named Step in the state machine graph (a
       "pre-defined node"), rather than one generic tool_executor step that
       looks up any tool by name at runtime. This makes the graph explicit
       and inspectable per tool, at the cost of only being able to execute
       one tool call per LLM turn (see the parallel_tool_calls note below).

    2. Long-term memory: before each LLM call, relevant memories about the
       `owner` are recalled from a LongTermMemory store and injected as
       context; after a final answer (no more tool calls), a memory
       fragment summarizing the exchange is registered back into it. This
       is separate from ShortTermMemory, which is still used for
       within-session conversation continuity, same as in the base Agent.

    Design constraint: state_machine.py's StateMachine.run() raises
    NotImplementedError if a transition resolves to more than one target,
    so this graph can only route to one tool node per LLM turn. OpenAI's
    API can return several simultaneous tool calls, which would break that
    routing - so llm_step calls the LLM with parallel_tool_calls=False
    (an extension added to lib/llm.py for this purpose) to guarantee at
    most one tool call per turn. The trade-off: the agent goes strictly
    sequential (retrieve -> evaluate -> maybe search -> answer, one step at
    a time) rather than batching multiple tool calls per turn like the
    base Agent's tool_executor does.
    """

    def __init__(self,
                 model_name: str,
                 instructions: str,
                 tools: List[Tool],
                 long_term_memory: LongTermMemory,
                 temperature: float = 0.7):
        self.instructions = instructions
        self.tools = tools or []
        self.model_name = model_name
        self.temperature = temperature

        self.memory = ShortTermMemory()  # short-term: within-session message history
        self.resource = Resource(vars={"long_term_memory": long_term_memory})
        self.workflow = self._create_state_machine()

    # --- steps -------------------------------------------------------

    def _memory_recall_step(self, state: StatefulAgentState, resource: Resource) -> StatefulAgentState:
        ltm: LongTermMemory = resource.vars["long_term_memory"]
        memory_context = ""
        try:
            result = ltm.search(query_text=state["user_query"], owner=state["owner"], limit=3)
            if result.fragments:
                memory_context = "\n".join(f"- {f.content}" for f in result.fragments)
        except Exception:
            # A fresh/empty long-term memory store, or a query against it,
            # shouldn't take the whole agent down - just proceed with no context.
            pass
        return {"memory_context": memory_context}

    def _message_prep_step(self, state: StatefulAgentState) -> StatefulAgentState:
        messages = state.get("messages", [])
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
        if state.get("memory_context"):
            messages = messages + [SystemMessage(
                content=f"Relevant memory about this user:\n{state['memory_context']}"
            )]
        messages = messages + [UserMessage(content=state["user_query"])]
        return {"messages": messages}

    def _llm_step(self, state: StatefulAgentState) -> StatefulAgentState:
        llm = LLM(model=self.model_name, temperature=self.temperature, tools=self.tools)
        response = llm.invoke(state["messages"], parallel_tool_calls=False)
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "total_tokens": current_total,
        }

    def _make_tool_step(self, tool_obj: Tool) -> Step[StatefulAgentState]:
        def node_logic(state: StatefulAgentState) -> StatefulAgentState:
            tool_calls = state["current_tool_calls"] or []
            # parallel_tool_calls=False guarantees at most one call here.
            call = tool_calls[0]
            args = json.loads(call.function.arguments)
            result = tool_obj(**args)
            # Our tools already return json.dumps(...) strings, so we can use
            # the result directly as ToolMessage content - no str()+json.dumps()
            # double-encoding like the base Agent's generic _tool_step does.
            tool_message = ToolMessage(content=result, tool_call_id=call.id, name=tool_obj.name)
            return {
                "messages": state["messages"] + [tool_message],
                "current_tool_calls": None,
            }
        return Step[StatefulAgentState](tool_obj.name, node_logic)

    def _memory_register_step(self, state: StatefulAgentState, resource: Resource) -> StatefulAgentState:
        ltm: LongTermMemory = resource.vars["long_term_memory"]
        final_content = None
        for m in reversed(state["messages"]):
            if getattr(m, "role", None) == "assistant" and m.content:
                final_content = m.content
                break
        if final_content:
            fragment = MemoryFragment(
                content=f"User asked: {state['user_query']}\nAssistant answered: {final_content}",
                owner=state["owner"],
            )
            try:
                ltm.register(fragment)
            except Exception:
                pass  # don't let a memory-write failure lose the answer already produced
        return {}

    # --- graph ---------------------------------------------------------

    def _create_state_machine(self) -> StateMachine[StatefulAgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[StatefulAgentState](StatefulAgentState)

        # Create steps
        entry = EntryPoint[StatefulAgentState]()
        memory_recall = Step[StatefulAgentState]("memory_recall", self._memory_recall_step)
        message_prep = Step[StatefulAgentState]("message_prep", self._message_prep_step)
        llm_processor = Step[StatefulAgentState]("llm_processor", self._llm_step)
        memory_register = Step[StatefulAgentState]("memory_register", self._memory_register_step)
        termination = Termination[StatefulAgentState]()

        tool_steps = {t.name: self._make_tool_step(t) for t in self.tools}

        machine.add_steps(
            [
                entry, memory_recall,
                message_prep,
                llm_processor,
                memory_register,
                termination
            ] + list(tool_steps.values())
        )
        # Add transitions
        machine.connect(entry, memory_recall)
        machine.connect(memory_recall, message_prep)
        machine.connect(message_prep, llm_processor)

        def route_after_llm(state: StatefulAgentState):
            tool_calls = state.get("current_tool_calls")
            if not tool_calls:
                return memory_register
            name = tool_calls[0].function.name
            return tool_steps.get(name, memory_register)  # unknown tool name -> safe fallback

        machine.connect(llm_processor, [memory_register] + list(tool_steps.values()), route_after_llm)

        for step in tool_steps.values():
            machine.connect(step, llm_processor)

        machine.connect(memory_register, termination)

        return machine

    # --- public API ------------------------------------------------------

    def invoke(self, query: str, session_id: Optional[str] = None, owner: str = "default_user") -> Run:
        """
        Run the agent on a query

        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)
            
        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: StatefulAgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
            "owner": owner,
            "total_tokens": 0,
            "memory_context": "",
        }

        run_object = self.workflow.run(initial_state, resource=self.resource)

        # Store the complete run object in memory
        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session
        
        Args:
            session_id: Optional session ID (uses "default" if None)
            
        Returns:
            List of Run objects in the session
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session

        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)
