from typing import TypedDict, List, Dict, Any
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    function_name: str
    code: str
    context: str
    summary: str
    critique: str
    attempts: int
    max_attempts: int
    metadata: Dict[str, Any]

class ReflectiveAgent:
    def __init__(self, inference_pipeline):
        self.pipeline = inference_pipeline
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("generate", self.generate_summary)
        workflow.add_node("critique", self.critique_summary)
        workflow.add_node("refine", self.refine_summary)

        # Define edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            self.check_critique,
            {
                "good": END,
                "bad": "refine"
            }
        )
        workflow.add_edge("refine", "critique")

        return workflow.compile()

    def generate_summary(self, state: AgentState):
        logger.info(f"Generating initial summary for {state['function_name']}...")
        
        # Use the existing pipeline logic to construct prompt and generate
        # We can reuse construct_hierarchical_prompt but we need to adapt it or just call summarize directly?
        # Calling summarize directly might be recursive if we integrate it there.
        # Let's use the lower-level methods if possible, or just call summarize with a specific instruction.
        
        instruction = "Summarize the code, focusing on its logic and dependencies."
        
        # We need to bypass the 'summarize' method's graph check if we want to use this agent 
        # as the primary driver, OR we can just use the pipeline's internal helpers.
        # For POC, let's just use the pipeline's generate logic.
        
        # Re-construct prompt using pipeline's helper
        # We need to fetch retrieval items again? Or just pass them in state?
        # Let's assume state has context.
        
        # For simplicity in POC, we will use the pipeline's summarize method but with a flag or just reuse logic.
        # But wait, pipeline.summarize does everything.
        # Let's extract the generation logic in inference.py later. 
        # For now, let's assume we can call a method on pipeline that takes a prompt.
        
        # HACK: We will call pipeline.summarize but we need to avoid infinite loop if we call this agent FROM summarize.
        # So we will assume this agent is called explicitly.
        
        summary = self.pipeline.generate_from_code(
            code=state['code'],
            metadata=state['metadata'],
            repo_context=state['context'],
            instruction=instruction
        )
        
        return {"summary": summary, "attempts": state.get("attempts", 0) + 1}

    def critique_summary(self, state: AgentState):
        logger.info("Critiquing summary...")
        summary = state['summary']
        context = state['context']
        
        # Simple heuristic critique: Check if key dependencies from context are mentioned.
        # Context format: "  - Function 'foo' (Relevance: 3.0): ..."
        
        missing = []
        if context and context != "No context found.":
            lines = context.split('\n')
            for line in lines:
                if "Function '" in line:
                    # Extract function name: "  - Function 'foo' ..."
                    try:
                        func_name = line.split("'")[1]
                        if func_name not in summary:
                            missing.append(func_name)
                    except:
                        pass
        
        if missing:
            critique = f"The summary is missing mentions of these dependencies: {', '.join(missing)}. Please include them."
            logger.info(f"Critique failed: {critique}")
            return {"critique": critique}
        else:
            logger.info("Critique passed.")
            return {"critique": "GOOD"}

    def refine_summary(self, state: AgentState):
        logger.info("Refining summary...")
        critique = state['critique']
        instruction = f"Refine the previous summary. {critique}"
        
        summary = self.pipeline.generate_from_code(
            code=state['code'],
            metadata=state['metadata'],
            repo_context=state['context'],
            instruction=instruction
        )
        
        return {"summary": summary, "attempts": state['attempts'] + 1}

    def check_critique(self, state: AgentState):
        if state['critique'] == "GOOD":
            return "good"
        elif state['attempts'] >= state['max_attempts']:
            logger.warning("Max attempts reached. Returning current summary.")
            return "good" # Stop anyway
        else:
            return "bad"

    def run(self, function_name, code, context, metadata, max_attempts=3):
        initial_state = {
            "function_name": function_name,
            "code": code,
            "context": context,
            "summary": "",
            "critique": "",
            "attempts": 0,
            "max_attempts": max_attempts,
            "metadata": metadata
        }
        
        final_state = self.workflow.invoke(initial_state)
        return final_state["summary"]
