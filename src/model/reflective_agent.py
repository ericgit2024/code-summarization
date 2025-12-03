from typing import TypedDict, List, Dict, Any, Optional
import logging
import json
import re
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    function_name: str
    code: str
    context: str
    summary: str
    critique: str
    missing_deps: List[str]
    consulted_functions: List[str]
    attempts: int
    max_attempts: int
    metadata: Dict[str, Any]
    action: str

class ReflectiveAgent:
    def __init__(self, inference_pipeline):
        self.pipeline = inference_pipeline
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("generate", self.generate_summary)
        workflow.add_node("critique", self.critique_summary)
        workflow.add_node("decide", self.decide_action)
        workflow.add_node("consult", self.consult_context)
        workflow.add_node("refine", self.refine_summary)

        # Define edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "critique")
        workflow.add_edge("critique", "decide")
        
        # Conditional edges from 'decide'
        workflow.add_conditional_edges(
            "decide",
            self.route_action,
            {
                "consult": "consult",
                "refine": "refine",
                "finish": END
            }
        )
        
        workflow.add_edge("consult", "refine")
        workflow.add_edge("refine", "critique")

        return workflow.compile()

    def generate_summary(self, state: AgentState):
        logger.info(f"Generating initial summary for {state['function_name']}...")
        
        # Use the detailed instruction consistent with the inference pipeline
        instruction = (
             "Provide a comprehensive, detailed, paragraph-like explanation of the code's functionality. "
             "Avoid brief bullet points; instead, write a cohesive narrative that breaks down the logic step-by-step. "
             "Explain the purpose of inputs, the flow of operations, and the role of outputs in depth. "
             "Crucially, integrate the 'Dependency Context' into the narrative, detailing how the function interacts with external calls "
             "(e.g., 'It validates credentials by calling `authenticate`, which checks the database...'). "
             "Ensure the summary is thorough, covering all key aspects of the implementation with high granularity."
        )
        
        summary = self.pipeline.generate_from_code(
            code=state['code'],
            metadata=state['metadata'],
            repo_context=state['context'],
            instruction=instruction
        )
        
        return {"summary": summary, "attempts": state.get("attempts", 0) + 1}

    def critique_summary(self, state: AgentState):
        logger.info("Critiquing summary with LLM...")
        
        prompt = (
            f"### Instruction\n"
            f"You are a senior code reviewer. Analyze the summary below against the provided code.\n"
            f"Identify if the summary misses important function calls, logic, or context.\n"
            f"Specifically list any function names mentioned in the code but missing or unexplained in the summary.\n\n"
            f"### Code\n```python\n{state['code']}\n```\n\n"
            f"### Current Summary\n{state['summary']}\n\n"
            f"### Response Format\n"
            f"Return a JSON object with:\n"
            f"- \"score\": (1-10)\n"
            f"- \"feedback\": \"concise critique string\"\n"
            f"- \"missing_deps\": [\"func1\", \"func2\"] (list of function names that need more explanation)\n\n"
            f"Example:\n"
            f'{{"score": 8, "feedback": "Good summary but misses the db connection part.", "missing_deps": ["connect_db"]}}\n'
            f"Do NOT output markdown formatting like ```json. JUST the JSON string."
        )
        
        response = self.pipeline.generate_response(prompt)
        
        # Parse JSON
        try:
            # Clean up potential markdown
            cleaned = response.replace("```json", "").replace("```", "").strip()
            # Find JSON object if there's extra text
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
            data = json.loads(cleaned)
            critique = data.get("feedback", "No feedback provided.")
            missing = data.get("missing_deps", [])
            score = data.get("score", 5)
        except:
            logger.warning(f"Failed to parse critique JSON. Response start: {response[:100]}...")
            # Fallback: Don't use the garbage response as critique.
            critique = "The summary is missing key details and needs to cover more dependencies."
            missing = []
            score = 5
            
        logger.info(f"Critique: {critique} (Score: {score})")
        return {"critique": critique, "missing_deps": missing}

    def decide_action(self, state: AgentState):
        logger.info("Deciding next action (Policy Step)...")
        
        if state['attempts'] >= state['max_attempts']:
            logger.info("Max attempts reached. Finishing.")
            return {"action": "finish"}

        # Heuristic Policy (Simulating RL Policy)
        # If missing deps and we haven't consulted them -> CONSULT
        # If score is low but no specific missing deps -> REFINE
        # If score is high -> FINISH
        
        missing = [d for d in state['missing_deps'] if d not in state.get('consulted_functions', [])]
        
        critique_text = state['critique'].lower()
        if missing:
            logger.info(f"Policy: Found missing dependencies {missing}. Action: CONSULT")
            return {"action": "consult"}
        elif "missing" in critique_text or "unclear" in critique_text or "needs to cover" in critique_text or "more detailed" in critique_text:
             logger.info("Policy: Critique indicates issues. Action: REFINE")
             return {"action": "refine"}
        else:
             logger.info("Policy: Summary looks good. Action: FINISH")
             return {"action": "finish"}

    def consult_context(self, state: AgentState):
        # Identify targets from missing_deps that haven't been consulted
        targets = [d for d in state['missing_deps'] if d not in state.get('consulted_functions', [])]
        
        if not targets:
            # Should not happen if policy is correct, but handle it
            return {"action": "refine"} # Fallback

        logger.info(f"Consulting RepoGraph for: {targets}")
        new_context_lines = []
        
        for target in targets:
            # Look up in graph
            node_data = None
            graph = self.pipeline.repo_graph.graph
            
            # Try exact match
            if target in graph:
                node_data = graph.nodes[target]
            else:
                # Try suffix match
                for n in graph.nodes():
                    if n.endswith(f".{target}") or n == target:
                        node_data = graph.nodes[n]
                        break
            
            if node_data:
                doc = node_data.get("docstring", "No docstring")
                sig = node_data.get("metadata", {}).get("args", [])
                args = ", ".join([a['name'] for a in sig])
                new_context_lines.append(f"  - Function '{target}': {doc}")
                new_context_lines.append(f"    Signature: def {target}({args})")
            else:
                new_context_lines.append(f"  - Function '{target}': Not found in repository.")
        
        current_context = state['context'] or ""
        if "Additional Context:" not in current_context:
            current_context += "\n\nAdditional Context Retrieved by Agent:"
            
        current_context += "\n" + "\n".join(new_context_lines)
        
        return {
            "context": current_context, 
            "consulted_functions": state.get("consulted_functions", []) + targets
        }

    def refine_summary(self, state: AgentState):
        logger.info("Refining summary with new context...")
        
        prompt = (
            f"### Instruction\n"
            f"Refine the summary based on the critique and any new context provided.\n"
            f"Critique: {state['critique']}\n\n"
            f"### Code\n```python\n{state['code']}\n```\n\n"
            f"### Updated Context\n{state['context']}\n\n"
            f"### Previous Summary\n{state['summary']}\n\n"
            f"### Refined Summary\n"
            f"Write ONLY the new summary in plain English. Do NOT output code."
        )
        
        summary = self.pipeline.generate_response(prompt)
        return {"summary": summary, "attempts": state['attempts'] + 1}

    def route_action(self, state: AgentState):
        return state.get("action", "finish")

    def run(self, function_name, code, context, metadata, max_attempts=3):
        initial_state = {
            "function_name": function_name,
            "code": code,
            "context": context,
            "summary": "",
            "critique": "",
            "missing_deps": [],
            "consulted_functions": [],
            "attempts": 0,
            "max_attempts": max_attempts,
            "metadata": metadata,
            "action": "start"
        }
        
        final_state = self.workflow.invoke(initial_state)
        return final_state["summary"]
