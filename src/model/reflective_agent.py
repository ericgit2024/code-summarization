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
             "Provide a comprehensive and structured summary of the code's functionality.\n"
             "The output MUST be organized into the following sections using Markdown headers:\n"
             "1. **Overview**: A high-level explanation of what the code does.\n"
             "2. **Detailed Logic**: A step-by-step breakdown of the operations, inputs, and outputs. **Use the provided 'Structural Analysis' (AST, CFG, PDG) to explain the code's flow:**\n"
             "   - Mention specific conditions and branches identified in the Control Flow Graph (CFG).\n"
             "   - Explain data transformations and dependencies identified in the Program Dependence Graph (PDG).\n"
             "3. **Dependency Analysis**: **CRITICAL REQUIREMENT** - You MUST identify and list ALL function calls made within this code.\n"
             "   - First, scan the code and identify every function call (e.g., service.method(), function_name(), etc.)\n"
             "   - For EACH function call, provide:\n"
             "     a) The function name and how it's called (e.g., 'customer_service.get_customer_by_id(customer_id)')\n"
             "     b) The purpose of this function call based on the 'Dependency Context' or code context\n"
             "     c) The source file if available in the format 'from filename.py' (e.g., 'CustomerService.get_customer_by_id() from customer_service.py')\n"
             "   - If no function calls are present, explicitly state 'This function makes no external function calls.'\n"
             "   - Example format: 'This function calls: 1) get_customer_by_id() from CustomerService to retrieve customer data, 2) get_product_by_id() from ProductService to fetch product details, 3) update_stock() from Product to modify inventory levels.'\n\n"
             "Ensure the content is detailed and thorough."
             "\n\n**CRITICAL NEGATIVE CONSTRAINT**: Do NOT output the AST, CFG, PDG, or any code blocks representing the structural analysis. These are provided for your understanding only. You should use the *information* from them to improve your summary (e.g. 'The control flow indicates...'), but do not copy the raw AST/CFG/PDG text."
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
        
        # Check if summary is an error message - if so, skip critique
        if state['summary'].startswith("Error:"):
            logger.warning("Summary is an error message. Skipping critique.")
            return {
                "critique": "Summary generation failed.",
                "missing_deps": []
            }
        
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
            # Better fallback: If we can't parse, assume the summary is good enough
            critique = "Summary looks acceptable."
            missing = []
            score = 7  # Give it a passing score to avoid infinite refinement
            
        logger.info(f"Critique: {critique} (Score: {score})")
        return {"critique": critique, "missing_deps": missing}

    def decide_action(self, state: AgentState):
        logger.info("Deciding next action (Policy Step)...")
        
        if state['attempts'] >= state['max_attempts']:
            logger.info("Max attempts reached. Finishing.")
            return {"action": "finish"}
        
        # If summary is an error message, finish immediately
        if state['summary'].startswith("Error:"):
            logger.error("Summary is an error message. Finishing workflow.")
            return {"action": "finish"}
        
        # If summary is too short, finish (don't keep trying)
        if len(state['summary']) < 100:
            logger.warning(f"Summary is very short ({len(state['summary'])} chars). Finishing to avoid infinite loop.")
            return {"action": "finish"}

        # Heuristic Policy (Simulating RL Policy)
        # If missing deps and we haven't consulted them -> CONSULT
        # If score is low but no specific missing deps -> REFINE (but only once)
        # If score is high -> FINISH
        
        missing = [d for d in state['missing_deps'] if d not in state.get('consulted_functions', [])]
        
        critique_text = state['critique'].lower()
        if missing:
            logger.info(f"Policy: Found missing dependencies {missing}. Action: CONSULT")
            return {"action": "consult"}
        elif "missing" in critique_text or "unclear" in critique_text or "needs to cover" in critique_text or "more detailed" in critique_text:
             # Only refine once to avoid cascading failures
             if state['attempts'] <= 1:
                 logger.info("Policy: Critique indicates issues. Action: REFINE")
                 return {"action": "refine"}
             else:
                 logger.info("Policy: Already refined once. Accepting current summary. Action: FINISH")
                 return {"action": "finish"}
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
        
        # Simplified prompt to avoid model confusion
        prompt = (
            f"Improve this code summary by addressing the following feedback: {state['critique']}\n\n"
            f"Code:\n```python\n{state['code']}\n```\n\n"
            f"Current Summary:\n{state['summary']}\n\n"
            f"Write an improved summary with these sections:\n"
            f"1. **Overview**: What the code does\n"
            f"2. **Detailed Logic**: Step-by-step explanation\n"
            f"3. **Dependency Analysis**: How it interacts with other functions\n\n"
            f"Improved Summary:"
        )
        
        summary = self.pipeline.generate_response(prompt)
        
        # If refinement fails, keep the previous summary
        if not summary or len(summary) < 50:
            logger.warning("Refinement produced empty/short output. Keeping previous summary.")
            return {"attempts": state['attempts'] + 1}  # Don't update summary
        
        return {"summary": summary, "attempts": state['attempts'] + 1}

    def route_action(self, state: AgentState):
        return state.get("action", "finish")

    def run(self, function_name, code, context, metadata, max_attempts=5):
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
        
        logger.info(f"Starting ReflectiveAgent workflow for function: {function_name} (max_attempts={max_attempts})")
        final_state = self.workflow.invoke(initial_state)
        summary = final_state.get("summary", "")
        
        # Validation
        if not summary or len(summary) < 20:
            logger.error(f"Agent workflow completed but summary is empty or too short: '{summary}'")
            error_msg = (
                "Error: Smart Agent failed to generate a summary.\n"
                "This could be due to:\n"
                "1. Model generation issues (check console logs for 'generate_response' debug output)\n"
                "2. Workflow errors during refinement\n"
                "3. Empty initial summary propagating through the workflow\n\n"
                "Please try normal mode or check the console logs for more details."
            )
            return error_msg
        
        logger.info(f"ReflectiveAgent completed successfully. Summary length: {len(summary)} chars")
        return summary
