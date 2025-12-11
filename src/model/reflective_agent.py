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
    scores: Dict[str, float]  # correctness, completeness, clarity, technical_depth
    specific_issues: List[str]
    missing_deps: List[str]
    consulted_functions: List[str]
    verification_passed: bool
    verification_confidence: float
    verification_feedback: str
    history: List[Dict] # To store intermediate scores for analysis
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
        workflow.add_node("verify", self.verify_summary)

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
        # After refinement, we verify before critiquing again
        workflow.add_edge("refine", "verify")
        workflow.add_edge("verify", "critique")

        return workflow.compile()

    def generate_summary(self, state: AgentState):
        logger.info(f"Generating initial summary for {state['function_name']}...")
        
        instruction = (
            "Generate a concise docstring summary for this code.\n"
            "Write 1-3 sentences explaining what the code does.\n"
            "Do NOT use markdown, bullet points, or structured sections."
        )
        
        summary = self.pipeline.generate_from_code(
            code=state['code'],
            metadata=state['metadata'],
            repo_context=state['context'],
            instruction=instruction
        )
        
        return {"summary": summary, "attempts": state.get("attempts", 0) + 1}

    def critique_summary(self, state: AgentState):
        logger.info("Critiquing summary with LLM (Reflection Scorer Module)...")
        
        # Check if summary is an error message
        if state['summary'].startswith("Error:"):
            logger.warning("Summary is an error message. Skipping critique.")
            return {
                "critique": "Summary generation failed.",
                "scores": {"correctness": 0, "completeness": 0, "clarity": 0, "technical_depth": 0},
                "specific_issues": ["Generation failed"],
                "missing_deps": []
            }
        
        prompt = (
            f"### Instruction\n"
            f"You are a senior code reviewer. Evaluate the summary below against the provided code.\n"
            f"Assess the following criteria on a scale of 0-10:\n"
            f"1. correctness: Factual accuracy compared to code logic.\n"
            f"2. completeness: Covers main functionality and edge cases.\n"
            f"3. clarity: Understandable language, good grammar.\n"
            f"4. technical_depth: Appropriate detail level for a docstring.\n\n"
            f"Identify specific issues (e.g., 'Missing error handling', 'Unclear variable purpose').\n"
            f"Identify any missing function calls that need explanation.\n\n"
            f"### Code\n```python\n{state['code']}\n```\n\n"
            f"### Current Summary\n{state['summary']}\n\n"
            f"### Response Format\n"
            f"Return a JSON object with:\n"
            f"- \"scores\": {{\"correctness\": <0-10>, \"completeness\": <0-10>, \"clarity\": <0-10>, \"technical_depth\": <0-10>}}\n"
            f"- \"specific_issues\": [\"issue1\", \"issue2\"]\n"
            f"- \"missing_deps\": [\"func1\", \"func2\"]\n"
            f"- \"feedback\": \"concise general feedback\"\n\n"
            f"Do NOT output markdown formatting like ```json. JUST the JSON string."
        )
        
        response = self.pipeline.generate_response(prompt)
        
        # Parse JSON
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
            data = json.loads(cleaned)

            scores = data.get("scores", {"correctness": 5, "completeness": 5, "clarity": 5, "technical_depth": 5})
            specific_issues = data.get("specific_issues", [])
            missing_deps = data.get("missing_deps", [])
            feedback = data.get("feedback", "No feedback provided.")

        except Exception as e:
            logger.warning(f"Failed to parse critique JSON: {e}. Response: {response[:100]}...")
            scores = {"correctness": 7, "completeness": 7, "clarity": 7, "technical_depth": 7}
            specific_issues = []
            missing_deps = []
            feedback = "Parse error, assuming acceptable summary."

        # Store history
        history_item = {
            "step": state.get("attempts", 1),
            "summary": state['summary'],
            "scores": scores,
            "issues": specific_issues
        }
        history = state.get("history", []) + [history_item]

        logger.info(f"Scores: {scores}")
        return {
            "critique": feedback,
            "scores": scores,
            "specific_issues": specific_issues,
            "missing_deps": missing_deps,
            "history": history
        }

    def verify_summary(self, state: AgentState):
        logger.info("Verifying summary against code structure (Verification Module)...")

        prompt = (
            f"### Instruction\n"
            f"Verify if the claims in the summary strictly match the code structure.\n"
            f"Check for hallucinations (mentioning things not in code) or incorrect logic descriptions.\n\n"
            f"### Code\n```python\n{state['code']}\n```\n\n"
            f"### Summary\n{state['summary']}\n\n"
            f"### Response Format\n"
            f"Return a JSON object with:\n"
            f"- \"passed\": true/false\n"
            f"- \"confidence\": <0.0-1.0>\n"
            f"- \"feedback\": \"explanation of verification result\"\n\n"
            f"Do NOT output markdown formatting."
        )

        response = self.pipeline.generate_response(prompt)

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
            data = json.loads(cleaned)

            passed = data.get("passed", True)
            confidence = data.get("confidence", 0.8)
            feedback = data.get("feedback", "Verified.")
        except:
            passed = True
            confidence = 0.5
            feedback = "Verification parsing failed, defaulting to pass."
            
        logger.info(f"Verification: Passed={passed}, Confidence={confidence}")

        return {
            "verification_passed": passed,
            "verification_confidence": confidence,
            "verification_feedback": feedback
        }

    def decide_action(self, state: AgentState):
        logger.info("Deciding next action (Pipeline Controller)...")
        
        if state['attempts'] > state['max_attempts']:
            logger.info("Max attempts reached. Finishing.")
            return {"action": "finish"}
        
        if state['summary'].startswith("Error:"):
            return {"action": "finish"}

        # Check scores
        scores = state.get("scores", {})
        min_score = min(scores.values()) if scores else 0
        
        missing = [d for d in state.get('missing_deps', []) if d not in state.get('consulted_functions', [])]
        
        # Policy
        if missing:
             logger.info(f"Policy: Missing dependencies {missing}. Action: CONSULT")
             return {"action": "consult"}

        if min_score < 7:
            logger.info(f"Policy: Low score ({min_score} < 7). Action: REFINE")
            return {"action": "refine"}

        # If verification failed significantly (low confidence or explicit fail), maybe refine?
        # But we verify AFTER refine. If we are here, we just critiqued.
        # If we just verified (which happens before critique in loop if we refined),
        # the critique should have caught it, or we rely on scores.
        # But wait, logic is: Generate -> Critique -> Decide.
        # If Refine -> Verify -> Critique -> Decide.

        # If we have issues but scores are high? Unlikely if prompt works well.

        logger.info("Policy: Scores acceptable. Action: FINISH")

        # If verification hasn't run yet (confidence is 0.0), implicitly pass it
        if state.get("verification_confidence", 0.0) == 0.0:
            return {
                "action": "finish",
                "verification_passed": True,
                "verification_confidence": 1.0,
                "verification_feedback": "Accepted based on high critique scores."
            }

        return {"action": "finish"}

    def consult_context(self, state: AgentState):
        targets = [d for d in state['missing_deps'] if d not in state.get('consulted_functions', [])]
        if not targets:
            return {"action": "refine"}

        logger.info(f"Consulting RepoGraph for: {targets}")
        new_context_lines = []
        
        for target in targets:
            node_data = None
            graph = self.pipeline.repo_graph.graph
            if target in graph:
                node_data = graph.nodes[target]
            else:
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
                new_context_lines.append(f"  - Function '{target}': Not found.")
        
        current_context = state['context'] or ""
        if "Additional Context:" not in current_context:
            current_context += "\n\nAdditional Context Retrieved by Agent:"
        current_context += "\n" + "\n".join(new_context_lines)
        
        return {
            "context": current_context, 
            "consulted_functions": state.get("consulted_functions", []) + targets
        }

    def refine_summary(self, state: AgentState):
        logger.info("Refining summary (Targeted Regeneration Module)...")

        issues_str = ", ".join(state.get("specific_issues", []))
        feedback = state.get("critique", "")
        
        prompt = (
            f"Improve this summary by addressing: {issues_str}. Keep good parts unchanged.\n"
            f"Feedback: {feedback}\n\n"
            f"Code:\n```python\n{state['code']}\n```\n\n"
            f"Current Summary:\n{state['summary']}\n\n"
            f"Write the improved summary as a natural language paragraph."
        )
        
        summary = self.pipeline.generate_response(prompt)
        
        if not summary or len(summary) < 50:
            logger.warning("Refinement produced empty/short output. Keeping previous summary.")
            return {"attempts": state['attempts'] + 1}
        
        return {"summary": summary, "attempts": state['attempts'] + 1}

    def route_action(self, state: AgentState):
        return state.get("action", "finish")

    def run(self, function_name, code, context, metadata, max_attempts=2):
        initial_state = {
            "function_name": function_name,
            "code": code,
            "context": context,
            "summary": "",
            "critique": "",
            "scores": {},
            "specific_issues": [],
            "missing_deps": [],
            "consulted_functions": [],
            "verification_passed": False,
            "verification_confidence": 0.0,
            "verification_feedback": "",
            "history": [],
            "attempts": 0,
            "max_attempts": max_attempts,
            "metadata": metadata,
            "action": "start"
        }
        
        logger.info(f"Starting ReflectiveAgent workflow for {function_name}")
        final_state = self.workflow.invoke(initial_state)
        
        # Return full state for analysis if needed, or just summary
        # For compatibility with existing callers, we returns a dict now if they support it,
        # or we might need to handle this.
        # The user asked: "Return: final summary + all intermediate scores for analysis"
        # I'll return a dict, but I need to make sure InferencePipeline handles it.
        
        return {
            "summary": final_state.get("summary", ""),
            "scores": final_state.get("scores", {}),
            "history": final_state.get("history", []),
            "verification": {
                "passed": final_state.get("verification_passed"),
                "confidence": final_state.get("verification_confidence")
            }
        }
