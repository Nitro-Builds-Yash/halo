import pytest
from src.agents.state import AgentState
from src.agents.graph import decide_to_generate

def test_decide_to_generate():
    state: AgentState = {
        "question": "test",
        "documents": [],
        "web_search_needed": True,
        "generation": "",
        "retry_count": 0
    }
    assert decide_to_generate(state) == "transform_query"
    
    state["web_search_needed"] = False
    assert decide_to_generate(state) == "generate"
