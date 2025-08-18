import pytest
from src.router.config import Config
from src.router.router import ModelRouter


class TestModelRouter:
    def setup_method(self):
        self.config = Config()
        self.router = ModelRouter(self.config)

    def test_plan_mode_detection(self):
        """Test routing decision for plan mode."""
        headers = {"X-Claude-Code-Mode": "plan"}
        request_data = {"model": "claude-3-sonnet"}
        
        decision = self.router.decide_route(headers, request_data)
        
        assert decision.target == "openai"
        assert decision.model == self.config.mapping.plan_model
        assert "plan mode detected" in decision.reasoning.lower()

    def test_haiku_model_detection(self):
        """Test routing decision for haiku models."""
        headers = {}
        request_data = {"model": "claude-3-haiku-20240307"}
        
        decision = self.router.decide_route(headers, request_data)
        
        assert decision.target == "openai"
        assert decision.model == self.config.mapping.background_model
        assert "haiku model detected" in decision.reasoning.lower()

    def test_passthrough_default(self):
        """Test default passthrough behavior."""
        headers = {}
        request_data = {"model": "claude-3-sonnet"}
        
        decision = self.router.decide_route(headers, request_data)
        
        assert decision.target == "anthropic"
        assert decision.model is None
        assert "passthrough" in decision.reasoning.lower()

    def test_reasoning_effort_mapping(self):
        """Test reasoning effort mapping from thinking budget tokens."""
        # Test minimal (no budget)
        assert self.router.get_reasoning_effort({}) == "minimal"
        assert self.router.get_reasoning_effort({"thinking": {}}) == "minimal"
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 0}}) == "minimal"
        
        # Test low effort
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 2000}}) == "low"
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 5000}}) == "low"
        
        # Test medium effort  
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 8000}}) == "medium"
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 15000}}) == "medium"
        
        # Test high effort
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 20000}}) == "high"
        assert self.router.get_reasoning_effort({"thinking": {"budget_tokens": 32000}}) == "high"

    def test_override_rules(self):
        """Test override rule matching."""
        # Add an override rule
        from src.router.config.schema import OverrideRule
        self.config.overrides = [
            OverrideRule(
                when={"header": {"X-Task": "background"}},
                model="gpt-4o-mini"
            )
        ]
        
        headers = {"X-Task": "background"}
        request_data = {"model": "claude-3-sonnet"}
        
        decision = self.router.decide_route(headers, request_data)
        
        assert decision.target == "openai"
        assert decision.model == "gpt-4o-mini"
        assert "override rule matched" in decision.reasoning.lower()

    def test_case_insensitive_header_matching(self):
        """Test case-insensitive header matching."""
        headers = {"x-claude-code-mode": "PLAN"}  # lowercase header, uppercase value
        request_data = {"model": "claude-3-sonnet"}
        
        decision = self.router.decide_route(headers, request_data)
        
        assert decision.target == "openai"
        assert decision.model == self.config.mapping.plan_model