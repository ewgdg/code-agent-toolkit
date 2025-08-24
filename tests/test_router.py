from src.claude_router.config import Config
from src.claude_router.config.schema import OverrideRule, ProviderConfig
from src.claude_router.router import ModelRouter


class TestModelRouter:
    def setup_method(self):
        self.config = Config()
        self.router = ModelRouter(self.config)

    def test_plan_mode_detection(self):
        """Test routing decision for plan mode using system prompt regex detection."""
        # Add override rule for plan mode
        self.config.overrides = [
            OverrideRule(
                when={
                    "request": {
                        "system_regex": r"plan mode is (activated|triggered|on)"
                    }
                },
                model="openai/gpt-4o-reasoning",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "system": [
                {"text": "You are Claude Code, an AI assistant."},
                {
                    "text": (
                        "Plan mode is activated. "
                        "You should break down complex tasks into steps."
                    )
                },
            ],
            "messages": [{"role": "user", "content": "Help me implement a feature"}],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o-reasoning"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_plan_mode_detection_string_system(self):
        """Test plan mode detection with string format system prompt."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"system_regex": r"planning.*mode"}},
                model="openai/gpt-4o-reasoning",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "system": (
                "You are Claude Code. "
                "Planning mode is now active for this complex task."
            ),
            "messages": [{"role": "user", "content": "Build an app"}],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o-reasoning"

    def test_plan_mode_not_detected(self):
        """Test that normal requests without plan mode are not routed."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"system_regex": r"\bplan mode\b"}},
                model="openai/gpt-4o-reasoning",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "system": [{"text": "You are Claude Code, an AI assistant."}],
            "messages": [{"role": "user", "content": "What's the weather?"}],
        }

        decision = self.router.decide_route(headers, request_data)

        # Should fallback to default (Anthropic passthrough)
        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_haiku_model_detection(self):
        """Test routing decision for haiku models using override rules."""
        # Add override rule for haiku models
        self.config.overrides = [
            OverrideRule(
                when={"request": {"model_regex": "haiku"}}, model="openai/gpt-5-mini"
            )
        ]

        headers = {}
        request_data = {"model": "claude-3-haiku-20240307"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5-mini"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_passthrough_default(self):
        """Test default passthrough behavior."""
        headers = {}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"
        assert "passthrough" in decision.reason.lower()

    def test_reasoning_effort_mapping(self):
        """Test reasoning effort mapping from thinking budget tokens."""
        # Test minimal (no budget)
        assert self.router.get_reasoning_effort({}) == "minimal"
        assert self.router.get_reasoning_effort({"thinking": {}}) == "minimal"
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 0}})
            == "minimal"
        )

        # Test low effort
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 2000}})
            == "low"
        )
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 5000}})
            == "low"
        )

        # Test medium effort
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 8000}})
            == "medium"
        )
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 15000}})
            == "medium"
        )

        # Test high effort
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 20000}})
            == "high"
        )
        assert (
            self.router.get_reasoning_effort({"thinking": {"budget_tokens": 32000}})
            == "high"
        )

    def test_override_rules(self):
        """Test override rule matching."""
        # Add an override rule
        self.config.overrides = [
            OverrideRule(
                when={"header": {"X-Task": "background"}}, model="openai/gpt-4o-mini"
            )
        ]

        headers = {"X-Task": "background"}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o-mini"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_case_insensitive_header_matching(self):
        """Test case-insensitive header matching with override rules."""
        # Add override rule for header matching
        self.config.overrides = [
            OverrideRule(when={"header": {"X-Task": "plan"}}, model="openai/gpt-5")
        ]

        headers = {"X-Task": "PLAN"}  # exact header name, uppercase value
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_override_rules_multiple_conditions(self):
        """Test override rules with multiple conditions."""
        self.config.overrides = [
            OverrideRule(
                when={
                    "header": {"X-Environment": "production"},
                    "request": {"model_regex": "claude"},
                },
                model="openai/gpt-4o",
            )
        ]

        # Should match when both conditions are met
        headers = {"X-Environment": "production"}
        request_data = {"model": "claude-3-sonnet-20240229"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o"
        assert "override rule 1 matched" in decision.reason.lower()

        # Should not match when only one condition is met
        headers = {"X-Environment": "development"}
        request_data = {"model": "claude-3-sonnet-20240229"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_override_rules_precedence(self):
        """Test that override rules are processed in order (first match wins)."""
        self.config.overrides = [
            # First rule (should win)
            OverrideRule(
                when={"request": {"model_regex": "haiku"}},
                model="openai/gpt-4o-mini",
            ),
            # Second rule (more specific but comes later)
            OverrideRule(
                when={"request": {"model_regex": r"^claude-3-haiku-20240307$"}},
                model="openai/gpt-5-mini",
            ),
        ]

        headers = {}
        request_data = {"model": "claude-3-haiku-20240307"}

        decision = self.router.decide_route(headers, request_data)

        # Should use first matching rule
        assert decision.target == "openai"
        assert decision.model == "gpt-4o-mini"

    def test_override_rules_exact_model_match(self):
        """Test exact model matching in override rules using regex anchors."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"model_regex": r"^claude-3-opus-20240229$"}},
                model="openai/gpt-5",
            )
        ]

        # Should match exact model
        headers = {}
        request_data = {"model": "claude-3-opus-20240229"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5"

        # Should not match different model
        request_data = {"model": "claude-3-sonnet-20240229"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_override_rules_header_list_values(self):
        """Test header matching with list of values."""
        self.config.overrides = [
            OverrideRule(
                when={"header": {"X-Priority": ["high", "critical"]}},
                model="openai/gpt-5",
            )
        ]

        # Should match first value
        headers = {"X-Priority": "high"}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5"

        # Should match second value
        headers = {"X-Priority": "critical"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5"

        # Should not match other values
        headers = {"X-Priority": "low"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_override_rules_tool_detection_complex(self):
        """Test tool availability detection scenarios."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"has_tool": "TodoWrite"}}, model="openai/gpt-4o"
            )
        ]

        # Should match when tool is available in tools list
        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "tools": [
                {"name": "TodoWrite", "description": "Create and manage todo lists"}
            ],
            "messages": [{"role": "user", "content": "Create a todo list"}],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o"

        # Should not match when tool is not available
        request_data = {
            "model": "claude-3-sonnet",
            "tools": [{"name": "SomeOtherTool", "description": "Different tool"}],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_override_rules_provider_parsing(self):
        """Test provider/model string parsing."""
        test_cases = [
            ("openai/gpt-5", "openai", "gpt-5"),
            ("anthropic/claude-3-sonnet", "anthropic", "claude-3-sonnet"),
            ("gpt-4o", "openai", "gpt-4o"),  # fallback to openai
        ]

        for provider_model_string, expected_provider, expected_model in test_cases:
            self.config.overrides = [
                OverrideRule(
                    when={"header": {"X-Test": "true"}}, model=provider_model_string
                )
            ]

            headers = {"X-Test": "true"}
            request_data = {"model": "claude-3-sonnet"}

            decision = self.router.decide_route(headers, request_data)

            assert decision.target == expected_provider
            assert decision.model == expected_model

    def test_override_rules_no_match(self):
        """Test behavior when no override rules match."""
        self.config.overrides = [
            OverrideRule(when={"header": {"X-Special": "value"}}, model="openai/gpt-4o")
        ]

        # No matching headers
        headers = {}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"
        assert "no routing rules matched" in decision.reason.lower()

    def test_override_rules_empty_list(self):
        """Test behavior with empty override rules list."""
        self.config.overrides = []

        headers = {"X-Anything": "value"}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_provider_model_query_parameters(self):
        """Test parsing model query parameters from override rules."""
        test_cases = [
            # Basic query parameter
            ("openai/gpt-5?temperature=0.5", "openai", "gpt-5", {"temperature": 0.5}),
            # Nested query parameter
            (
                "openai/gpt-5?reasoning.effort=low",
                "openai",
                "gpt-5",
                {"reasoning": {"effort": "low"}},
            ),
            # Multiple parameters
            (
                "openai/gpt-5?reasoning.effort=high&temperature=0.3",
                "openai",
                "gpt-5",
                {"reasoning": {"effort": "high"}, "temperature": 0.3},
            ),
            # Boolean parameter
            ("openai/gpt-5?stream=true", "openai", "gpt-5", {"stream": True}),
            # No query parameters
            ("openai/gpt-5", "openai", "gpt-5", {}),
        ]

        for (
            provider_model_string,
            expected_provider,
            expected_model,
            expected_config,
        ) in test_cases:
            provider, model, config = self.router._parse_provider_model(
                provider_model_string
            )

            assert provider == expected_provider
            assert model == expected_model
            assert config == expected_config

    def test_override_rule_with_query_parameters(self):
        """Test that override rules pass through model configuration."""
        self.config.overrides = [
            OverrideRule(
                when={"header": {"X-Test": "config"}},
                model="openai/gpt-5?reasoning.effort=high&temperature=0.2",
            )
        ]

        headers = {"X-Test": "config"}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-5"
        assert decision.model_config == {
            "reasoning": {"effort": "high"},
            "temperature": 0.2,
        }

    def test_parameter_type_conversion(self):
        """Test automatic type conversion of query parameters."""
        test_cases = [
            ("temperature=0.5", 0.5),  # float
            ("max_tokens=1000", 1000),  # int
            ("stream=true", True),  # boolean true
            ("stream=false", False),  # boolean false
            ("model_name=gpt-5", "gpt-5"),  # string
        ]

        for param_string, expected_value in test_cases:
            provider, model, config = self.router._parse_provider_model(
                f"openai/gpt-5?{param_string}"
            )
            param_name = param_string.split("=")[0]

            assert config[param_name] == expected_value

    def test_user_regex_detection(self):
        """Test routing decision for system reminder messages using user_regex."""
        # Add override rule for system-reminder detection
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"<system-reminder>"}},
                model="anthropic/claude-3-sonnet",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>This is a system reminder about the project"
                        "</system-reminder>What should I do?"
                    ),
                }
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "claude-3-sonnet"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_user_regex_detection_with_text_blocks(self):
        """Test user_regex detection with structured content blocks."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"CLAUDE\.md"}},
                model="openai/gpt-4o",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Contents of /home/user/project/CLAUDE.md "
                                "(project instructions)"
                            ),
                        }
                    ],
                }
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o"

    def test_user_regex_not_detected(self):
        """Test that normal user messages without matching regex are not routed."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"<system-reminder>"}},
                model="openai/gpt-4o",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "user", "content": "What's the weather today?"},
                {"role": "assistant", "content": "I can't check the weather."},
                {"role": "user", "content": "How about tomorrow?"},
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        # Should fallback to default (Anthropic passthrough)
        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_user_regex_last_user_message_only(self):
        """Test user_regex matching only the last user message."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"urgent.*help"}},
                model="openai/gpt-5",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "user", "content": "I need urgent help with my code"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Actually, just hello there"},
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        # Should NOT match because the last user message doesn't contain "urgent help"
        assert decision.target == "anthropic"
        assert decision.model == "passthrough"

    def test_user_regex_matches_last_user_message(self):
        """Test user_regex matches when pattern is in the last user message."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"urgent.*help"}},
                model="openai/gpt-5",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "I need urgent help with my code"},
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        # Should match because the last user message contains "urgent help"
        assert decision.target == "openai"
        assert decision.model == "gpt-5"

    def test_user_regex_case_insensitive(self):
        """Test that user_regex is case insensitive."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"user_regex": r"ERROR"}},
                model="openai/gpt-4o-mini",
            )
        ]

        headers = {}
        request_data = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "user", "content": "I'm getting an error in my application"},
            ],
        }

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o-mini"

    def test_provider_based_routing(self):
        """Test new provider-based routing with explicit provider field."""
        # Configure a custom provider
        self.config.providers = {
            "llama-local": ProviderConfig(
                base_url="http://localhost:8080/v1",
                adapter="openai-chat-completions",
                api_key_env="LLAMA_API_KEY",
            )
        }

        # Add override rule with explicit provider
        self.config.overrides = [
            OverrideRule(
                when={"request": {"model_regex": "llama"}},
                model="llama3.1",
                provider="llama-local",
            )
        ]

        headers = {}
        request_data = {"model": "llama-latest"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "llama-local"
        assert decision.model == "llama3.1"
        assert decision.provider == "llama-local"
        assert decision.adapter == "openai-chat-completions"
        assert "override rule 1 matched" in decision.reason.lower()

    def test_provider_resolution_from_model_prefix(self):
        """Test provider resolution from model prefix when no explicit provider."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"model_regex": "test"}}, model="openai/gpt-4o"
            )
        ]

        headers = {}
        request_data = {"model": "test-model"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "openai"
        assert decision.model == "gpt-4o"
        assert decision.provider == "openai"
        assert decision.adapter == "openai-responses"  # default for openai

    def test_provider_unknown_defaults_to_chat_completions(self):
        """Test that unknown providers default to openai-chat-completions adapter."""
        self.config.overrides = [
            OverrideRule(
                when={"request": {"model_regex": "test"}},
                model="unknown-model",
                provider="unknown-provider",
            )
        ]

        headers = {}
        request_data = {"model": "test-model"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "unknown-provider"
        assert decision.model == "unknown-model"
        assert decision.provider == "unknown-provider"
        assert decision.adapter == "openai-chat-completions"  # default for unknown

    def test_default_anthropic_passthrough_fields(self):
        """Test default routing includes correct provider and adapter fields."""
        headers = {}
        request_data = {"model": "claude-3-sonnet"}

        decision = self.router.decide_route(headers, request_data)

        assert decision.target == "anthropic"
        assert decision.model == "passthrough"
        assert decision.provider == "anthropic"
        assert decision.adapter == "anthropic-passthrough"
