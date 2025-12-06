"""
Tests for PLP operators: plate, handlers, param, deterministic.
"""

from plp import (
    PLPState,
    LLMClient,
    sample,
    plate,
    plate_map,
    plate_reduce,
    PlateContext,
    replay,
    block,
    do,
    param,
    text_param,
    get_registry,
    deterministic,
    store,
    summarize,
    canonicalize,
)


class TestPlate:
    """Tests for the plate operator (NumPyro-style)."""

    def test_plate_context_numpyro_style(self):
        """Test NumPyro-style plate iteration."""
        results = []

        with plate("candidates", 5) as p:
            assert p.name == "candidates"
            assert p.size == 5
            assert len(p) == 5

            for i in p:
                results.append(f"result_{i}")

        assert len(results) == 5
        assert results[0] == "result_0"
        assert results[4] == "result_4"

    def test_plate_with_add_result(self):
        """Test plate with add_result for tracking."""
        with plate("test", 3) as p:
            for i in p:
                state = PLPState()
                output = f"output_{i}"
                p.add_result(state, output)

        assert len(p.results) == 3
        assert p.results == ["output_0", "output_1", "output_2"]

    def test_plate_subsampling(self):
        """Test plate with subsampling (mini-batch)."""
        with plate("data", 100, subsample_size=10) as p:
            assert p.size == 100
            assert p.subsample_size == 10
            assert len(p) == 10  # Only 10 indices
            assert p.scale == 10.0  # 100/10

    def test_plate_map_basic(self, plp_state: PLPState):
        """Test plate_map with a simple function."""
        call_count = {"n": 0}

        def simple_fn(state: PLPState) -> tuple[PLPState, str]:
            call_count["n"] += 1
            new_state = state.copy()
            return new_state, f"result_{call_count['n']}"

        states, outputs = plate_map("simple", 5, simple_fn, plp_state)

        assert len(states) == 5
        assert len(outputs) == 5
        assert call_count["n"] == 5

    def test_plate_reduce_majority(self, plp_state: PLPState):
        """Test plate_reduce with majority vote."""
        outputs_to_return = ["a", "b", "a", "a", "b"]
        idx = {"i": 0}

        def fn(state: PLPState) -> tuple[PLPState, str]:
            result = outputs_to_return[idx["i"]]
            idx["i"] += 1
            return state.copy(), result

        def majority(outs: list[str]) -> str:
            return max(set(outs), key=outs.count)

        state, best = plate_reduce("ensemble", 5, fn, majority, plp_state)

        assert best == "a"  # "a" appears 3 times


class TestHandlers:
    """Tests for program handlers: replay, block, do."""

    def test_replay_basic(self, llm: LLMClient, plp_state: PLPState):
        """Test replay returns cached outputs."""

        @sample(llm, name="test_sample")
        def generate(state: PLPState) -> str:
            return "Say hello"

        # First run
        state1, out1 = generate(plp_state)

        # Replay should use cached output
        with replay(state1.trace) as r:
            state2, out2 = generate(plp_state)
            assert r.replay_count == 1

        assert out2 == out1  # Same output

    def test_block_skips_sites(self, plp_state: PLPState):
        """Test block skips specified sites."""
        from plp import factor

        @factor(name="blocked_factor")
        def my_factor(state: PLPState) -> float:
            return 5.0  # Would add 5.0 to log_weight

        with block(["blocked_factor"]):
            state = my_factor(plp_state)

        # Factor was skipped, log_weight unchanged
        assert state.log_weight == 0.0
        assert state.trace[-1].meta.get("blocked") is True

    def test_do_intervention(self, llm: LLMClient, plp_state: PLPState):
        """Test do forces sample output."""

        @sample(llm, name="intervention_test")
        def generate(state: PLPState) -> str:
            return "What is 2+2?"

        with do({"intervention_test": "FORCED_OUTPUT"}):
            state, out = generate(plp_state)

        assert out == "FORCED_OUTPUT"
        assert state.trace[-1].meta.get("handler") == "do"


class TestParam:
    """Tests for learnable parameters."""

    def test_numeric_param(self):
        """Test numeric parameter creation and bounds."""
        temp = param("test_temp", default=0.7, bounds=(0.1, 1.5))

        assert temp.get() == 0.7

        temp.set(1.0)
        assert temp.get() == 1.0

        # Should clamp to bounds
        temp.set(2.0)
        assert temp.get() == 1.5

        temp.reset()
        assert temp.get() == 0.7

    def test_text_param(self):
        """Test text parameter with candidates."""
        style = text_param(
            "test_style",
            default="Think step by step",
            candidates=["Think step by step", "Let's be careful"],
        )

        assert style.get() == "Think step by step"

        style.set("Custom value")
        assert style.get() == "Custom value"


class TestDeterministic:
    """Tests for deterministic operators."""

    def test_deterministic_decorator(self, plp_state: PLPState):
        """Test @deterministic records in trace."""

        @deterministic(name="extractor")
        def extract_first_word(state: PLPState, text: str) -> str:
            return text.split()[0]

        state, result = extract_first_word(plp_state, "Hello world")

        assert result == "Hello"
        assert len(state.trace) == 1
        assert state.trace[0].primitive == "extractor"
        assert state.trace[0].meta.get("deterministic") is True

    def test_store(self, plp_state: PLPState):
        """Test store helper."""
        state = store(plp_state, "plan", "Do X then Y")

        assert state.vars["plan"] == "Do X then Y"
        assert len(state.trace) == 1

    def test_summarize(self, plp_state: PLPState):
        """Test summarize helper."""
        long_text = "a" * 1000

        state, summary = summarize(plp_state, long_text, max_length=100)

        assert len(summary) == 100
        assert summary.endswith("...")

    def test_canonicalize(self, plp_state: PLPState):
        """Test canonicalize helper."""
        state, result = canonicalize(
            plp_state, "  HELLO World  ", strip=True, lowercase=True
        )

        assert result == "hello world"


class TestScaleMask:
    """Tests for scale and mask on factor."""

    def test_factor_scale(self, plp_state: PLPState):
        """Test factor with scale."""
        from plp import factor

        @factor(name="scaled", scale=0.5)
        def half_factor(state: PLPState) -> float:
            return 10.0

        state = half_factor(plp_state)

        assert state.log_weight == 5.0  # 10 * 0.5
        assert state.trace[-1].meta["scale"] == 0.5

    def test_factor_mask_active(self, plp_state: PLPState):
        """Test factor with mask that activates."""
        from plp import factor

        plp_state.vars["mode"] = "summary"

        @factor(name="masked", mask=lambda s: s.vars.get("mode") == "summary")
        def length_penalty(state: PLPState) -> float:
            return -2.0

        state = length_penalty(plp_state)

        assert state.log_weight == -2.0  # Mask passed, factor applied

    def test_factor_mask_inactive(self, plp_state: PLPState):
        """Test factor with mask that doesn't activate."""
        from plp import factor

        plp_state.vars["mode"] = "full"

        @factor(name="masked2", mask=lambda s: s.vars.get("mode") == "summary")
        def length_penalty(state: PLPState) -> float:
            return -2.0

        state = length_penalty(plp_state)

        assert state.log_weight == 0.0  # Mask failed, factor skipped
        assert state.trace[-1].meta.get("masked") is True
