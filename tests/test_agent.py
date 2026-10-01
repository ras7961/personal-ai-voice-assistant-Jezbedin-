# Agent behavior over the wire is covered by the simulations in scenarios.yaml,
# which run full conversations against the agent on LiveKit Cloud (see README.md).
# The tests below verify the session configuration and instruction contract that
# keep Jzbedin answering completely, staying grounded, and introducing himself.
#

from livekit.agents import inference
from livekit.plugins import google

import agent as agent_module
from agent import (
    Assistant,
    build_instructions,
    build_llm,
    build_turn_handling,
)
from prompts import AGENT_NAME


def test_llm_is_google_realtime_model() -> None:
    llm = build_llm()
    assert isinstance(llm, google.realtime.RealtimeModel)


def test_llm_disables_server_turn_detection() -> None:
    # LiveKit's own TurnDetector owns turn detection, so Gemini's built-in
    # automatic activity detection must be disabled or the two will conflict.
    llm = build_llm()
    assert llm.capabilities.turn_detection is False
    aad = llm._opts.realtime_input_config.automatic_activity_detection
    assert aad.disabled is True


def test_llm_bounds_output_tokens() -> None:
    # A realtime model with an unbounded output budget is the other way Jzbedin
    # gets cut off: the response stops when the provider's cap is hit, not at a
    # sentence boundary. Give the length ladder a hard ceiling it cannot cross.
    llm = build_llm()
    assert llm._opts.max_output_tokens == 800


def test_turn_handling_uses_livekit_turn_detector() -> None:
    options = build_turn_handling()
    assert isinstance(options["turn_detection"], inference.TurnDetector)
    assert options["interruption"]["mode"] == "adaptive"


def test_interruption_tolerates_backchannel() -> None:
    # The default min_words of 0 let a single "mm" cut Jzbedin off mid-answer,
    # and Gemini reports message_truncation=False, so the framework never
    # reconciles the cut and the agent never resumes.
    options = build_turn_handling()
    interruption = options["interruption"]
    assert interruption["min_words"] == 2
    assert interruption["min_duration"] == 0.8
    assert interruption["resume_false_interruption"] is True


def test_preemptive_generation_is_disabled() -> None:
    # Preemptive generation is gated on isinstance(llm, llm.LLM), and
    # RealtimeModel is not a subclass of it, so the option never took effect.
    # Assert it off explicitly so a future model swap cannot silently turn on
    # speculative replies to half-heard questions.
    options = build_turn_handling()
    assert options.get("preemptive_generation") == {"enabled": False}


def test_user_turn_limit_is_set() -> None:
    options = build_turn_handling()
    assert options["user_turn_limit"] == {"max_words": 60}


def test_instructions_are_not_deeply_indented() -> None:
    # Regression guard: the old prompt was built with a textwrap.dedent whose
    # indentation never matched, so all but its first line kept 24 spaces of
    # leading whitespace. A couple of spaces to wrap a bullet in the source is
    # fine and expected; anything deeper is the old bug coming back.
    instructions = Assistant()._instructions
    worst = max(
        (
            len(line) - len(line.lstrip())
            for line in instructions.splitlines()
            if line.strip()
        ),
        default=0,
    )
    assert worst <= 2, f"prompt line indented by {worst} spaces"


def test_instructions_state_the_agent_name() -> None:
    instructions = Assistant()._instructions
    assert AGENT_NAME in instructions
    assert "butler" in instructions
    # He must know he has already greeted, or he introduces himself twice.
    assert "already greeted" in instructions


def test_instructions_require_a_complete_answer() -> None:
    # Guards against the three mutually reinforcing brevity rules that made
    # Jzbedin answer only part of a multi-part question.
    instructions = Assistant()._instructions
    assert "whole question" in instructions
    assert "Never stop part-way" in instructions


def test_instructions_require_grounding() -> None:
    instructions = Assistant()._instructions
    assert "exactly three places" in instructions
    assert "Never report success you did not get" in instructions
    assert "no record of it" in instructions


def test_instructions_have_one_presence_rule() -> None:
    # The old prompt stated the "you there?" rule twice at conflicting levels of
    # strictness, which gave the model room to invent a third behaviour.
    instructions = Assistant()._instructions
    assert instructions.count("you there") == 1


def test_instructions_acknowledge_camera_input() -> None:
    instructions = Assistant()._instructions
    assert "camera" in instructions.lower()
    assert "only what is genuinely visible" in instructions


def test_user_name_is_never_invented() -> None:
    without_name = build_instructions()
    assert AGENT_NAME in without_name
    assert "not yet been told" in without_name

    with_name = build_instructions("Priya")
    assert "Priya" in with_name
    assert "not yet been told" not in with_name


def test_name_capture_handles_ordinary_introductions() -> None:
    from agent import _extract_name

    assert _extract_name("My name is Priya") == "Priya"
    assert _extract_name("I'm Priya") == "Priya"
    assert _extract_name("I am Priya") == "Priya"
    assert _extract_name("call me Priya") == "Priya"
    assert _extract_name("They call me Priya") == "Priya"
    assert _extract_name("Priya") == "Priya"
    assert _extract_name("priya rao") is None  # lowercase: too risky to trust
    assert _extract_name("My name is O'Neill") == "O'Neill"
    assert _extract_name("My name is Jzbedin") == "Jzbedin"  # the user may match


def test_name_capture_prefers_a_missing_name_to_a_wrong_one() -> None:
    from agent import _extract_name

    for reply in (
        "I'm fine, thanks",
        "yeah sure, go ahead",
        "not sure, could you repeat that?",
        "what can you do?",
        "",
        "   ",
        "hello there",
    ):
        assert _extract_name(reply) is None, f"invented a name from {reply!r}"


def test_name_capture_does_not_grab_a_sentence() -> None:
    from agent import _extract_name

    assert _extract_name("I am going to open the website now") is None
    assert _extract_name("I'm looking for a flat in Manchester") is None
    assert _extract_name("I am ready to help you with that task") is None


def test_greeting_states_the_agent_name() -> None:
    assert AGENT_NAME in agent_module.GREETING


def test_greeting_asks_for_a_name() -> None:
    greeting = agent_module.GREETING
    assert "name" in greeting.lower()
    assert "Good day" in greeting


def test_assistant_registers_search_and_browser_tools() -> None:
    assistant = Assistant()
    names = {tool._info.name for tool in assistant._tools}
    assert "search_web" in names
    for expected in ("browse", "browse_click", "browse_interact"):
        assert expected in names
    assert "confirm_browser_action" in names


def test_consolidated_tool_surface_is_small() -> None:
    # 22 tools in every turn was the core of the ineffectiveness: Jzbedin
    # mis-selected and narrated instead of acting. browse folds navigation,
    # reading, and inspection into one call, so the surface can shrink.
    names = {tool._info.name for tool in Assistant()._tools}
    assert len(names) <= 12, f"tool surface grew: {sorted(names)}"
    # The redundant in-browser search path is gone; search_web covers lookups.
    assert "search_the_web" not in names
