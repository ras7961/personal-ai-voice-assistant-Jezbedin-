import logging
import re

from dotenv import load_dotenv
from google.genai import types
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    UserTurnExceededEvent,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics, google

from browser_tools import BrowserController
from prompts import AGENT_NAME, GREETING, build_instructions
from tools import search_web

logger = logging.getLogger("agent")

load_dotenv(".env.local")

__all__ = [
    "AGENT_NAME",
    "GREETING",
    "Assistant",
    "build_instructions",
    "build_llm",
    "build_turn_handling",
    "server",
]

# Ways a person introduces themselves in reply to "what is your name?".
#
# The keyword phrases are matched case-insensitively but the captured name is
# not: requiring a capital is what keeps "i am going to open the website" from
# yielding "Going". Speech-to-text capitalises proper nouns reliably, so a
# lower-case candidate is far more likely to be an ordinary word than a name
# the user declined to capitalise.
_NAME_PATTERNS = (
    r"\b(?i:my name(?:'s| is))\s+([A-Z][A-Za-z'\-]{1,20})",
    r"\b(?i:they call me)\s+([A-Z][A-Za-z'\-]{1,20})",
    r"\b(?i:call me)\s+([A-Z][A-Za-z'\-]{1,20})",
    r"\b(?i:this is)\s+([A-Z][A-Za-z'\-]{1,20})",
    r"\b(?i:it(?:'s| is))\s+([A-Z][A-Za-z'\-]{1,20})",
    # A reply that is nothing but a name, as in "Priya." or "Priya Nair".
    r"^\W*([A-Z][A-Za-z'\-]{1,20})\W*$",
)

# Weaker signals, trusted only in a short reply. Without the length guard
# "I'm looking for a flat in Manchester" would hand back "Manchester", and a
# wrong name is worse than none: it gets used for the rest of the call.
_WEAK_NAME_PATTERNS = (
    r"\b(?i:i'?m)\s+([A-Z][A-Za-z'\-]{1,20})",
    r"\b(?i:i am)\s+([A-Z][A-Za-z'\-]{1,20})",
)
_WEAK_MAX_WORDS = 4

# Capitalised words that match the weak patterns but are never a name.
_NOT_NAMES = frozenset(
    {
        "afraid",
        "available",
        "busy",
        "done",
        "fine",
        "free",
        "good",
        "great",
        "here",
        "listening",
        "looking",
        "okay",
        "ready",
        "sorry",
        "still",
        "sure",
        "testing",
        "trying",
        "waiting",
    }
)


def _extract_name(said: str) -> str | None:
    """Pull a name out of the user's reply, or return None.

    Returns None far more often than a name is present, which is the safe
    direction: a missed name costs one repeat question, a wrong one gets used
    for the rest of the call.
    """
    text = said.strip().strip("\"'")
    if not text:
        return None

    patterns = _NAME_PATTERNS
    if len(text.split()) <= _WEAK_MAX_WORDS:
        patterns = patterns + _WEAK_NAME_PATTERNS

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(1).strip(".,!?'\" ")
        if len(candidate) < 2 or candidate.lower() in _NOT_NAMES:
            continue
        return candidate[0].upper() + candidate[1:]
    return None


def build_llm() -> google.realtime.RealtimeModel:
    """Build the Gemini Live realtime model that powers Jzbedin's voice.

    LiveKit's TurnDetector owns end-of-turn detection, so Gemini's built-in
    automatic activity detection is disabled to avoid conflicting turn-taking.
    See https://docs.livekit.io/agents/models/realtime/plugins/gemini/#turn-detection
    """
    return google.realtime.RealtimeModel(
        model="gemini-3.1-flash-live-preview",
        voice="Enceladus",
        language="en-GB",
        max_output_tokens=800,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            ),
        ),
    )


def build_turn_handling() -> TurnHandlingOptions:
    """Turn handling for the realtime model.

    The LiveKit turn detector decides when the user has finished speaking; the
    model's own detection is disabled in build_llm().

    The interruption thresholds are deliberately stricter than the defaults.
    With ``min_words`` at its default of 0 a single "mm" counted as an
    interruption and cut Jzbedin off mid-answer, and Gemini reports
    ``message_truncation=False``, so the framework never reconciles the cut:
    the agent's context keeps the whole generated sentence as though it had
    been spoken, and he never resumes. Requiring two words and 0.8s of speech
    filters the coughs and keyboard clicks that were causing it.
    """
    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(),
        interruption={
            "mode": "adaptive",
            "min_words": 2,
            "min_duration": 0.8,
            "resume_false_interruption": True,
        },
        # Speculative replies to half-heard questions are a fabrication source,
        # and they never ran here anyway: the framework gates this on
        # isinstance(llm, LLM) and RealtimeModel is not a subclass of it.
        preemptive_generation={"enabled": False},
        # Let Jzbedin cut in rather than sit silent if the user keeps talking.
        user_turn_limit={"max_words": 60},
    )


class Assistant(Agent):
    def __init__(self) -> None:
        self.browser = BrowserController()
        self._user_name: str | None = None
        super().__init__(
            # A realtime model does speech-to-text, the LLM, and text-to-speech
            # in one duplex session, so there are no separate STT/TTS nodes.
            # See all available models at https://docs.livekit.io/agents/models/
            llm=build_llm(),
            tools=[search_web, *self.browser.build_tools()],
            instructions=build_instructions(),
        )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Learn the user's name the first time they give it, and nothing else.

        Jzbedin asks for a name in his greeting, so this is where the answer
        arrives. Learning it is the only way he can address the user properly
        later; the alternative is inventing one when asked, which the grounding
        rules forbid.

        Gemini drops ``system`` and ``developer`` messages from the chat context,
        so the name is threaded through the instructions rather than pushed in
        as a message. That update is sent mid-session without a reconnect
        (Gemini reports ``mutable_instructions=True``).
        """
        if self._user_name is not None:
            return
        said = (new_message.text_content() or "").strip()
        if not said:
            return
        name = _extract_name(said)
        if not name:
            return
        self._user_name = name
        await self.update_instructions(build_instructions(user_name=name))
        logger.info("Learned user name from the first turn")

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        """Cut in gently instead of leaving the user talking into silence."""
        await self.session.say(
            "Go on, sir, I am listening.",
            allow_interruptions=True,
        )

    async def aclose(self) -> None:
        # Close the agent-controlled browser so it is released when the
        # session ends instead of lingering between rooms.
        await self.browser.aclose()


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        turn_handling=build_turn_handling(),
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    assistant = Assistant()
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # Enable live video input so the agent sees the user's camera feed
            # via the Gemini Live realtime model.
            # See https://docs.livekit.io/agents/models/realtime/plugins/gemini/#video-input
            video_input=True,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )
    # Release the agent-controlled browser when the session ends
    ctx.add_shutdown_callback(assistant.aclose)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()

    # Speak the opening greeting rather than asking the model to invent one.
    # generate_reply() made the introduction non-deterministic: with the old
    # instructions, Jzbedin sometimes skipped saying his own name entirely.
    # session.say() is immediate, costs no model latency, and adds itself to
    # the chat context so he knows he has already introduced himself.
    await session.say(GREETING, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
