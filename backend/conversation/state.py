"""
In-memory conversation session state, per context.md Section 10.

Keeps the session's language, location, current topic/story, the last numbered
set of stories shown, and turn history -- so follow-ups can be resolved against
what was actually shown to the user instead of the user repeating themselves.
Storage is a plain in-memory dataclass (per task.md Phase 5 scope: no database yet).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    session_id: str
    language: str = "english"
    location: Optional[Dict[str, Optional[str]]] = None
    current_topic: Optional[str] = None
    current_story: Optional[Dict[str, Any]] = None
    previous_stories: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


def new_session(session_id: str = "default") -> SessionState:
    return SessionState(session_id=session_id)


def set_story_listing(state: SessionState, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Records a fresh numbered set of retrieved chunks as "the stories last shown to the
    user", replacing any prior listing. The first story also becomes current_story, so
    an immediate un-numbered follow-up ("tell me more") has something to resolve to.
    """
    numbered = [{"number": i + 1, **chunk} for i, chunk in enumerate(chunks)]
    state.previous_stories = numbered
    state.current_story = numbered[0] if numbered else state.current_story
    return numbered


def find_story_by_number(state: SessionState, number: int) -> Optional[Dict[str, Any]]:
    for story in state.previous_stories:
        if story.get("number") == number:
            return story
    return None


def record_turn(state: SessionState, utterance: str, answer: str) -> None:
    state.conversation_history.append({"utterance": utterance, "answer": answer})


def session_context_for_intent_parser(state: SessionState) -> Dict[str, Any]:
    """
    The subset of session state fed back into the intent parser (see
    backend.llm.prompts.intent_parser.parse_intent) so it can resolve references
    like "number 2" or "that story" -- deliberately excludes full chunk text/history,
    since the parser only needs titles/numbers to resolve a reference, not the evidence.
    """
    return {
        "language": state.language,
        "current_topic": state.current_topic,
        "current_story_title": state.current_story.get("title") if state.current_story else None,
        "previous_stories": [
            {"number": s["number"], "title": s.get("title")} for s in state.previous_stories
        ],
    }
