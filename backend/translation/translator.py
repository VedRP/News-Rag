"""
Translates a grounded English answer into the session's target language.

Design: grounded generation (backend.rag.answer.generate_grounded_answer) always
happens in English -- the LLM reasons against English source chunks, which keeps
grounding/citation fidelity as reliable as Phase 3-5 already proved it to be.
Localization to the target language happens as a separate step, AFTER generation:

- Hindi: Argos Translate (https://github.com/argosopentech/argos-translate), an
  offline, free, open-source neural MT engine -- no per-call API cost, no network
  dependency once the language package is installed locally.
- Marathi: Argos Translate has no Marathi model in its package index (checked
  directly against argostranslate.package.get_available_packages() -- "mr" isn't
  in the supported language list at all), so Marathi falls back to asking Groq's
  LLM to translate the English answer as a dedicated, narrow translation call
  (distinct from the grounded-generation call -- this one's only job is faithful
  translation, not reasoning about evidence).
- English: passthrough, no translation step.

Both paths protect "[Source: ..., Page ...]" citations from ever being sent through
a translator (translating a page number or garbling a filename would violate
context.md's grounding/citation-fidelity rules). Initially this used a placeholder
token ("@@CITE0@@") swapped in before translation and restored after -- but Argos
Translate's MT model isn't instruction-followable like an LLM, and testing showed
it silently mangled the placeholder itself (into "@Cite0"), which would have broken
citation restoration and silently dropped citations from Hindi answers. Fixed by
never sending citation text through a translator at all: split the text around each
citation, translate only the prose segments, then splice the untouched citations
back into their original positions.
"""
import re
from typing import Callable, List

CITATION_PATTERN = re.compile(r"\[Source:[^\]]*\]")


def _translate_preserving_citations(text: str, translate_segment: Callable[[str], str]) -> str:
    """
    Splits text on citation boundaries, translates only the non-citation segments,
    and reassembles with the original (untranslated) citations spliced back in.
    """
    segments = CITATION_PATTERN.split(text)
    citations = CITATION_PATTERN.findall(text)

    translated_segments = [translate_segment(s) if s.strip() else s for s in segments]

    # Normalize whitespace at segment/citation boundaries ourselves rather than trusting
    # the translator to preserve it -- MT engines commonly trim leading/trailing spaces.
    result = translated_segments[0].rstrip()
    for citation, following_segment in zip(citations, translated_segments[1:]):
        result += (" " if result else "") + citation
        following_stripped = following_segment.lstrip()
        if following_stripped:
            result += " " + following_stripped
    return result


_hindi_package_checked = False


def _ensure_hindi_package_installed() -> None:
    """Installs the Argos Translate en->hi package on first use if not already present."""
    global _hindi_package_checked
    if _hindi_package_checked:
        return

    import argostranslate.package as package

    installed = package.get_installed_packages()
    if not any(p.from_code == "en" and p.to_code == "hi" for p in installed):
        package.update_package_index()
        available = package.get_available_packages()
        pkg = next(p for p in available if p.from_code == "en" and p.to_code == "hi")
        download_path = pkg.download()
        package.install_from_path(download_path)

    _hindi_package_checked = True


def _translate_to_hindi(text: str) -> str:
    import argostranslate.translate as translate

    _ensure_hindi_package_installed()
    return _translate_preserving_citations(text, lambda s: translate.translate(s, "en", "hi"))


TRANSLATION_SYSTEM_PROMPT = """You are a precise translator. Translate the given English \
text into {language}, using its native script. Preserve meaning exactly -- do not add, \
remove, or reinterpret any facts, numbers, names, or dates. Output ONLY the translated \
text, nothing else -- no explanation, no quotes, no markdown."""


def _translate_via_llm(text: str, target_language: str) -> str:
    from backend.llm.client import chat

    def translate_segment(segment: str) -> str:
        return chat(
            system_prompt=TRANSLATION_SYSTEM_PROMPT.format(language=target_language.title()),
            user_prompt=segment,
            role="fast",
            temperature=0.0,
        )

    return _translate_preserving_citations(text, translate_segment)


def translate_answer(english_text: str, target_language: str) -> str:
    """
    Translates a grounded English answer into target_language ("english"/"hindi"/"marathi").
    Returns the text unchanged for "english" or any language this module doesn't yet
    handle (callers should already have validated target_language against
    backend.backend_validation.SUPPORTED_LANGUAGES before calling this).
    """
    language = (target_language or "english").lower()
    if language == "english" or not english_text.strip():
        return english_text
    if language == "hindi":
        return _translate_to_hindi(english_text)
    if language == "marathi":
        return _translate_via_llm(english_text, "marathi")
    return english_text
