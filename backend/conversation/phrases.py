"""
Static, hand-translated template phrases for the conversation pipeline's
non-LLM-generated text (listing headers, clarifications, acknowledgements).

These are deliberately NOT LLM-generated: they're fixed UI strings, not grounded
newspaper content, so a static translation is faster, free, and more predictable
than a model call per turn. The actual answer content (follow-up details, plain
Q&A) IS LLM-generated in the target language -- see backend.rag.prompts.build_grounded_system_prompt.

Only languages in backend.backend_validation.SUPPORTED_LANGUAGES need an entry here.
"""
from typing import Dict

_PHRASES: Dict[str, Dict[str, str]] = {
    "english": {
        "clarify": (
            "I couldn't confidently understand that. Could you rephrase it, e.g. with a "
            "topic/location, or a story number to follow up on?"
        ),
        "clarify_reference": "Which story are you asking about? Try \"number 2\" or ask for news first.",
        "end": "Ending the conversation. Goodbye!",
        "language_ack": "Okay, I'll respond in {language} from now on.",
        "language_unsupported": (
            "Sorry, {language} isn't supported yet. Supported languages right now: "
            "{supported}. I'll keep responding in {current}."
        ),
        "nothing_to_repeat": "There's nothing to repeat yet -- ask me for some news first.",
        "unsupported_intent": "The \"{intent}\" feature isn't implemented yet -- ask me for news instead.",
        "listing_header": "Here are {n} stories:",
        "listing_footer": "Ask \"tell me more about number N\" for details on any of these.",
        "no_stories": "I couldn't find any newspaper stories matching that.",
    },
    "hindi": {
        "clarify": (
            "मुझे यह ठीक से समझ नहीं आया। कृपया इसे दोबारा कहें, जैसे किसी विषय या स्थान के साथ, "
            "या फॉलो-अप के लिए स्टोरी नंबर बताएं।"
        ),
        "clarify_reference": "आप किस खबर के बारे में पूछ रहे हैं? \"नंबर 2\" कहकर देखें, या पहले खबर मांगें।",
        "end": "बातचीत समाप्त हो रही है। अलविदा!",
        "language_ack": "ठीक है, अब से मैं {language} में जवाब दूंगा।",
        "language_unsupported": (
            "क्षमा करें, {language} अभी समर्थित नहीं है। फिलहाल समर्थित भाषाएं: {supported}। "
            "मैं {current} में जवाब देना जारी रखूंगा।"
        ),
        "nothing_to_repeat": "दोहराने के लिए अभी कुछ नहीं है -- पहले मुझसे कोई खबर मांगें।",
        "unsupported_intent": "\"{intent}\" फीचर अभी उपलब्ध नहीं है -- कृपया खबर के लिए पूछें।",
        "listing_header": "यहाँ {n} खबरें हैं:",
        "listing_footer": "किसी भी खबर के बारे में और जानने के लिए \"नंबर N के बारे में और बताओ\" कहें।",
        "no_stories": "मुझे इससे मेल खाती कोई अखबार की खबर नहीं मिली।",
    },
    "marathi": {
        "clarify": (
            "मला ते नीट समजले नाही. कृपया पुन्हा सांगा, उदा. विषय किंवा ठिकाणासह, किंवा "
            "फॉलो-अपसाठी स्टोरी नंबर सांगा."
        ),
        "clarify_reference": "तुम्ही कोणत्या बातमीबद्दल विचारत आहात? \"नंबर 2\" असे विचारून पहा, किंवा आधी बातम्या मागा.",
        "end": "संभाषण संपत आहे. निरोप!",
        "language_ack": "ठीक आहे, आतापासून मी {language} मध्ये उत्तर देईन.",
        "language_unsupported": (
            "क्षमस्व, {language} अजून समर्थित नाही. सध्या समर्थित भाषा: {supported}. "
            "मी {current} मध्येच उत्तर देत राहीन."
        ),
        "nothing_to_repeat": "पुन्हा सांगण्यासाठी सध्या काही नाही -- आधी मला एखादी बातमी विचारा.",
        "unsupported_intent": "\"{intent}\" वैशिष्ट्य अजून उपलब्ध नाही -- कृपया बातमीसाठी विचारा.",
        "listing_header": "या {n} बातम्या आहेत:",
        "listing_footer": "कोणत्याही बातमीबद्दल अधिक माहितीसाठी \"नंबर N बद्दल अधिक सांगा\" असे विचारा.",
        "no_stories": "याच्याशी जुळणारी कोणतीही वृत्तपत्र बातमी मला सापडली नाही.",
    },
}


def phrase(session_language: str, key: str, **kwargs: str) -> str:
    """
    Looks up a template phrase for session_language (falling back to English if the
    language or key is missing) and formats it with kwargs. Note kwargs can itself
    include a "language" key (e.g. for the language_ack template) -- distinct from
    session_language, which only selects which translation table to read from.
    """
    table = _PHRASES.get(session_language, _PHRASES["english"])
    template = table.get(key, _PHRASES["english"].get(key, ""))
    return template.format(**kwargs) if kwargs else template
