import random


AGENT_NAMES = (
    "HN Tech Agent",
    "Signal Scout",
    "Briefing Agent",
    "Startup Watch Agent",
    "Research Digest Agent",
)

DEFAULT_AGENT_NAME = AGENT_NAMES[0]


def random_agent_name() -> str:
    return random.choice(AGENT_NAMES)
