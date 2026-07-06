import random


AGENT_NAMES = (
    "Maya Chen",
    "Alex Rivera",
    "Priya Shah",
    "Marcus Reed",
    "Elena Torres",
    "Christian Miller",
    "James Carter",
    "Matthew Brooks",
    "Daniel Foster",
    "Thomas Bennett",
    "Olivia Parker",
    "Emily Morgan",
    "Michael Turner",
    "Sophie Collins",
    "Nathan Hayes",
)

DEFAULT_AGENT_NAME = AGENT_NAMES[0]


def random_agent_name() -> str:
    return random.choice(AGENT_NAMES)
