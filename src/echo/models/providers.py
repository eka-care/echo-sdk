from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    BEDROCK = "bedrock"
