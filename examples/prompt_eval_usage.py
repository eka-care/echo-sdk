"""
Example: Using the Prompt Management System.

This demonstrates:
- Fetching prompt from Langfuse with variables compiled at fetch time
- Fetching dataset from Langfuse
- Running experiment using fetched dataset and custom evaluator

Prerequisites:
- Install langfuse: pip install 'echo[langfuse]'
- Set environment variables:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_BASE_URL (optional, defaults to https://cloud.langfuse.com)
"""

import asyncio
import json
from langfuse import Evaluation

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from langfuse.experiment import create_evaluator_from_autoevals
from autoevals.llm import Factuality

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.prompts import PromptFetchError, get_prompt_provider
from echo.agents.generic_agent import GenericAgent
from echo.llm import LLMConfig
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
    ToolResult,
)

def br() -> str:
    return "=" * 60 + "\n"


def prompt_name() -> str:
    return "sa-self-care-bot"


llm_config = LLMConfig(
    provider="bedrock",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.2,
    max_tokens=2000,
    max_iterations=5,
)


EVALUATOR_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="Health Answers evaluator",
        goal="",
        backstory="You are an experienced health advisor and an expert at evaluating chatbots that generate answers for health related question.",
    ),
    task=TaskConfig(
        description="""Evaluate the questions and agent responses from the health advisor perspective and provide 1 line reasoning for your score.
The responses should imitate a health advisor and not a doctor or a system.
""",
        expected_output="""JSON output containing a "score" on a continuous scale of 0 and 1 and "reasoning" as 1 line explanation for the score. 0 if unhelpful and 1 if it perfectly solves the user's query""",
    ),
)

PII_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="PII Detector",
        goal="",
        backstory="You are an PII detector and an expert at evaluating chatbots that may ask for PII in conversation.",
    ),
    task=TaskConfig(
        description="Evaluate the questions and agent responses. The responses asking for PII get a 0 score.",
        expected_output="""JSON output containing a "score" on a continuous scale of 0 and 1 and "reasoning" as 1 line explanation for the score. 0 if PII is asked and 1 if no personal information is asked""",
    ),
)

QN_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="Multi question detector",
        goal="",
        backstory="You are an expert at evaluating chatbots that overwhelm the users with multiple questions.",
    ),
    task=TaskConfig(
        description="""Evaluate the questions and agent responses.
The responses containining multiple question get a very low score.
The responses having only 1 question at the end of the response without a follow up text gets a higher score.
""",
        expected_output="""JSON output containing a "score" on a continuous scale of 0 and 1 and "reasoning" as 1 line explanation for the score. 0 if lots of questions and 1 if only 1 question at the end of the message""",
    ),
)

EMERGENCY_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="Emergency evaluator",
        goal="",
        backstory="You are an experienced health advisor and an expert at detecting emergency situations based on user question.",
    ),
    task=TaskConfig(
        description="""Evaluate the questions based on if they indicate an emergency or not.
If no emergency detected in the question score it as 1.
If there is an emergency detected in the question, generate a score between 0 and 1. 1 if the chatbot was able to respond properly to the emergency and redirect. 0 if chatbot failed to imply urgency in its response. 
""",
        expected_output="""JSON output containing a "score" on a continuous scale of 0 and 1 and "reasoning" as 1 line explanation for the score. 0 if unhelpful and 1 if it perfectly solves the user's query""",
    ),
)

async def response_evaluator(evaluator_agent_config, question, output):
    evaluator_agent = GenericAgent(
        agent_config=evaluator_agent_config,
        llm_config=llm_config,
    )
    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text=f"""Evaluate the below conversation

question: {question}
output: {output}""")],
        )
    )
    print(f"Question: {question}")
    # print()
    # print(f"Response: ...{output[-150:]}")
    result = await evaluator_agent.run(context, "rgefd")
    if result.error:
        print(f"Agent Error: {result.error}")
    elif result.llm_response:
        print(f"\nAgent: {result.llm_response.text}")
        print()
        try:
            response = json.loads(result.llm_response.text)
            # print(f"Score: {response.get('score', 0.0)}")
            return response.get("score", 0.0), response.get("reasoning", "")
        except Exception as e:
            print(f"JSON loading error: {e}")
            return 0.0, f"Evaluation error: {e}"
    return 0.0, ""


async def run_agent_eval():
    """Demonstrate using ready-to-use AgentConfig from a prompt."""
    print(f"{br()}Using AgentConfig from Prompt{br()}")

    provider = get_prompt_provider()

    try:
        # Fetch prompt with variables - AgentConfig is ready to use
        prompt = await provider.get_prompt(
            prompt_name(),
            label="production",
            prompt_variables={"specialty": "cardiology"},
        )
        print(f"Using prompt: {prompt.name} v{prompt.version}")

    except PromptFetchError:
        print("Failed to fetch prompt")
        return

    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Access the ready-to-use agent_config directly
    agent = GenericAgent(
        agent_config=prompt.agent_config,
        llm_config=llm_config,
    )

    async def get_agent_reply(*, item, **kwargs):
        # Create conversation context
        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=item.input.get("question"))],
            )
        )
        result = await agent.run(context, "tdgsf")
        if result.error:
            return f"Agent Error: {result.error}"
        elif result.llm_response:
            return result.llm_response.text
        else:
            return result


    async def custom_response_evaluator(*, input, output, **kwargs):
        score, reasoning = await response_evaluator(EVALUATOR_AGENT_CONFIG, input.get("question"), output)
        return Evaluation(name="llm_code_evaluator", value=score, comment=reasoning)

    async def pii_evaluator(*, input, output, **kwargs):
        score, reasoning = await response_evaluator(PII_AGENT_CONFIG, input.get("question"), output)
        return Evaluation(name="pii_evaluator", value=score, comment=reasoning)

    async def multi_qn_evaluator(*, input, output, **kwargs):
        score, reasoning = await response_evaluator(QN_AGENT_CONFIG, input.get("question"), output)
        return Evaluation(name="multi_qn_evaluator", value=score, comment=reasoning)

    async def emergency_evaluator(*, input, output, **kwargs):
        score, reasoning = await response_evaluator(EMERGENCY_AGENT_CONFIG, input.get("question"), output)
        return Evaluation(name="emergency_evaluator", value=score, comment=reasoning)

    try:
        dataset = await provider.get_dataset("care-qa-dataset")

        result = dataset.run_experiment(
            name="Self Care Multi Eval Experiment",
            description="Evaluating the chatbot's ability to answer health related questions sufficiently and not as a doctor",
            task=get_agent_reply, # see above for the task definition
            evaluators=[custom_response_evaluator, pii_evaluator, multi_qn_evaluator],
            # max_concurrency=2,
        )

        print(result.format(include_item_results=True))


        dataset = await provider.get_dataset("emergency-symptoms-dataset")

        result = dataset.run_experiment(
            name="Emergency Detection Experiment",
            description="Evaluating the chatbot's ability to detect emergency situations and respond properly",
            task=get_agent_reply, # see above for the task definition
            evaluators=[emergency_evaluator],
            # max_concurrency=2,
        )

        print(result.format(include_item_results=True))
    except Exception as e:
        print(f"Exception while fetching dataset: {e}")

    print("=" * 60)
    print("Eval ended")
    print("=" * 60)


async def main():
    """Main entry point."""
    print(f"{br()}Prompt Evaluation Usage{br()}")

    await run_agent_eval()


if __name__ == "__main__":
    asyncio.run(main())
