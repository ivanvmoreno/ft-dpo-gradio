import json
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def replace_template(template: str, data: dict) -> str:
    """Replace string template variables with values from a dict."""
    for key, value in data.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def json_to_dict(json_file: str) -> dict:
    """Open, read and deserialize a json file into a dict."""
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def get_last_human_msg(memory: ConversationBufferWindowMemory):
    """Get the last human message from the conversation history."""
    hist = memory.load_memory_variables({})["history"]
    return list(filter(lambda x: x.startswith("Candidate:"), hist.split("\n")))[-1]
