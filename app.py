import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import Repository
from langchain import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from utils import force_git_push


def replace_template(template: str, data: dict) -> str:
    """Replace template variables with data."""
    for key, value in data.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def json_to_dict(json_file: str) -> dict:
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def generate_response(chatbot: ConversationChain, input: str, count=1) -> List[str]:
    """Generates responses for a `langchain` chatbot."""
    return [chatbot.predict(input=input) for _ in range(count)]


def generate_responses(chatbots: List[ConversationChain], inputs: List[str]) -> List[str]:
    """Generates parallel responses for a list of `langchain` chatbots."""
    results = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in executor.map(
            generate_response,
            chatbots,
            inputs,
            [NUM_RESPONSES] * len(inputs),
        ):
            results += result
    return results


if Path(".env").is_file():
    load_dotenv(".env")
DATASET_REPO_URL = os.getenv("DATASET_REPO_URL")
FORCE_PUSH = os.getenv("FORCE_PUSH")
HF_TOKEN = os.getenv("HF_TOKEN")
PROMPT_TEMPLATES = Path("prompt_templates")
NUM_RESPONSES = 3  # Number of responses to generate per interaction

DATA_FILENAME = "data.jsonl"
DATA_FILE = os.path.join("data", DATA_FILENAME)
repo = Repository(local_dir="data", clone_from=DATASET_REPO_URL, use_auth_token=HF_TOKEN)

TOTAL_CNT = 3  # How many user inputs to collect

PUSH_FREQUENCY = 60


def asynchronous_push(f_stop):
    if repo.is_repo_clean():
        print("Repo currently clean. Ignoring push_to_hub")
    else:
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Auto commit by space")
        if FORCE_PUSH == "yes":
            force_git_push(repo)
        else:
            repo.git_push()
    if not f_stop.is_set():
        # call again in 60 seconds
        threading.Timer(PUSH_FREQUENCY, asynchronous_push, [f_stop]).start()


f_stop = threading.Event()
asynchronous_push(f_stop)

[input_vars, prompt_tpl] = json_to_dict(PROMPT_TEMPLATES / "prompt_01.json").values()
prompt_data = json_to_dict(PROMPT_TEMPLATES / "data_01.json")
prompt_tpl = replace_template(prompt_tpl, prompt_data)
prompt = PromptTemplate(template=prompt_tpl, input_variables=input_vars)

chatbot = ConversationChain(
    llm=HuggingFaceHub(
        repo_id="Open-Orca/Mistral-7B-OpenOrca",
        model_kwargs={"temperature": 1},
        huggingfacehub_api_token=HF_TOKEN,
    ),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant"),
)

demo = gr.Blocks()

with demo:
    # We keep track of state as a JSON
    state_dict = {
        "conversation_id": str(uuid.uuid4()),
        "cnt": 0,
        "data": [],
        "past_user_inputs": [],
        "generated_responses": [],
    }
    state = gr.JSON(state_dict, visible=False)

    gr.Markdown("# Talk to the assistant")

    state_display = gr.Markdown(f"Your messages: 0/{TOTAL_CNT}")

    # Generate model prediction
    def _predict(txt, state):
        start = time.time()
        responses = generate_response(chatbot, txt, count=NUM_RESPONSES)
        print(f"Time taken to generate {len(responses)} responses : {time.time() - start:.2f} seconds")

        state["cnt"] += 1

        metadata = {"cnt": state["cnt"], "text": txt}
        for idx, response in enumerate(responses):
            metadata[f"response_{idx + 1}"] = response

        state["data"].append(metadata)
        state["past_user_inputs"].append(txt)

        past_conversation_string = "<br />".join(
            [
                "<br />".join(["Human 😃: " + user_input, "Assistant 🤖: " + model_response])
                for user_input, model_response in zip(state["past_user_inputs"], state["generated_responses"] + [""])
            ]
        )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True, choices=responses, interactive=True, value=responses[0]),
            gr.update(value=past_conversation_string),
            state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _select_response(selected_response, state):
        done = state["cnt"] == TOTAL_CNT
        state["generated_responses"].append(selected_response)
        state["data"][-1]["selected_response"] = selected_response
        if state["cnt"] == TOTAL_CNT:
            with open(DATA_FILE, "a") as jsonlfile:
                json_data_with_assignment_id = [
                    json.dumps(
                        dict(
                            {
                                "assignmentId": state["assignmentId"],
                                "conversation_id": state["conversation_id"],
                            },
                            **datum,
                        )
                    )
                    for datum in state["data"]
                ]
                jsonlfile.write("\n".join(json_data_with_assignment_id) + "\n")
        toggle_example_submit = gr.update(visible=not done)
        past_conversation_string = "<br />".join(
            [
                "<br />".join(["😃: " + user_input, "🤖: " + model_response])
                for user_input, model_response in zip(state["past_user_inputs"], state["generated_responses"])
            ]
        )
        toggle_final_submit = gr.update(visible=False)

        if done:
            # Wipe the memory
            chatbot.memory = ConversationBufferMemory(ai_prefix="Assistant")
        else:
            # Sync model's memory with the conversation path that
            # was actually taken.
            chatbot.memory = state["data"][-1][selected_response].memory

        text_input = gr.update(visible=False) if done else gr.update(visible=True)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            text_input,
            gr.update(visible=False),
            state,
            gr.update(value=past_conversation_string),
            toggle_example_submit,
            toggle_final_submit,
        )

    # Input fields
    past_conversation = gr.Markdown()
    text_input = gr.Textbox(placeholder="Enter a statement", show_label=False)
    select_response = gr.Radio(
        choices=[None, None],
        visible=False,
        label="Choose the most helpful and honest response",
    )
    select_response_button = gr.Button("Select Response", visible=False)
    with gr.Column() as example_submit:
        submit_ex_button = gr.Button("Submit")
    with gr.Column(visible=False) as final_submit:
        submit_hit_button = gr.Button("Submit HIT")

    select_response_button.click(
        _select_response,
        inputs=[select_response, state],
        outputs=[
            select_response,
            example_submit,
            text_input,
            select_response_button,
            state,
            past_conversation,
            example_submit,
            final_submit,
        ],
    )

    submit_ex_button.click(
        _predict,
        inputs=[text_input, state],
        outputs=[
            text_input,
            select_response_button,
            select_response,
            past_conversation,
            state,
            example_submit,
            final_submit,
            state_display,
        ],
    )

    submit_hit_button.click(
        lambda state: state,
        inputs=[state],
        outputs=[state],
    )

demo.launch()
