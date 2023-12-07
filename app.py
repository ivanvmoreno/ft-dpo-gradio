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
from langchain.prompts import load_prompt

from utils import force_git_push


def json_to_dict(json_file: str) -> dict:
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def generate_response(chatbot: ConversationChain, input: str, count=1, prompt_data: dict = None) -> List[str]:
    """Generates responses for a `langchain` chatbot."""
    if prompt_data:
        input = chatbot.prompt.template.format(**prompt_data, input=input)
    return [chatbot.predict(input=input) for _ in range(count)]


def generate_responses(chatbots: List[ConversationChain], inputs: List[str], prompt_data: dict = None) -> List[str]:
    """Generates parallel responses for a list of `langchain` chatbots."""
    results = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in executor.map(
            generate_response,
            chatbots,
            inputs,
            [NUM_RESPONSES] * len(inputs),
            [prompt_data] * len(inputs),
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

prompt = load_prompt(PROMPT_TEMPLATES / "template_01.json")
prompt_data = json_to_dict(PROMPT_TEMPLATES / "data_01.json")
prompt.partial_variables = prompt_data

MODEL_IDS = ["Open-Orca/Mistral-7B-OpenOrca"]
chatbots = []

for model_id in MODEL_IDS:
    chatbots.append(
        ConversationChain(
            llm=HuggingFaceHub(
                repo_id=model_id,
                model_kwargs={"temperature": 1},
                huggingfacehub_api_token=HF_TOKEN,
            ),
            prompt=prompt,
            verbose=False,
            memory=ConversationBufferMemory(ai_prefix="Assistant"),
        )
    )


model_id2model = {chatbot.llm.repo_id: chatbot for chatbot in chatbots}

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
    for idx in range(len(chatbots)):
        state_dict[f"response_{idx+1}"] = ""
    state = gr.JSON(state_dict, visible=False)

    gr.Markdown("# Talk to the assistant")

    state_display = gr.Markdown(f"Your messages: 0/{TOTAL_CNT}")

    # Generate model prediction
    def _predict(txt, state):
        start = time.time()
        responses = generate_responses(chatbots, [txt] * len(chatbots), [prompt_data])
        print(f"Time taken to generate {len(chatbots)} responses : {time.time() - start:.2f} seconds")

        response2model_id = {}
        for chatbot, response in zip(chatbots, responses):
            response2model_id[response] = chatbot.llm.repo_id

        state["cnt"] += 1

        new_state_md = f"Inputs remaining in HIT: {state['cnt']}/{TOTAL_CNT}"

        metadata = {"cnt": state["cnt"], "text": txt}
        for idx, response in enumerate(responses):
            metadata[f"response_{idx + 1}"] = response

        metadata["response2model_id"] = response2model_id

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
            new_state_md,
        )

    def _select_response(selected_response, state, dummy):
        done = state["cnt"] == TOTAL_CNT
        state["generated_responses"].append(selected_response)
        state["data"][-1]["selected_response"] = selected_response
        state["data"][-1]["selected_model"] = state["data"][-1]["response2model_id"][selected_response]
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
            for chatbot in chatbots:
                chatbot.memory = ConversationBufferMemory(ai_prefix="Assistant")
        else:
            # Sync all of the model's memories with the conversation path that
            # was actually taken.
            for chatbot in chatbots:
                chatbot.memory = model_id2model[state["data"][-1]["response2model_id"][selected_response]].memory

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
            dummy,
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

    # Button event handlers
    get_window_location_search_js = """
        function(select_response, state, dummy) {
            return [select_response, state, window.location.search];
        }
        """

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
        _js=get_window_location_search_js,
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
