from utils import get_last_human_msg
from config import init

import gradio as gr

CONFIG = {
    "model_id": "microsoft/DialoGPT-medium",
    "stop_list": [
        "\nCandidate:",
    ],
    "load_in_4bit": True,
    "rails_cf_path": "rails/",
    "prompt_fname": "prompt_templates/prompt.json",
    "prompt_data_fname": "prompt_templates/prompt_data.json",
}

app = init(CONFIG)


def predict(message, history):
    response = chain.predict(input=message)
    yield response
    

def vote(data: gr.LikeData):
    [i, o] = (get_last_human_msg(chain.memory), "Assistant: " + data.value)
    if data.liked:
        print("👍 You upvoted this response: ", i, o)
    else:
        print("👎 You downvoted this response: ", i, o)


with gr.Blocks() as demo:
    reset = gr.Button("🙈 Reset Conversation", render=False)
    reset.click(fn=lambda: chain.memory.clear())
    chatbot = gr.Chatbot(render=False)
    chatbot.like(vote, None, None)
    chat = gr.ChatInterface(
        predict,
        chatbot=chatbot,
        title="🤖 HR Agent – 🚦 RLHF Test Environment",
        description="Please, provide feedback (👍 positive, 👎 negative) for the agent's responses.",
        examples=user_prompts,
        clear_btn=reset,
        retry_btn="🔄 Regenerate Last",
        undo_btn=None,
    )

demo.queue().launch()
