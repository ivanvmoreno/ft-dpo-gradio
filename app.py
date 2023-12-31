import json
import os
from pathlib import Path
from typing import List

import gradio as gr
import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import Repository
from langchain import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList


def replace_template(template: str, data: dict) -> str:
    """Replace template variables with data."""
    for key, value in data.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def json_to_dict(json_file: str) -> dict:
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def get_last_human_msg(memory: ConversationBufferWindowMemory):
    hist = memory.load_memory_variables({})["history"]
    return list(filter(lambda x: x.startswith("Candidate:"), hist.split("\n")))[-1]


if Path(".env").is_file():
    load_dotenv(".env")
DATASET_REPO_URL = os.getenv("DATASET_REPO_URL")
FORCE_PUSH = os.getenv("FORCE_PUSH")
HF_TOKEN = os.getenv("HF_TOKEN")
PROMPT_TEMPLATES = Path("prompt_templates")
NUM_RESPONSES = 3  # Number of responses to generate per interaction

DATA_FILENAME = "data.jsonl"
DATA_FILE = os.path.join("data", DATA_FILENAME)
repo = Repository(local_dir="data", clone_from=DATASET_REPO_URL, token=HF_TOKEN)

TOTAL_CNT = 3  # How many user inputs to collect

PUSH_FREQUENCY = 60

# Load prompt
model_name = "llama2"
model_version = "13B"
prompt_version = "0.1".replace(".", "_")
prompt_fname = PROMPT_TEMPLATES / f"{model_name}_{model_version}_prompt_{prompt_version}.json"
[model_id, input_vars, prompt_tpl] = json_to_dict(prompt_fname).values()
prompt_data = json_to_dict(PROMPT_TEMPLATES / "prompt_data.json")
user_prompts = prompt_data.pop("user_prompts")
examples = prompt_data.pop("few_shot_examples")
prompt_examples = PromptTemplate(
    input_variables=["question", "answer"], template="Candidate: {question}\nAssistant: {answer}"
)

prompt_tpl = replace_template(prompt_tpl, prompt_data)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt_examples,
    prefix=prompt_tpl + "\nExample interactions:\n",
    suffix="\nCurrent conversation:\n{history}\nCandidate: {input}\nAssistant:",
    input_variables=input_vars,
)

# Run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🏎️ Using GPU" if device.type == "cuda" else "🐌 Using CPU")

# Quantization config
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# HF model config
model_config = transformers.AutoConfig.from_pretrained(model_id, token=HF_TOKEN)

# Load model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN,
)

# List of stop words
stop_list = [
    "\nCandidate:",
]
stop_list = [tokenizer(w)["input_ids"] for w in stop_list]


class StopOnTokens(StoppingCriteria):
    def __init__(self, eos_sequence: List[int]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


def predict(message, history):
    response = chain.predict(input=message)
    yield response


generator = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    temperature=0.75,
    do_sample=True,
    max_new_tokens=256,
    repetition_penalty=1.1,
    device_map="auto",
    stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_list)]),
)

chain = ConversationChain(
    llm=HuggingFacePipeline(
        pipeline=generator,
    ),
    prompt=prompt,
    memory=ConversationBufferWindowMemory(k=5, ai_prefix="Assistant", human_prefix="Candidate"),
    verbose=True,
)


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
