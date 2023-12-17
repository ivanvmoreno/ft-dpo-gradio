import os
from pathlib import Path
from typing import List, Union

from utils import json_to_dict, replace_template

from dotenv import load_dotenv
from langchain import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMemory
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import (
    HuggingFacePipelineCompatible,
    register_llm_provider,
)
import torch
import transformers
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.pipelines import Pipeline

if Path(".env").is_file():
    load_dotenv(".env")
HF_TOKEN = os.getenv("HF_TOKEN")


def _get_stopping_criteria(stop_list: List[int], tokenizer: AutoTokenizer) -> StoppingCriteriaList:
    class StopOnTokens(StoppingCriteria):
        def __init__(self, eos_sequence: List[int]):
            self.eos_sequence = eos_sequence

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
            return self.eos_sequence in last_ids
    
    stop_list = [tokenizer(w)["input_ids"] for w in stop_list]
    return StoppingCriteriaList([StopOnTokens(stop_list)])


def _get_quant_config() -> transformers.BitsAndBytesConfig:
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    return bnb_config


def _get_pipeline(model_id: str, stop_list: List[int], hf_token: str = HF_TOKEN, load_in_4bit: bool = True) -> Pipeline:
    model_config = transformers.AutoConfig.from_pretrained(model_id, token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=_get_quant_config() if load_in_4bit else None,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
    )
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        temperature=0.75,
        do_sample=True,
        max_new_tokens=256,
        repetition_penalty=1.1,
        device_map="auto",
        stopping_criteria=_get_stopping_criteria(stop_list, tokenizer),
    )
    return pipeline


def _get_prompt(prompt_fname: str, prompt_data_fname: str) -> PromptTemplate:
    [model_id, input_vars, prompt_tpl] = json_to_dict(prompt_fname).values()
    prompt_data = json_to_dict(prompt_data_fname)
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
    return prompt


def _get_memory() -> BaseMemory:
    return ConversationBufferWindowMemory(k=5, ai_prefix="Assistant", human_prefix="Candidate")


def _get_chain(llm: Pipeline, prompt: PromptTemplate, memory: Union[BaseMemory,None] = None, verbose: bool = False):
    return ConversationChain(
        llm=HuggingFacePipeline(llm),
        prompt=prompt,
        memory=memory,
        verbose=verbose,
    )


def init(config: dict):
    # Run on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🏎️ Using GPU" if device.type == "cuda" else "🐌 Using CPU")
    
    config = RailsConfig.from_path(config.get("rails_cf_path"))
    app = LLMRails(config=config)
    
    llm = _get_pipeline(
        model_id=config.get("model_id"),
        stop_list=config.get("stop_list"),
        load_in_4bit=config.get("load_in_4bit"),
    )
    hf_llm = HuggingFacePipelineCompatible(pipeline=llm)
    provider = get_llm_instance_wrapper(
        llm_instance=hf_llm, llm_type="hf_pipeline_llama2_13b"
    )
    register_llm_provider("hf_pipeline_llama2_13b", provider)
    # prompt = _get_prompt(
    #     prompt_fname=config.get("prompt_fname"),
    #     prompt_data_fname=config.get("prompt_data_fname"),
    # )
    # memory = _get_memory()
    # chain = _get_chain(llm, prompt, memory, verbose=True)
    # app.register_action(chain, "chat_chain")
    return app
