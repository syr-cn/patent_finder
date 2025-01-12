from .gpt import modelGPT4o, modelGPTo1
from .vllm import modelVLLM
from .gemini import model_gemini_flash, model_gemini_pro
from .claude import model_claude_opus, model_claude_sonnet3_5, model_claude_sonnet3
import random
import json
from pydantic import BaseModel

REFORMAT_PROMPT = """\
You are an expert in chemical patents and molecular representations. Above is the response from a chemical expert system. Your task is to reformat the response into a structured format that adheres to the specified schema. The output should be a JSON instance that conforms to the JSON schema provided below.

Note that yous should not change the content of the response, but only reformat it into a structured format. Do not include any personal opinions or additional information in the output.
"""

def get_llm(llm_name):
    return {
        "gpt-4o": modelGPT4o,
        "gpt-o1": modelGPTo1,
        # "gpt-4": modelGPT4,
        "vllm": modelVLLM,
        "vllm-gpt_reformat": modelVLLM,
        "gemini_flash": model_gemini_flash,
        "gemini_pro": model_gemini_pro,
        "claude_opus": model_claude_opus,
        "claude_sonnet3_5": model_claude_sonnet3_5,
        "claude_sonnet3": model_claude_sonnet3,
    }[llm_name]

class StructuredLLM:
    def __init__(self, llm_name='gpt-4o'):
        self.llm_name = llm_name
        self.llm_model = get_llm(llm_name)
    
    def wrap_messages(self, messages):
        if self.llm_name == 'gpt-o1': # o1 does not support system prompt
            for message in messages:
                message['role'] = 'user'
        return messages

    def chat(self, messages, return_format: BaseModel=None):
        messages = self.wrap_messages(messages)
        if not return_format:
            return self.llm_model.invoke(messages).content

        if self.llm_name in [
            'gpt-o1', 'vllm-gpt_reformat'
        ]: # does not support formatted output naturally
            return self.chat_and_reformat(messages, return_format)
        if self.llm_name == 'vllm': # does not support formatted output naturally
            return self.chat_and_reformat_vllm(messages, return_format)

        structured_llm = self.llm_model.with_structured_output(return_format)
        response = structured_llm.invoke(messages)
        if not isinstance(response, dict):
            response = response.dict()
        return response

    def chat_and_reformat_vllm(self, messages, return_format: BaseModel):
        raw_response = self.llm_model.invoke(messages).content
        formatted_response = self.llm_model.reformat(raw_response, return_format)
        formatted_response['raw_reasoning'] = raw_response
        return formatted_response

    def chat_and_reformat(self, messages, return_format: BaseModel, reformat_model='gpt-4o'):
        raw_response = self.chat(messages)
        return self.reformat(raw_response, return_format, reformat_model)

    def reformat(self, raw_response: str, return_format: BaseModel, reformat_model='gpt-4o'):
        reformat_messages = [
            {
                "role": "system",
                "content": REFORMAT_PROMPT
            },
            {
                "role": "user",
                "content": raw_response
            }
        ]

        reformat_model = get_llm(reformat_model)
        structured_llm = reformat_model.with_structured_output(return_format)
        formatted_response = structured_llm.invoke(reformat_messages)
        if not isinstance(formatted_response, dict):
            formatted_response = formatted_response.dict()
        formatted_response['raw_reasoning'] = raw_response
        return formatted_response