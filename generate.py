from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import json

load_dotenv()

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

class GenerateEmail():    
    def __init__(self, model: str):
        # initialize client once
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self, messages):
        """Call the OpenAI ChatCompletions API"""
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        return response.choices[0].message.content
    
    def get_prompt(self, prompt_name, prompt_type='user', **kwargs):
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)
    
    def send_prompt(self, user_prompt: str, system_msg="You are a helpful assistant."):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)
    
    def generate(self, action: str, email_content: str, tone: str = None) -> str:
        """Generate edited email based on action"""

        if action == "shorten":
            system_prompt = self.get_prompt('shorten', prompt_type='system')
            user_prompt = self.get_prompt('shorten', email_content=email_content)
            
        elif action == "lengthen":
            system_prompt = self.get_prompt('lengthen', prompt_type='system')
            user_prompt = self.get_prompt('lengthen', email_content=email_content)
            
        else:  # tone
            system_prompt = self.get_prompt('tone', prompt_type='system', tone=tone)
            user_prompt = self.get_prompt('tone', email_content=email_content, tone=tone)
        
        return self.send_prompt(user_prompt, system_prompt)
    
    def judge(self, original_email: str, edited_email: str) -> dict:
        """Judge the edited email on 6 metrics"""
        
        results = {}
        
        # Metrics that need both original and edited email
        comparison_metrics = ['judge_faithfulness', 'judge_completeness', 'judge_conciseness', 'judge_url_preservation']
        # Metrics that only need edited email
        single_metrics = ['judge_grammar_clarity', 'judge_tone_consistency']
        
        # Judge comparison metrics
        for metric in comparison_metrics:
            system_prompt = self.get_prompt(metric, prompt_type='system')
            user_prompt = self.get_prompt(metric,
                                          original_email=original_email,
                                          edited_email=edited_email)
            response = self.send_prompt(user_prompt, system_prompt)
            
            try:
                results[metric.replace('judge_', '')] = json.loads(response)
            except json.JSONDecodeError:
                results[metric.replace('judge_', '')] = {"score": "N/A", "explanation": response}
        
        # Judge single metrics (only need edited email)
        for metric in single_metrics:
            system_prompt = self.get_prompt(metric, prompt_type='system')
            user_prompt = self.get_prompt(metric, edited_email=edited_email)
            response = self.send_prompt(user_prompt, system_prompt)
            
            try:
                results[metric.replace('judge_', '')] = json.loads(response)
            except json.JSONDecodeError:
                results[metric.replace('judge_', '')] = {"score": "N/A", "explanation": response}
        
        return results
    
    def generate_synthetic(self, id: int, task_type: str, topic: str, 
                           persona: str, tone: str, length: str) -> dict:
        """Generate a synthetic email for dataset expansion"""
        
        system_prompt = self.get_prompt('synthetic_generate', prompt_type='system')
        user_prompt = self.get_prompt('synthetic_generate',
                                       id=id,
                                       task_type=task_type,
                                       topic=topic,
                                       persona=persona,
                                       tone=tone,
                                       length=length)
        
        response = self.send_prompt(user_prompt, system_prompt)
        
        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_response": response}