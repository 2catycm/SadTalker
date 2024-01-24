import json
import yaml
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
get_path_under_here = lambda s:(this_directory/str(s)).as_posix()
def get_file_content_str(filepath)->str:
    with open(filepath, 'r', encoding='utf-8') as in_file:
        return in_file.read()
def load_yaml_file(file_path: str) -> dict:
    """Reads and returns the contents of a YAML file as dictionary"""
    return yaml.safe_load(get_file_content_str(file_path))

def load_local_yaml_prompt_here(file_path):
    file_path = get_path_under_here(file_path)
    json_template = load_yaml_file(file_path)
    return json_template['chat_prompt'], json_template['system_prompt']

import openai
def gpt3Turbo_completion(chat_prompt="", system="You are an AI that can give the answer to anything", 
                         temp=0.7, model="gpt-3.5-turbo", 
                         max_tokens=1000, remove_nl=True, 
                         conversation=None):
    openai.api_key = ApiKeyManager.get_api_key("OPENAI")
    max_retry = 5
    retry = 0
    while True:
        try:
            if conversation:
                messages = conversation
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": chat_prompt}
                ]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp)
            text = response['choices'][0]['message']['content'].strip()
            if remove_nl:
                text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('.logs/gpt_logs'):
                os.makedirs('.logs/gpt_logs')
            with open('.logs/gpt_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
                outfile.write(f"System prompt: ===\n{system}\n===\n"+f"Chat prompt: ===\n{chat_prompt}\n===\n" + f'RESPONSE:\n====\n{text}\n===\n')
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                raise Exception("GPT3 error: %s" % oops)
            print('Error communicating with OpenAI:', oops)
            sleep(1)