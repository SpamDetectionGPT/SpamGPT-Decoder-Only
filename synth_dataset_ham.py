from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import random

load_dotenv()
deep_seek_key = os.getenv('DEEPSEEK_KEY')

client = OpenAI(api_key=deep_seek_key, base_url="https://api.deepseek.com")

with open("combined_ham.json", "rb") as h:
    ham = json.load(h)
    ham = ham["dataset"]


while True:
    try:
        chosen_examples = random.sample(ham, 5)
        chosen_examples = {"dataset": chosen_examples}
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a data synthesiser."},
                {"role": "user", "content": f"Please generate a dataset of benign emails displaying variety. (This is for the negative class of a spam/phishing dataset) \
                    The dataset should be a list of JSON objects each containing a single \"text\" field. Please make each email around 500 words long. Please generate 50 examples. ONLY produce a list with JSON NO explaination text!!! ALWAYS keep the structure of the examples.\nExample: {chosen_examples}"}
            ],
            stream=False,
            temperature=0.7,
            max_tokens=8000,
            response_format={
            'type': 'json_object'
            }
        )
        print(response.choices[0].message.content)

        with open('combined_ham.json', 'r') as f:
            dataset = json.load(f)
            print(response.choices[0].message.content)
            dataset["dataset"] += json.loads(response.choices[0].message.content)["dataset"]
            with open('combined_ham.json', 'w') as new_f:
                json.dump(dataset, new_f, indent=4)
        print("Dataset updated successfully.")
        if len(dataset["dataset"]) > 350000:
                print("Reached target number")
                break
    except:
        print("Error")
