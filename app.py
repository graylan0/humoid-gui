import torch
import tkinter as tk
import customtkinter
import threading
import os
import aiosqlite
import weaviate
import logging
import numpy as np
import base64
import queue
import uuid
import requests
import io
import sys
import random
import asyncio
import re
import uvicorn
import json
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
from llama_cpp import Llama
from os import path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from collections import Counter
from bark import SAMPLE_RATE, generate_audio, preload_models
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from summa import summarizer
import nltk
from textblob import TextBlob
from weaviate.util import generate_uuid5
from nltk import pos_tag, word_tokenize
import httpx
from weaviate.util import generate_uuid5
from nltk.corpus import wordnet as wn
from datetime import datetime
import aiosqlite
import uuid
import json
#rom elevenlabs import generate, play
import asyncio
#from elevenlabs import set_api_key
import concurrent.futures
from weaviate.util import generate_uuid5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "false"




executor = ThreadPoolExecutor(max_workers=5)
bundle_dir = path.abspath(path.dirname(__file__))
path_to_config = path.join(bundle_dir, 'config.json')
model_path = path.join(bundle_dir, 'llama-2-7b-chat.ggmlv3.q8_0.bin')
logo_path = path.join(bundle_dir, 'logo.png')


API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def load_config(file_path=path_to_config):
    with open(file_path, 'r') as file:
        return json.load(file)

q = queue.Queue()

config = load_config()
ELEVEN_LABS_KEY =  config['ELEVEN_LABS_KEY']
#set_api_key(ELEVEN_LABS_KEY)
DB_NAME = config['DB_NAME']
API_KEY = config['API_KEY']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_API_URL = config['WEAVIATE_API_URL']
weaviate_client = weaviate.Client(url=WEAVIATE_ENDPOINT)
weaviate_api_url = WEAVIATE_API_URL
app = FastAPI()


def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def generate_uuid_for_weaviate():
    return str(uuid.uuid4())


def is_valid_uuid(uuid_to_test):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=4)
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False
    

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    

def get_current_multiversal_time():
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    x, y, z, t = 34, 76, 12, 5633
    return f"X:{x}, Y:{y}, Z:{z}, T:{t}, Time:{current_time}"


async def init_db():
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS local_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    response TEXT,
                    response_time TEXT
                )
            """)
            await db.commit()

        interaction_history_class = {
            "class": "InteractionHistory",
            "properties": [
                {"name": "user_id", "dataType": ["string"]},
                {"name": "response", "dataType": ["string"]},
                {"name": "response_time", "dataType": ["string"]}
            ]
        }

        existing_classes = weaviate_client.schema.get()['classes']
        if not any(cls['class'] == 'InteractionHistory' for cls in existing_classes):
            weaviate_client.schema.create_class(interaction_history_class)

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


async def save_user_message(user_id, user_input):
    logger.info(f"save_user_message called with user_id: {user_id}, user_input: {user_input}")

    if not user_input:
        logger.error("User input is None or empty.")
        return

    try:
        response_time = get_current_multiversal_time()
        data_object = {
            "user_id": user_id,
            "response": user_input,
            "response_time": response_time
        }
        generated_uuid = generate_uuid5(data_object)

        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute("INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                             (user_id, user_input, response_time))
            await db.commit()

        async with httpx.AsyncClient() as client:
            await client.post(weaviate_api_url, json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": data_object
            })

    except Exception as e:
        logger.error(f"Error saving user message: {e}")

async def save_bot_response(bot_id, bot_response):
    logger.info(f"save_bot_response called with bot_id: {bot_id}, bot_response: {bot_response}")

    if not bot_response:
        logger.error("Bot response is None or empty.")
        return

    try:
        response_time = get_current_multiversal_time()
        data_object = {
            "user_id": bot_id,
            "response": bot_response,
            "response_time": response_time
        }
        generated_uuid = generate_uuid5(data_object)

        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute("INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                             (bot_id, bot_response, response_time))
            await db.commit()

        async with httpx.AsyncClient() as client:
            await client.post(weaviate_api_url, json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": data_object
            })

    except Exception as e:
        logger.error(f"Error saving bot response: {e}")

def download_nltk_data():
    try:

        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }


        for path, package in resources.items():
            try:
                nltk.data.find(path)
                print(f"'{package}' already downloaded.")
            except LookupError:
                nltk.download(package)
                print(f"'{package}' downloaded successfully.")

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


class UserInput(BaseModel):
    message: str

@app.post("/process/")
async def process_input(user_input: UserInput, api_key: str = Depends(get_api_key)):
    try:
        response = llama_generate(user_input.message, weaviate_client)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=3990,
)


def is_code_like(chunk):
   code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
   return bool(re.search(code_patterns, chunk))


def determine_token(chunk, memory, max_words_to_check=1100):
   combined_chunk = f"{memory} {chunk}"
   if not combined_chunk:
       return "[attention]"

   if is_code_like(combined_chunk):
       return "[code]"

   words = word_tokenize(combined_chunk)[:max_words_to_check]
   tagged_words = pos_tag(words)

   pos_counts = Counter(tag[:2] for _, tag in tagged_words)
   most_common_pos, _ = pos_counts.most_common(1)[0]

   if most_common_pos == 'VB':
       return "[action]"
   elif most_common_pos == 'NN':
       return "[subject]"
   elif most_common_pos in ['JJ', 'RB']:
       return "[description]"
   else:
       return "[general]"


def find_max_overlap(chunk, next_chunk):
   max_overlap = min(len(chunk), 540)
   return next((overlap for overlap in range(max_overlap, 0, -1) if chunk.endswith(next_chunk[:overlap])), 0)


def truncate_text(text, max_words=600):
   return ' '.join(text.split()[:max_words])


def fetch_relevant_info(chunk, weaviate_client, user_input):
   if not user_input:
       logger.error("User input is None or empty.")
       return ""

   summarized_chunk = summarizer.summarize(chunk)
   query_chunk = summarized_chunk if summarized_chunk else chunk

   if not query_chunk:
       logger.error("Query chunk is empty.")
       return ""

   query = {
       "query": {
           "nearText": {
               "concepts": [user_input],
               "certainty": 0.7
           }
       }
   }

   try:
       response = weaviate_client.query.raw(json.dumps(query))
       logger.debug(f"Query sent: {json.dumps(query)}")
       logger.debug(f"Response received: {response}")

       if response and 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
           interaction = response['data']['Get']['InteractionHistory'][0]
           return f"{interaction['user_message']} {interaction['ai_response']}"
       else:
           logger.error("Weaviate client returned no relevant data for query: " + json.dumps(query))
           return ""
   except Exception as e:
       logger.error(f"Weaviate query failed: {e}")
       return ""
   

def llama_generate(prompt, weaviate_client=None, user_input=None):
   config = load_config()
   max_tokens = config.get('MAX_TOKENS', 3990)
   chunk_size = config.get('CHUNK_SIZE', 1558)
   try:
       prompt_chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
       responses = []
       last_output = ""
       memory = ""

       for i, current_chunk in enumerate(prompt_chunks):
           relevant_info = fetch_relevant_info(current_chunk, weaviate_client, user_input)
           combined_chunk = f"{relevant_info} {current_chunk}"
           token = determine_token(combined_chunk, memory)
           output = tokenize_and_generate(combined_chunk, token, max_tokens, chunk_size)

           if output is None:
               logger.error(f"Failed to generate output for chunk: {combined_chunk}")
               continue

           if i > 0 and last_output:
               overlap = find_max_overlap(last_output, output)
               output = output[overlap:]

           memory += output
           responses.append(output)
           last_output = output

       final_response = ''.join(responses)
       return final_response if final_response else None
   except Exception as e:
       logger.error(f"Error in llama_generate: {e}")
       return None


def tokenize_and_generate(chunk, token, max_tokens, chunk_size):
   try:
       inputs = llm(f"[{token}] {chunk}", max_tokens=min(max_tokens, chunk_size))
       if inputs is None or not isinstance(inputs, dict):
           logger.error(f"Llama model returned invalid output for input: {chunk}")
           return None

       choices = inputs.get('choices', [])
       if not choices or not isinstance(choices[0], dict):
           logger.error("No valid choices in Llama output")
           return None

       return choices[0].get('text', '')
   except Exception as e:
       logger.error(f"Error in tokenize_and_generate: {e}")
       return None


def run_async_in_thread(self, loop, coro_func, user_input, result_queue):
    try:
        asyncio.set_event_loop(loop)
        coro = coro_func(user_input, result_queue)
        loop.run_until_complete(coro)
    finally:
        loop.close()


def truncate_text(self, text, max_length=95):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return text if len(text) <= max_length else text[:max_length] + '...'

    except Exception as e:

        print(f"Error in truncate_text: {e}")
        return ""


def extract_verbs_and_nouns(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        verbs_and_nouns = [word for word, tag in tagged_words if tag.startswith('VB') or tag.startswith('NN')]
        return verbs_and_nouns

    except Exception as e:

        print(f"Error in extract_verbs_and_nouns: {e}")
        return []


class App(customtkinter.CTk):
    def __init__(self, user_identifier):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.loop = asyncio.get_event_loop()
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        self.executor = ThreadPoolExecutor(max_workers=4)


    async def retrieve_past_interactions(self, user_input, result_queue):
        try:

            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)


            def fetch_relevant_info(chunk, weaviate_client):
                if weaviate_client:
                    query = f"""
                    {{
                        Get {{
                            InteractionHistory(nearText: {{
                                concepts: ["{chunk}"],
                                certainty: 0.7
                            }}) {{
                                user_message
                                ai_response
                                .with_limit(1)
                            }}
                        }}
                    }}
                    """
                    response = weaviate_client.query.raw(query)

                    if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                        interaction = response['data']['Get']['InteractionHistory'][0]
                        return interaction['user_message'], interaction['ai_response']
                    else:
                        return "", ""
                return "", ""

            user_message, ai_response = fetch_relevant_info(concepts_query, self.client)

            if user_message and ai_response:
                summarized_interaction = summarizer.summarize(f"{user_message} {ai_response}")
                sentiment = TextBlob(summarized_interaction).sentiment.polarity
                processed_interaction = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized_interaction,
                    "sentiment": sentiment
                }
                result_queue.put([processed_interaction])
            else:
                logger.error("No relevant interactions found for the given user input.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])


    def process_response_and_store_in_weaviate(self, user_message, ai_response):
        response_blob = TextBlob(ai_response)
        keywords = response_blob.noun_phrases
        sentiment = response_blob.sentiment.polarity
        enhanced_keywords = set()
        for phrase in keywords:
            enhanced_keywords.update(phrase.split())

        interaction_object = {
            "userMessage": user_message,
            "aiResponse": ai_response,
            "keywords": list(enhanced_keywords),
            "sentiment": sentiment
        }

        interaction_uuid = str(uuid.uuid4())

        try:
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )

            print(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")

        except Exception as e:            
            print(f"Error storing interaction in Weaviate: {e}")


    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)


    def create_interaction_history_object(self, user_message, ai_response):
        interaction_object = {
            "user_message": user_message,
            "ai_response": ai_response
        }

        try:
            object_uuid = uuid.uuid4()
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            print(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating interaction history object in Weaviate: {e}")


    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:

            summarized_context = summarizer.summarize(context)
        except Exception as e:
            print(f"Error in summarizing context: {e}")
            summarized_context = context

        try:

            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment

        positive_class_mappings = {
            "keyword1": "PositiveClassA",
                "keyword2": "PositiveClassB",

        }

        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",

        }

        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",

        }

        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping
            
        mapped_classes = {}
        for keyword in keywords:
            try:
                if keyword in mapping:
                    mapped_classes[keyword] = mapping[keyword]
            except KeyError as e:
                print(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes


    async def retrieve_past_interactions(self, user_input, result_queue):
        try:

            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)


            query = f"""
            {{
                Get {{
                    InteractionHistory(nearText: {{
                        concepts: ["{concepts_query}"],
                        certainty: 0.8
                    }}) {{
                        user_message
                        ai_response
                        .with_limit(12)
                    }}
                }}
            }}
            """
            response = self.client.query.raw(query)
            if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                interactions = response['data']['Get']['InteractionHistory']
                result_queue.put(interactions)
            else:
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])


    async def generate_response(self, user_input):
        try:
            if not user_input:
                logger.error("User input is None or empty.")
                return

            user_id = self.user_id
            bot_id = self.bot_id

           
            await save_user_message(user_id, user_input)

            
            include_past_context = "[pastcontext]" in user_input
            user_input = user_input.replace("[pastcontext]", "").replace("[/pastcontext]", "")
            past_context = ""

            if include_past_context:
                result_queue = queue.Queue()
                await self.retrieve_past_interactions(user_input, result_queue)
                past_interactions = result_queue.get()
                if past_interactions:
                    past_context_combined = "\n".join(
                        [f"User: {interaction['user_message']}\nAI: {interaction['ai_response']}" 
                         for interaction in past_interactions])
                    past_context = past_context_combined[-1500:]

            complete_prompt = f"{past_context}\nUser: {user_input}"
            logger.info(f"Generating response for prompt: {complete_prompt}")

       
            response = llama_generate(complete_prompt, self.client)
            if response:
                logger.info(f"Generated response: {response}")

               
                await save_bot_response(bot_id, response)

                
                self.process_generated_response(response)
            else:
                logger.error("No response generated by llama_generate")

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")


    def process_generated_response(self, response_text):
        try:
            self.response_queue.put({'type': 'text', 'data': response_text})
            self.play_response_audio(response_text)
        except Exception as e:
            logger.error(f"Error in process_generated_response: {e}")

#    def play_response_audio(self, response_text):
#        try:
#           
#            audio = generate(
##                text=response_text,
 #               voice="Bill",
#                model="eleven_multilingual_v2"  
#            )

            
#            play(audio)

    def play_response_audio(self, response_text):
        try:
            sentences = re.split('(?<=[.!?]) +', response_text)
            silence = np.zeros(int(0.75 * SAMPLE_RATE))

            def generate_sentence_audio(sentence):
                try:
                    return generate_audio(sentence, history_prompt="v2/en_speaker_6")
                except Exception as e:
                    logger.error(f"Error generating audio for sentence '{sentence}': {e}")
                    return np.zeros(0)

            with ThreadPoolExecutor(max_workers=min(2, len(sentences))) as executor:
                futures = [executor.submit(generate_sentence_audio, sentence) for sentence in sentences]

            audio_arrays = []
            for future in concurrent.futures.as_completed(futures):
                audio = future.result()
                if audio.size > 0:
                    audio_arrays.append(audio)

            if audio_arrays:
                pieces = [piece for audio in audio_arrays for piece in (audio, silence.copy())]
                audio = np.concatenate(pieces[:-1])

                file_name = str(uuid.uuid4()) + ".wav"
                write_wav(file_name, SAMPLE_RATE, audio)
                sd.play(audio, samplerate=SAMPLE_RATE)
            else:
                logger.error("No audio generated due to errors in all sentences.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in play_response_audio: {e}")

    def run_async_in_thread(self, loop, coro_func, user_input, result_queue):
        asyncio.set_event_loop(loop)
        coro = coro_func(user_input, result_queue)
        loop.run_until_complete(coro)
        

    async def fetch_interactions(self):
        try:
            query = {
                "query": """
                {
                    Get {
                        InteractionHistory(sort: [{path: "response_time", order: desc}], limit: 15) {
                            user_message
                            ai_response
                            response_time
                        }
                    }
                }
                """
            }
            response = self.client.query.raw(query)
            if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                interactions = response['data']['Get']['InteractionHistory']
                return [{'user_message': interaction['user_message'], 'ai_response': interaction['ai_response'], 'response_time': interaction['response_time']} for interaction in interactions]
            else:
                return []
        except Exception as e:
            logger.error(f"Error fetching interactions from Weaviate: {e}")
            return []


    def on_submit(self, event=None):
        user_input = self.input_textbox.get("1.0", tk.END).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)
            self.executor.submit(asyncio.run, self.generate_response(user_input))
            self.executor.submit(self.generate_images, user_input)
            self.after(100, self.process_queue)

        return "break"


    def create_object(self, class_name, object_data):

        unique_string = f"{object_data['time']}-{object_data['user_message']}-{object_data['ai_response']}"


        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex


        try:
            self.client.data_object.create(object_data, object_uuid, class_name)
            print(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating object in Weaviate: {e}")

        return object_uuid


    def process_queue(self):
        try:
            while True:
                response = self.response_queue.get_nowait()
                if response['type'] == 'text':
                    self.text_box.insert(tk.END, f"\n  \n \n AI: {response['data']}\n")
                elif response['type'] == 'image':
                    self.image_label.configure(image=response['data'])
                    self.image_label.image = response['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)


    def extract_keywords(self, message):
        try:
            blob = TextBlob(message)
            nouns = blob.noun_phrases
            return list(nouns)
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []


    def generate_images(self, message):
        try:
            url = config['IMAGE_GENERATION_URL']
            payload = self.prepare_image_generation_payload(message)
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                self.process_image_response(response)
            else:
                logger.error(f"Error generating image: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error in generate_images: {e}")


    def prepare_image_generation_payload(self, message):
        return {
            "prompt": message,
            "steps": 51,
            "seed": random.randrange(sys.maxsize),
            "enable_hr": "false",
            "denoising_strength": "0.7",
            "cfg_scale": "7",
            "width": 526,
            "height": 756,
            "restore_faces": "true",
        }


    def process_image_response(self, response):
        try:
            image_data = response.json()['images']
            for img_data in image_data:
                img_tk = self.convert_base64_to_tk(img_data)
                if img_tk is not None:
                    self.response_queue.put({'type': 'image', 'data': img_tk})
                    self.save_generated_image(img_tk)
                else:
                    logger.error("Failed to convert base64 to tk")
        except ValueError as e:
            logger.error("Error processing image data: ", e)


    def convert_base64_to_tk(self, base64_data):
        if ',' in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        image_data = base64.b64decode(base64_data)
        try:
            image = Image.open(io.BytesIO(image_data))
            return ImageTk.PhotoImage(image)
        except Exception as e:
            logger.error(f"Error converting base64 to tk: {e}")
            return None


    def save_generated_image(self, img_tk):
        file_name = f"generated_image_{uuid.uuid4()}.png"
        image_path = os.path.join("saved_images", file_name)
        if not os.path.exists("saved_images"):
            os.makedirs("saved_images")
        try:
            img_pil = ImageTk.getimage(img_tk)
            img_pil.save(image_path)
            print(f"Image saved to {image_path}")
        except IOError as e:
            logger.error(f"Error saving image: {e}")


    def update_username(self):
        """Update the username based on the input field."""
        new_username = self.username_entry.get()
        if new_username:
            self.user_id = new_username
            print(f"Username updated to: {self.user_id}")
        else:
            print("Please enter a valid username.")


    def setup_gui(self):

        self.title("OneLoveIPFS AI")
        window_width = 1400
        window_height = 1000
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=900, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        logo_img = Image.open(logo_path)
        logo_photo = ImageTk.PhotoImage(logo_img)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo)
        self.logo_label.image = logo_photo
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.image_label = customtkinter.CTkLabel(self.sidebar_frame)
        self.image_label.grid(row=3, column=0, padx=20, pady=10)
        placeholder_image = Image.new('RGB', (140, 140), color = (73, 109, 137))
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)
        self.image_label.configure(image=self.placeholder_photo)
        self.image_label.image = self.placeholder_photo
        self.text_box = customtkinter.CTkTextbox(self, bg_color="white", text_color="white", border_width=0, height=260, width=50, font=customtkinter.CTkFont(size=20))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)
        self.input_textbox = tk.Text(self.input_textbox_frame, font=("Roboto Medium", 13),
                                     bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
                                     fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0], relief="flat", height=1)
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")
        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)
        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)

        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.username_label = customtkinter.CTkLabel(self.settings_frame, text="Username:")
        self.username_label.grid(row=0, column=0, padx=5, pady=5)

        self.username_entry = customtkinter.CTkEntry(self.settings_frame, width=120, placeholder_text="Enter username")
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.username_entry.insert(0, "gray00")

        self.update_username_button = customtkinter.CTkButton(self.settings_frame, text="Update", command=self.update_username)
        self.update_username_button.grid(row=0, column=2, padx=5, pady=5)


if __name__ == "__main__":
    try:
        user_id = "gray00"
        app = App(user_id)
        loop = asyncio.get_event_loop()
        asyncio.run(init_db())
        app.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
