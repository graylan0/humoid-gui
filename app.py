import tkinter as tk
import threading
import os
import aiosqlite
import logging
import numpy as np
import base64
import queue
import uuid
import customtkinter
import requests
from customtkinter import CTkImage
import io
import sys
import random
import asyncio
import weaviate
from concurrent.futures import ThreadPoolExecutor
from summa import summarizer
from textblob import TextBlob
from weaviate.util import generate_uuid5
from PIL import Image, ImageTk
from llama_cpp import Llama
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
import nltk
import json
from os import path


bundle_dir = path.abspath(path.dirname(__file__))
path_to_config = path.join(bundle_dir, 'config.json')
model_path = path.join(bundle_dir, 'llama-2-7b-chat.ggmlv3.q8_0.bin')
logo_path = path.join(bundle_dir, 'logo.png')

def download_nltk_data():
    try:
        # Define the resources to download
        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }

        # Check and download each resource
        for path, package in resources.items():
            try:
                nltk.data.find(path)
                print(f"'{package}' already downloaded.")
            except LookupError:
                nltk.download(package)
                print(f"'{package}' downloaded successfully.")

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")



q = queue.Queue()
logger = logging.getLogger(__name__)


def load_config(file_path=path_to_config):
    with open(file_path, 'r') as file:
        return json.load(file)


config = load_config()
DB_NAME = config['DB_NAME']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_QUERY_PATH = config['WEAVIATE_QUERY_PATH']

client = weaviate.Client(
    url=WEAVIATE_ENDPOINT,
)
weaviate_client = weaviate.Client(url=WEAVIATE_ENDPOINT)

async def init_db():
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    response TEXT,
                    response_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INT
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    summarization_context TEXT,
                    full_text TEXT
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    relationship_state TEXT
                )
            """)

            await db.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=3900,
)


def llama_generate(prompt, weaviate_client=None):
    config = load_config()
    max_tokens = config.get('MAX_TOKENS', 3999)  # Default to 3999 if not specified
    chunk_size = config.get('CHUNK_SIZE', 1250)  # Default to 1250 if not specified

    try:
        
        def find_max_overlap(chunk, next_chunk):
            max_overlap = min(len(chunk), 400)
            for overlap in range(max_overlap, 0, -1):
                if chunk.endswith(next_chunk[:overlap]):
                    return overlap
            return 0

        def determine_token(chunk):

            words = word_tokenize(chunk)
            tagged_words = pos_tag(words)
            verbs = [word for word, tag in tagged_words if tag.startswith('VB')]

            if verbs:
                return "[action]"
            else:
                return "[attention]"

        def fetch_relevant_info(chunk, weaviate_client):

            def truncate_text(text, max_words=25):
                words = text.split()
                return ' '.join(words[:max_words])

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

                if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']                ['Get']:
                    interaction = response['data']['Get']['InteractionHistory'][0]
                    user_message = truncate_text(interaction['user_message'])
                    ai_response = truncate_text(interaction['ai_response'])
                    return f"{user_message} {ai_response}"
                else:
                    return ""
            return ""

        def tokenize_and_generate(chunk, token):
            inputs = llm(f"[{token}] {chunk}", max_tokens=min(max_tokens, chunk_size))

            if not isinstance(inputs, dict):
                logger.error(f"Output from Llama is not a dictionary: {type(inputs)}")
                return None

            choices = inputs.get('choices', [])
            if not choices or not isinstance(choices[0], dict):
                logger.error(f"No valid choices in Llama output")
                return None

            output = choices[0].get('text', '')
            if not output:
                logger.error(f"No text found in Llama output")
                return None

            return output

        prompt_chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
        responses = []
        last_output = ""

        for i, chunk in enumerate(prompt_chunks):
            relevant_info = fetch_relevant_info(chunk, weaviate_client)
            combined_chunk = f"{relevant_info} {chunk}"
            

            token = determine_token(combined_chunk)
            output = tokenize_and_generate(combined_chunk, token)

            if i > 0 and last_output:
                overlap = find_max_overlap(last_output, output)
                output = output[overlap:]

            responses.append(output)
            last_output = output

        final_response = ''.join(responses)
        return final_response
    except Exception as e:
        logger.error(f"Error in llama_generate: {e}")
        return None


def run_async_in_thread(loop, coro_func, *args):
    try:
        asyncio.set_event_loop(loop)
        coro = coro_func(*args)
        loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error in async thread: {e}")
    finally:
        loop.close()

def truncate_text(self, text, max_length=35):
    return text if len(text) <= max_length else text[:max_length] + '...'  


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def retrieve_past_interactions(self, generated_reply, result_queue):
        try:
            def sync_query():
                # Define the query as a string
                query = f"""
                {{
                    Get {{
                        InteractionHistory(nearText: {{
                            concepts: ["{generated_reply}"],
                            certainty: 0.7
                            .with_limit(111)
                        }}) {{
                            user_message
                            ai_response
                        }}
                    }}
                }}
                """
                # Pass the query string directly to the raw method
                return self.client.query.raw(query)

            def truncate_text(text, max_length=35):
                return text if len(text) <= max_length else text[:max_length] + '...'

            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(executor, sync_query)

            if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                interactions = response['data']['Get']['InteractionHistory']

                processed_interactions = []
                for interaction in interactions:
                    truncated_user_message = truncate_text(interaction['user_message'], 100)
                    truncated_ai_response = truncate_text(interaction['ai_response'], 100)
                    summarized_interaction = summarizer.summarize(f"{truncated_user_message} {truncated_ai_response}")
                    sentiment = TextBlob(summarized_interaction).sentiment.polarity

                    processed_interactions.append({
                        "user_message": truncated_user_message,
                        "ai_response": truncated_ai_response,
                        "summarized_interaction": summarized_interaction,
                        "sentiment": sentiment
                    })

                result_queue.put(processed_interactions)
            else:
                logger.error("No interactions found for the given generated reply.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])




    def process_response_and_store_in_weaviate(self, user_message, ai_response):
        response_blob = TextBlob(ai_response)
        keywords = response_blob.noun_phrases
        sentiment = response_blob.sentiment.polarity

        interaction_object = {
            "userMessage": user_message,
            "aiResponse": ai_response,
            "keywords": list(keywords),
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


    def run_async_in_thread(loop, coro_func, message, result_queue):
        asyncio.set_event_loop(loop)
        coro = coro_func(message, result_queue)
        loop.run_until_complete(coro)

    def generate_response(self, message):
        try:
            result_queue = queue.Queue()
            loop = asyncio.new_event_loop()
            past_interactions_thread = threading.Thread(target=run_async_in_thread, args=(loop, self.retrieve_past_interactions, message, result_queue))
            past_interactions_thread.start()
            past_interactions_thread.join()

            past_interactions = result_queue.get()

            past_context = "\n".join([f"User: {interaction['user_message']}\nAI: {interaction['ai_response']}" for interaction in past_interactions])
            complete_prompt = f"{past_context}\nUser: {message}"

            response = llama_generate(complete_prompt)
            if response:
                response_text = response
                self.response_queue.put({'type': 'text', 'data': response_text})

                keywords = self.extract_keywords(message)
                mapped_classes = self.map_keywords_to_weaviate_classes(keywords, message)

                self.create_interaction_history_object(message, response_text)
            else:
                logger.error("No response generated by llama_generate")

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")


    def on_submit(self, event=None):
        download_nltk_data()
        message = self.input_textbox.get("1.0", tk.END).strip()
        if message:
            self.text_box.insert(tk.END, f"You: {message}\n")
        
            # Clear the input_textbox after processing
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)  # Reset the height

            self.text_box.see(tk.END)
            self.executor.submit(self.generate_response, message)
            self.executor.submit(self.generate_images, message)
            self.after(100, self.process_queue)

        return "break"  # Prevents the default 'Return' binding

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
                    self.text_box.insert(tk.END, f"AI: {response['data']}\n")
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
            payload = {
                "prompt": message,
                "steps": 79,
                "seed": random.randrange(sys.maxsize),
                "enable_hr": "false",
                "denoising_strength": "0.7",
                "cfg_scale": "7",
                "width": 326,
                "height": 656,
                "restore_faces": "true",
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                try:
                    r = response.json()
                    for img_data in r['images']:
                        # Check if img_data contains a comma and split accordingly
                        if ',' in img_data:
                            base64_data = img_data.split(",", 1)[1]
                        else:
                            base64_data = img_data

                        # Decode and open the image
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        img_tk = ImageTk.PhotoImage(image)

                        # Display the image in the GUI
                        self.response_queue.put({'type': 'image', 'data': img_tk})
                        self.image_label.image = img_tk

                        # Save the image to a file
                        file_name = f"generated_image_{uuid.uuid4()}.png"
                        image_path = os.path.join("saved_images", file_name)
                        if not os.path.exists("saved_images"):
                            os.makedirs("saved_images")
                        image.save(image_path)

                except ValueError as e:
                    print("Error processing image data: ", e)
            else:
                print("Error generating image: ", response.status_code)

        except Exception as e:
            logger.error(f"Error in generate_images: {e}")


    def setup_gui(self):
        self.title("OneLoveIPFS AI")
        window_width = 1200
        window_height = 900
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=440, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        logo_img = Image.open(logo_path)




        logo_photo = ImageTk.PhotoImage(logo_img)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo)
        self.logo_label.image = logo_photo
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.image_label = customtkinter.CTkLabel(self.sidebar_frame)
        self.image_label.grid(row=1, column=0, padx=20, pady=10)
        placeholder_image = Image.new('RGB', (140, 140), color = (73, 109, 137))
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)
        self.image_label.configure(image=self.placeholder_photo)
        self.image_label.image = self.placeholder_photo
        # Text Box for Display (No changes here)
        self.text_box = customtkinter.CTkTextbox(self, bg_color="white", text_color="white", border_width=0, height=260, width=50, font=customtkinter.CTkFont(size=18))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Frame for the new Text Input Box
        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)

        # New Text Input Box
        self.input_textbox = tk.Text(self.input_textbox_frame, font=("Roboto Medium", 10),
                                     bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
                                     fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0], relief="flat", height=1)
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")

        # Scrollbar for the new Text Input Box
        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)

        # Send Button (No changes here)
        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)


if __name__ == "__main__":
    try:
        
        app = App()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(init_db())
        app.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
