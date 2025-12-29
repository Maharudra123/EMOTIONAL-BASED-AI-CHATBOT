import datetime
import requests
import wikipedia
import google.generativeai as genai
from transformers import pipeline
import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    text = re.sub(r'\*+([^\*]+)\*+', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = ' '.join(text.split())
    return text

def wishMe():
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        return "Hello, Good Morning!"
    elif 12 <= hour < 18:
        return "Hello, Good Afternoon!"
    else:
        return "Hello, Good Evening!"

def get_weather(city_name: str, api_key: str) -> str:
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"

    try:
        response = requests.get(complete_url)
        x = response.json()
        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"]
            current_humidity = y["humidity"]
            z = x["weather"]
            weather_description = z[0]["description"]
            return (f"Temperature in Kelvin is {current_temperature}, "
                    f"Humidity is {current_humidity} percent, "
                    f"and the weather description is {weather_description}.")
        else:
            return "City not found"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

def get_top_news(api_key: str) -> List[str]:
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    news_headlines = []
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok' and 'articles' in data:
            articles = data['articles']
            for i, article in enumerate(articles[:10]):
                news_headlines.append(f"News {i + 1}: {article['title']}")
        return news_headlines
    except Exception as e:
        return [f"Error fetching news: {str(e)}"]

class HybridAssistant:
    def __init__(self, google_api_key: str, weather_api_key: str, news_api_key: str):
        self.history = []
        genai.configure(api_key=google_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.chat = self.gemini_model.start_chat(history=self.history)

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        self.weather_api_key = weather_api_key
        self.news_api_key = news_api_key

    def detect_emotion(self, text: str) -> str:
        result = self.sentiment_pipeline(text)[0]
        return result['label']

    def generate_response(self, user_input: str, detected_emotion: str) -> str:
        response_templates = {
            "POSITIVE": "I'm glad you're feeling good! ",
            "NEGATIVE": "I understand that this might be tough. ",
            "NEUTRAL": "Got it! "
        }
        return response_templates.get(detected_emotion, "")

    def process_input(self, user_input: str) -> Tuple[str, str]:
        user_input = user_input.lower()

        if 'weather' in user_input:
            city_name = self.extract_city_name(user_input)
            if not city_name:
                return "Please specify a city name for the weather report.", ""
            response = get_weather(city_name, self.weather_api_key)
            return response, response

        elif 'time' in user_input:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            response = f"The time is {current_time}"
            return response, response

        elif 'news' in user_input:
            headlines = get_top_news(self.news_api_key)
            response = "\n".join(headlines)
            return response, response

        elif 'wikipedia' in user_input:
            query = user_input.replace("wikipedia", "").strip()
            try:
                results = wikipedia.summary(query, sentences=3)
                return f"According to Wikipedia: {results}", f"According to Wikipedia: {results}"
            except Exception:
                return "Sorry, I couldn't fetch the information from Wikipedia.", ""

        emotion = self.detect_emotion(user_input)
        emotional_context = self.generate_response(user_input, emotion)

        self.history.append({"role": "user", "parts": [user_input]})
        response = self.chat.send_message(user_input, stream=True)

        display_response = ""
        speech_response = ""
        for chunk in response:
            if chunk.text:
                display_response += chunk.text
                speech_response += clean_text(chunk.text)

        if display_response:
            self.history.append({"role": "model", "parts": [display_response]})

        final_display = emotional_context + display_response
        final_speech = emotional_context + speech_response
        return final_display, final_speech

    def extract_city_name(self, user_input: str) -> str:
        match = re.search(r'weather in ([a-zA-Z ]+)', user_input)
        return match.group(1).strip() if match else ""
