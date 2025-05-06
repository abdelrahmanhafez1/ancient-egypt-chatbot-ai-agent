#libraries
import os
import json
import uvicorn
from fastapi import FastAPI
from langdetect import detect
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.tools import tool
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from langchain_core.prompts import PromptTemplate
from crewai import Agent, Task, Crew, Process ,LLM
from langchain_core.language_models import BaseLanguageModel

# LLM-API
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
chat_history_file = "outputs/chat_history.txt"
session_log = []

# تحميل المتغيرات من .env
load_dotenv()
api_keys = os.getenv("SAMBANOVA_API_KEYS", "").split(",")


# basic_llm = LLM(
#     model="sambanova/Meta-Llama-3.3-70B-Instruct",
#     temperature=0,
#     provider="sambanova",
#     api_key=os.environ["SAMBANOVA_API_KEY"]
# )

basic_llm = None
for key in api_keys:
    try:
        basic_llm = LLM(
            model="sambanova/Meta-Llama-3.3-70B-Instruct",
            temperature=0,
            provider="sambanova",
            api_key=key
        )
        break
    except:
        continue

if basic_llm is None:
    raise Exception("All API keys failed to initialize the LLM.")
# Sutep Agents
## Agent-1
class QuestionAnalysis(BaseModel):
    intent: str = Field(..., title="User's main intent or topic")
    keywords: list[str] = Field(..., title="List of key terms extracted from the question")
    simplified_question: str = Field(..., title="Rephrased or clarified version of the question")

# Agent for analyzing the question
question_analyzer_agent = Agent(
    role="Egyptology question analyzer",
    goal="\n".join([
        "Understand and analyze user questions related to Ancient Egypt, including pharaohs, temples, artifacts, and events.",
        "Extract the main intent and keywords from the question.",
        "Rephrase the question clearly for a history expert agent to answer more effectively.",
        "Identify if the question is about a specific pharaoh, temple, artifact, or another topic in Ancient Egyptian history.",
        "Use advanced NLP techniques to extract the most relevant terms and context."
    ]),
    backstory="This agent analyzes user questions about Ancient Egypt, extracts important keywords, and simplifies the question to help the next agent answer more effectively. It ensures a clear understanding of the user's intent and provides better context.",
    llm=basic_llm,
    verbose=True
)

# Define the task for analyzing the user's question
question_analysis_task = Task(
    description="\n".join([
        "Analyze the user's question related to Ancient Egypt.",
        "Identify the user's intent, such as asking about a specific pharaoh, temple, artifact, or general historical aspect.",
        "Extract key keywords with high accuracy (e.g., names of pharaohs, gods, locations, events).",
        "Rephrase the question into a simplified form for the next agent to process more effectively.",
        "Ensure the keywords are accurate and context-specific."
    ]),
    expected_output="A JSON object with the intent of the question, the list of keywords, and a simplified version of the question.",
    output_json=QuestionAnalysis,
    output_file=os.path.join(output_dir, "analyzed_question.json"),
    output_key="simplified_question",
    agent=question_analyzer_agent
)

class SearchResult(BaseModel):
    source: str = Field(..., title="Source name or URL")
    content: str = Field(..., title="Detailed answer extracted from the source")

class SearchResultsList(BaseModel):
    results: list[SearchResult] = Field(..., title="List of detailed results from online sources")

search_agent = Agent(
    role="Ancient Egypt internet researcher",
    goal="\n".join([
        "To search the internet for detailed and accurate information about Ancient Egypt, including pharaohs, temples, and artifacts.",
        "Use trusted sources like Wikipedia, academic sites, and museum archives.",
        "Ensure results are specific to the question, considering if it is about a particular pharaoh, a temple, an artifact, or another element of Egyptian history.",
        "Return 5 distinct and detailed results from various sources, avoiding repetition and ensuring diversity."
    ]),
    backstory="This agent gathers accurate, diverse, and up-to-date information from reliable sources to support the final response. It provides essential context for pharaohs, temples, statues, and other significant topics.",
    llm=basic_llm,
    verbose=True
)

search_task = Task(
    description="\n".join([
        "Use the simplified question from the previous agent to search the internet.",
        "Search for information from trusted websites like Wikipedia, Egyptology databases, and museum archives.",
        "Return 5 distinct, detailed results that are informative and relevant to the specific topic of the query.",
        "Avoid repetitive information and ensure the results focus on the exact subject (e.g., Hatshepsut, Ramses II, etc.)."
    ]),
    expected_output="A JSON object containing a list of 5 detailed search results, each with content and its source.",
    output_json=SearchResultsList,
    output_file=os.path.join(output_dir, "search_results.json"),
    output_key="results",
    agent=search_agent
)


class EngagingResponse(BaseModel):
    response: str = Field(..., title="Final user-friendly response with follow-up question")

storyteller_agent = Agent(
    role="Storyteller & Conversation Manager",
    goal="\n".join([
        "Review multiple factual responses from different sources and synthesize them into one engaging, accurate, and natural-sounding answer.",
        "Ensure the final response is clear, friendly, and informative, as if narrated by a museum guide or a history podcast host.",
        "Always add a follow-up question that invites the user to explore a related aspect of Ancient Egyptian history."
    ]),
    backstory="This agent synthesizes diverse factual data and presents it in a storytelling format. It ensures accuracy and clarity while maintaining a conversational tone.",
    llm=basic_llm,
    verbose=True
)

storytelling_task = Task(
    description="\n".join([
        "Read the 5 detailed search results about the topic (e.g., Hatshepsut, Ramses II, etc.).",
        "Synthesize the most relevant and important information from these results.",
        "Write an engaging, informative response, ensuring that the tone is friendly and conversational.",
        "End the response with a follow-up question to keep the conversation going."
    ]),
    expected_output="A JSON object with the final answer and a follow-up question.",
    output_json=EngagingResponse,
    output_file=os.path.join(output_dir, "final_response.json"),
    output_key="response",
    agent=storyteller_agent
)

#main-chat
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_text(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except:
        return text


def run_ancient_egypt_chatbot():
    print("Welcome to the Ancient Egyptian Civilization Chatbot!")
    print("Type 'exit' to end the conversation.\n")

    last_input_time = datetime.now()
    timeout_duration = timedelta(minutes=30)

    while True:
        if datetime.now() - last_input_time > timeout_duration:
            print("Session timed out due to inactivity. Chat history saved. Goodbye!")
            break

        user_input = input("You: ")
        last_input_time = datetime.now()

        user_lang = detect_language(user_input)

        if user_input.strip().lower() in ["exit", "quit", "no"]:
            farewell_text = "Chat history saved. Goodbye!"
            translated_farewell = translate_text(farewell_text, "en", user_lang)
            with open(chat_history_file, "w", encoding="utf-8") as f:
                for log in session_log:
                    f.write(f"Time: {log['timestamp']}\n")
                    f.write(f"User Question: {log['user_question']}\n")
                    f.write(f"Simplified Question: {log['simplified_question']}\n")
                    f.write(f"Bot Answer: {log['final_answer']}\n")
                    f.write("\n=================================================\n")
            print(translated_farewell)
            break

        print("Analyzing your question...")
        analyzer_crew = Crew(
            agents=[question_analyzer_agent],
            tasks=[question_analysis_task]
        )
        analyzer_crew.kickoff(inputs={"input": user_input})

        with open(os.path.join(output_dir, "analyzed_question.json"), "r", encoding="utf-8") as f:
            simplified_question = json.load(f)["simplified_question"]

        print("Searching trusted websites...")
        search_crew = Crew(
            agents=[search_agent],
            tasks=[search_task]
        )
        search_crew.kickoff(inputs={"input": simplified_question})

        print("Crafting a user-friendly response...")
        storyteller_crew = Crew(
            agents=[storyteller_agent],
            tasks=[storytelling_task]
        )
        storyteller_crew.kickoff()

        with open(os.path.join(output_dir, "final_response.json"), "r", encoding="utf-8") as f:
            final_response = json.load(f)["response"]

        translated_response = translate_text(final_response, "en", user_lang)
        print(f"\nBot: {translated_response}\n")

        session_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_question": user_input,
            "simplified_question": simplified_question,
            "final_answer": translated_response
        })

        suggested_questions = [
            "Would you like to explore another pharaoh's life?",
            "Interested in learning about a famous Egyptian temple?",
            "Curious about daily life in Ancient Egypt?"
        ]

        print("Here are some questions you might find interesting:")
        for q in suggested_questions:
            translated_q = translate_text(q, "en", user_lang)
            print(f"- {translated_q}")

        print("\nFeel free to ask a new question or type 'no' to end the chat.\n")

# run_ancient_egypt_chatbot()
# Initialize FastAPI app
app = FastAPI()

# Request schema
class ChatRequest(BaseModel):
    question: str

# Session log
session_log = []

# Paths
output_dir = "outputs"
chat_history_file = "chat_history.txt"

# Supported 'no' translations
NO_WORDS = [
    "no", "exit", "quit", "non", "nein", "いいえ", "нет", "não", "nie", "لا", "नहीं", "όχι"
]

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Translation
def translate_text(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except:
        return text

# Generate dynamic suggested questions based on current question
def generate_suggestions(current_question):
    base = current_question.lower()
    if "pharaoh" in base:
        return [
            "Would you like to know about his achievements?",
            "Want to explore the dynasty he belonged to?",
            "Interested in his burial site or tomb?"
        ]
    elif "temple" in base:
        return [
            "Do you want to know when the temple was built?",
            "Curious about the gods worshipped there?",
            "Would you like to explore another temple?"
        ]
    else:
        return [
            "Would you like to learn about another historical topic?",
            "Interested in the culture of ancient Egyptians?",
            "Want to explore another fascinating mystery?"
        ]

# Main chat endpoint
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_input = req.question
    user_lang = detect_language(user_input)

    if user_input.strip().lower() in NO_WORDS:
        farewell = "Chat history saved. Goodbye!"
        translated_farewell = translate_text(farewell, "en", user_lang)

        with open(chat_history_file, "w", encoding="utf-8") as f:
            for log in session_log:
                f.write(f"Time: {log['timestamp']}\n")
                f.write(f"User Question: {log['user_question']}\n")
                f.write(f"Simplified Question: {log['simplified_question']}\n")
                f.write(f"Bot Answer: {log['final_answer']}\n")
                f.write("\n=================================================\n")

        return {
            "response": translated_farewell,
            "suggested_questions": []
        }

    # Step 1: Analyze question
    analyzer_crew = Crew(agents=[question_analyzer_agent], tasks=[question_analysis_task])
    analyzer_crew.kickoff(inputs={"input": user_input})

    with open(os.path.join(output_dir, "analyzed_question.json"), "r", encoding="utf-8") as f:
        simplified_question = json.load(f)["simplified_question"]

    # Step 2: Search
    search_crew = Crew(agents=[search_agent], tasks=[search_task])
    search_crew.kickoff(inputs={"input": simplified_question})

    # Step 3: Generate answer
    storyteller_crew = Crew(agents=[storyteller_agent], tasks=[storytelling_task])
    storyteller_crew.kickoff()

    with open(os.path.join(output_dir, "final_response.json"), "r", encoding="utf-8") as f:
        final_response = json.load(f)["response"]

    translated_response = translate_text(final_response, "en", user_lang)

    # Save session log
    session_log.append({
        "timestamp": datetime.now().isoformat(),
        "user_question": user_input,
        "simplified_question": simplified_question,
        "final_answer": translated_response
    })

    # Dynamic suggested questions
    suggestions = generate_suggestions(user_input)
    translated_suggestions = [translate_text(q, "en", user_lang) for q in suggestions]

    return {
        "response": translated_response,
        "suggested_questions": translated_suggestions
    }

# Run with: uvicorn main:app --reload --port 10000


