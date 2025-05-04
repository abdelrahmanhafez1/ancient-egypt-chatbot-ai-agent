#libraries
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from crewai import Agent, Task, Crew, Process ,LLM
from firecrawl import FirecrawlApp
from langchain.tools import tool
from datetime import datetime
from langdetect import detect
from deep_translator import GoogleTranslator
from datetime import datetime
import json
import os
from dotenv import load_dotenv
#LLM-API
# LLM-API
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
chat_history_file = "outputs/chat_history.txt"
session_log = []
load_dotenv()

basic_llm = LLM(
    model="sambanova/Meta-Llama-3.3-70B-Instruct",
    temperature=0,
    provider="sambanova",
    api_key=os.environ["SAMBANOVA_API_KEY"]
)

# Sutep Agents
## Agent-1
class QuestionAnalysis(BaseModel):
    intent: str = Field(..., title="User's main intent or topic")
    keywords: list[str] = Field(..., title="List of key terms extracted from the question")
    simplified_question: str = Field(..., title="Rephrased or clarified version of the question")

question_analyzer_agent = Agent(
    role="Egyptology question analyzer",
    goal="\n".join([
        "Understand and analyze user questions related to Ancient Egypt, including pharaohs, temples, artifacts, and events.",
        "Extract the main intent and keywords from the question.",
        "Rephrase the question clearly for a history expert agent to answer more effectively."
    ]),
    backstory="This agent is designed to act as the first step in the pipeline. It analyzes user questions about Ancient Egypt, extracts important keywords, and simplifies the question to help other agents answer more effectively.",
    llm=basic_llm,
    verbose=True
)

question_analysis_task = Task(
    description="\n".join([
        "Analyze the user's question about Ancient Egypt.",
        "Identify the user's intent (e.g., asking about a pharaoh, a temple, a battle, or daily life).",
        "Extract key keywords from the question.",
        "Rephrase the question into a simpler and clearer form for another agent to answer.",
    ]),
    expected_output="A JSON object with intent, list of keywords, and a simplified question.",
    output_json=QuestionAnalysis,
    output_file=os.path.join(output_dir, "analyzed_question.json"),
    output_key="simplified_question",
    agent=question_analyzer_agent
)

# Agent-2
class SearchResult(BaseModel):
    source: str = Field(..., title="Source name or URL")
    content: str = Field(..., title="Detailed answer extracted from the source")

class SearchResultsList(BaseModel):
    results: list[SearchResult] = Field(..., title="List of detailed results from online sources")

search_agent = Agent(
    role="Ancient Egypt internet researcher",
    goal="\n".join([
        "To search the internet for detailed and accurate information about temples, statues, pharaohs, and symbolic meanings in Ancient Egypt.",
        "Use trusted sources such as Wikipedia, academic sites, and museum archives.",
        "Return 5 detailed and diverse results from different sources."
    ]),
    backstory="This agent is responsible for gathering real-time, accurate, and diverse information from trusted sources across the internet. It supports the symbolic agent by providing raw knowledge to work with.",
    llm=basic_llm,
    verbose=True
)

search_task = Task(
    description="\n".join([
        "Use the simplified question from the previous agent to search the internet.",
        "Search for information on websites such as Wikipedia, Egyptology databases, academic articles, and museum archives.",
        "Return 5 distinct, detailed explanations or insights related to the query, each labeled with its source.",
        "Ensure the content is informative and relevant to temples, statues, or pharaonic symbolism."
    ]),
    expected_output="A JSON object containing a list of 5 detailed search results, each with content and its source.",
    output_json=SearchResultsList,
    output_file=os.path.join(output_dir, "search_results.json"),
    output_key="results",
    agent=search_agent
)

# Agent-3
class EngagingResponse(BaseModel):
    response: str = Field(..., title="Final user-friendly response with follow-up question")

storyteller_agent = Agent(
    role="Storyteller & Conversation Manager",
    goal="\n".join([
        "To review multiple factual responses and synthesize them into one engaging, accurate, and natural-sounding answer.",
        "Make sure the final message sounds friendly, informative, and suitable for casual conversation.",
        "Always add a follow-up question that invites the user to explore a related aspect."
    ]),
    backstory="This agent acts like a knowledgeable museum guide who knows how to summarize research and communicate it in a smooth and engaging way. It chooses the best bits from multiple detailed sources and turns them into a compelling answer.",
    llm=basic_llm,
    verbose=True
)

storytelling_task = Task(
    description="\n".join([
        "Read the 5 detailed search results about a pharaonic statue, temple, or symbolic topic.",
        "Synthesize the most important and relevant information from those answers.",
        "Write a single clear, engaging response that includes the key details in 3 to 6 sentences.",
        "Ensure the tone is warm, storytelling-style, like a museum guide or history podcast.",
        "End the response with a thoughtful follow-up question to keep the user engaged."
    ]),
    expected_output="A JSON object with the final answer and a follow-up question.",
    output_json=EngagingResponse,
    output_file=os.path.join(output_dir, "final_response.json"),
    output_key="response",
    agent=storyteller_agent
)

# play
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

    while True:
        user_input = input("You: ")

        user_lang = detect_language(user_input)

        if user_input.strip().lower() in ["exit", "quit", "خروج", "no", "لا"]:
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

        next_question_en = "Would you like to learn about another temple, pharaoh, or Egyptian custom?"
        translated_suggestion = translate_text(next_question_en, "en", user_lang)
        print(f"\n{translated_suggestion}\n")

        follow_up_question = "Would you like to ask another question? (yes/no)"
        translated_follow_up = translate_text(follow_up_question, "en", user_lang)
        follow_input = input(f"{translated_follow_up}\nYou: ").strip().lower()

        if follow_input in ["no", "لا", "exit", "quit"]:
            farewell_text = "Chat history saved. Goodbye!"
            translated_farewell = translate_text(farewell_text, "en", user_lang)
            print(translated_farewell)
            with open(chat_history_file, "w", encoding="utf-8") as f:
                for log in session_log:
                    f.write(f"Time: {log['timestamp']}\n")
                    f.write(f"User Question: {log['user_question']}\n")
                    f.write(f"Simplified Question: {log['simplified_question']}\n")
                    f.write(f"Bot Answer: {log['final_answer']}\n")
                    f.write("\n=================================================\n")
            break

run_ancient_egypt_chatbot()

