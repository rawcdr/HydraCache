import os
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Configure Langchain models using Groq via OpenAI proxy
    api_key = os.getenv("GROQ_API_KEY")
    chat_model = ChatGroq(
        api_key=api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0
    )
    
    # Use our existing embeddings model
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Dummy data
    data = {
        "question": ["What is Apple's 2025 net sales?"],
        "answer": ["Apple's 2025 net sales were $416,161 million."],
        "contexts": [["Apple reported total net sales of $416,161 million in 2025 across all segments."]],
        "ground_truth": ["$416,161 million"]
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running ragas evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=chat_model,
        embeddings=embeddings
    )
    
    print(result)

if __name__ == "__main__":
    main()
