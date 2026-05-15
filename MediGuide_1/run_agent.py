"""Build Agent using Microsoft Agent Framework in Python
# Run this python script
> pip install agent-framework==1.0.0rc6
> python <this-script-path>.py
"""

import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from agent_framework_foundry import FoundryAgent
from azure.identity.aio import DefaultAzureCredential
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Vector database path
VECTOR_DB_DIR = "../chroma_db"

class ClinicalSearchTool:
    """Tool for searching clinical Q&A database"""

    def __init__(self):
        self.db = None
        self._load_database()

    def _load_database(self):
        """Load the Chroma vector database"""
        if os.path.isdir(VECTOR_DB_DIR):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        else:
            print(f"Warning: Vector database not found at {VECTOR_DB_DIR}")

    def search_clinical_cases(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant clinical cases based on symptoms or questions"""
        if not self.db:
            return [{"error": "Vector database not available"}]

        try:
            docs = self.db.similarity_search(query, k=k)
            results = []
            for doc in docs:
                # Parse the document content
                content = doc.page_content
                if "Question:" in content and "Answer:" in content:
                    question_part = content.split("Question:")[1].split("Answer:")[0].strip()
                    answer_part = content.split("Answer:")[1].strip()
                else:
                    question_part = content
                    answer_part = "No answer available"

                results.append({
                    "question": question_part,
                    "answer": answer_part,
                    "source": doc.metadata.get("source", "Unknown")
                })
            return results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

# Initialize the search tool
clinical_tool = ClinicalSearchTool()

async def main() -> None:
    # For authentication, DefaultAzureCredential supports multiple authentication methods. Run `az login` in terminal for Azure CLI auth.
    async with FoundryAgent(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        agent_name="MediGuide",
        agent_version="1",
        credential=DefaultAzureCredential(),
    ) as agent:

        print("🤖 MediGuide Clinical Assistant is ready!")
        print("Describe your symptoms or ask a medical question. Type 'quit' to exit.\n")

        while True:
            # Get user input
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! Stay healthy! 👋")
                break

            if not user_input:
                continue

            # Search for relevant clinical cases
            print("🔍 Searching clinical database...")
            search_results = clinical_tool.search_clinical_cases(user_input, k=3)

            # Format the retrieved information for the agent
            context = "Based on the clinical database, here are relevant cases:\n\n"
            for i, result in enumerate(search_results, 1):
                if "error" in result:
                    context += f"Error: {result['error']}\n"
                else:
                    context += f"Case {i}:\n"
                    context += f"Question: {result['question']}\n"
                    context += f"Answer: {result['answer']}\n"
                    context += f"Source: {result['source']}\n\n"

            # Combine user input with clinical context
            enhanced_prompt = f"""User question: {user_input}

{context}

Please provide a helpful, conversational response as a medical assistant. Use the clinical information above to inform your answer, but respond naturally like a doctor would. Include relevant medical advice, but always remind the user to consult a physician for proper diagnosis and treatment."""

            print("🧠 Thinking...")

            # Process with the agent
            print(f"\n🤖 MediGuide:", end=" ", flush=True)
            printed_tool_calls = set()
            async for chunk in agent.run(enhanced_prompt, stream=True):
                # log tool calls if any
                function_calls = [
                    c for c in chunk.contents
                    if c.type == "function_call"
                ]
                for call in function_calls:
                    if call.call_id not in printed_tool_calls:
                        print(f"Tool calls: {call.name}")
                        printed_tool_calls.add(call.call_id)
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished.")
