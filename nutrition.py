# nutrition_agent_optimized.py
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import time
import re
import random
from pathlib import Path

# Load environment
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FastNutritionAgent:
    def __init__(self):
        # Initialize clients
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "nutrition-agent")
        
        # Initialize embedding model for chat history
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        
        # Get Pinecone index
        self.pinecone_index = self.pc.Index(self.index_name)
        
        # Persistent memory file
        self.memory_file = Path("user_memory.json")
        
        # Memory for current session
        self.session_memory = []
        self.persistent_memory = self._load_persistent_memory()
        self.user_name = self.persistent_memory.get("user_name")
        
        # Load nutrition data once
        csv_path = "/Users/samsuljahith/nutrition_agent/FOOD-DATA-GROUP1.csv"
        self.nutrition_df = self._load_nutrition_data(csv_path)
        
        # Fast response patterns
        self.quick_responses = {
            r'hi|hello|hey': "Hello! I'm your nutrition assistant. What would you like to know?",
            r'bye|quit|exit': "Goodbye! Stay healthy!",
            r'thanks?|thank you': "You're welcome! Any other nutrition questions?",
            r'my name is (.+)': self._handle_name_introduction,
            r'what.* my name|do you know my name': self._recall_name,
        }
        
        print(f"Fast Nutrition Agent initialized and ready!")
        if self.user_name:
            print(f"Welcome back, {self.user_name}! 👋")

    def _load_persistent_memory(self) -> Dict:
        """Load persistent memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load memory: {e}")
        return {}

    def _save_persistent_memory(self):
        """Save persistent memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.persistent_memory, f, indent=2)
        except Exception as e:
            print(f"Could not save memory: {e}")

    def _save_chat_to_pinecone(self, user_msg: str, bot_response: str):
        """Save chat exchange to Pinecone for future retrieval"""
        try:
            # Create conversation text for embedding
            conversation_text = f"User: {user_msg}\nAssistant: {bot_response}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(conversation_text).tolist()
            
            # Create metadata
            metadata = {
                "user_message": user_msg,
                "assistant_response": bot_response,
                "timestamp": datetime.now().isoformat(),
                "user_name": self.user_name or "unknown",
                "conversation_type": "chat_history"
            }
            
            # Generate unique ID
            chat_id = f"chat_{uuid.uuid4()}"
            
            # Upsert to Pinecone
            self.pinecone_index.upsert(
                vectors=[{
                    "id": chat_id,
                    "values": embedding,
                    "metadata": metadata
                }],
                namespace="chat_history"
            )
            
        except Exception as e:
            print(f"Error saving to Pinecone: {e}")

    def _search_chat_history(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search relevant past conversations from Pinecone"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="chat_history",
                filter={"user_name": self.user_name} if self.user_name else None
            )
            
            # Extract relevant conversations
            past_conversations = []
            for match in results.matches:
                if match.score > 0.7:  # Only high similarity matches
                    past_conversations.append({
                        "user_msg": match.metadata["user_message"],
                        "assistant_msg": match.metadata["assistant_response"],
                        "timestamp": match.metadata["timestamp"],
                        "similarity": match.score
                    })
            
            return past_conversations
            
        except Exception as e:
            print(f"Error searching chat history: {e}")
            return []

    def _load_nutrition_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean nutrition data efficiently"""
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        df.fillna(0, inplace=True)
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col not in ['food', 'Unnamed: 0']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    def _handle_name_introduction(self, match):
        """Handle when user introduces themselves"""
        self.user_name = match.group(1).strip().title()
        # Save to persistent memory
        self.persistent_memory["user_name"] = self.user_name
        self.persistent_memory["first_met"] = datetime.now().isoformat()
        self._save_persistent_memory()
        return f"Nice to meet you, {self.user_name}! I'll remember your name for next time. How can I help with your nutrition questions?"

    def _recall_name(self, match):
        """Recall user's name"""
        if self.user_name:
            return f"Yes, your name is {self.user_name}!"
        return "You haven't told me your name yet. What should I call you?"

    def _check_quick_response(self, message: str) -> Optional[str]:
        """Check for quick responses first"""
        message_lower = message.lower().strip()
        
        for pattern, response in self.quick_responses.items():
            if match := re.search(pattern, message_lower):
                if callable(response):
                    return response(match)
                return response
        return None

    def _search_nutrition_data(self, query: str) -> str:
        """Fast nutrition data search using pandas"""
        query_lower = query.lower()
        
        # Handle specific nutrition queries
        if "protein" in query_lower:
            return self._handle_protein_query(query_lower)
        elif "low sodium" in query_lower:
            return self._handle_low_sodium_query(query_lower)
        elif "low carb" in query_lower:
            return self._handle_low_carb_query(query_lower)
        elif "high fiber" in query_lower:
            return self._handle_high_fiber_query(query_lower)
        elif "calories" in query_lower:
            return self._handle_calorie_query(query_lower)
        else:
            return self._general_food_search(query_lower)

    def _handle_protein_query(self, query: str) -> str:
        """Handle protein-related queries"""
        if any(food in query for food in ['chicken', 'beef', 'fish', 'salmon']):
            # Search for specific protein foods
            protein_foods = self.nutrition_df[
                self.nutrition_df['food'].str.contains('chicken|beef|fish|salmon', case=False, na=False)
            ].nlargest(5, 'Protein')
            
            if not protein_foods.empty:
                result = "Here are high-protein options:\n"
                for _, row in protein_foods.iterrows():
                    result += f"• {row['food']}: {row['Protein']}g protein\n"
                return result
        
        # General high protein foods
        high_protein = self.nutrition_df.nlargest(10, 'Protein')
        result = "Top high-protein foods:\n"
        for _, row in high_protein.iterrows():
            if row['Protein'] > 0:
                result += f"• {row['food']}: {row['Protein']}g protein\n"
        return result

    def _handle_low_sodium_query(self, query: str) -> str:
        """Handle low sodium queries"""
        if "soup" in query:
            # Find low sodium soups
            soups = self.nutrition_df[
                self.nutrition_df['food'].str.contains('soup', case=False, na=False)
            ].nsmallest(5, 'Sodium')
            
            if not soups.empty:
                result = "Low sodium soup options:\n"
                for _, row in soups.iterrows():
                    result += f"• {row['food']}: {row['Sodium']}mg sodium\n"
                return result
        
        # General low sodium foods
        low_sodium = self.nutrition_df.nsmallest(10, 'Sodium')
        result = "Low sodium food options:\n"
        for _, row in low_sodium.iterrows():
            result += f"• {row['food']}: {row['Sodium']}mg sodium\n"
        return result

    def _handle_low_carb_query(self, query: str) -> str:
        """Handle low carb queries"""
        low_carb = self.nutrition_df.nsmallest(10, 'Carbohydrates')
        result = "Low carb food options:\n"
        for _, row in low_carb.iterrows():
            result += f"• {row['food']}: {row['Carbohydrates']}g carbs\n"
        return result

    def _handle_high_fiber_query(self, query: str) -> str:
        """Handle high fiber queries"""
        high_fiber = self.nutrition_df.nlargest(10, 'Dietary Fiber')
        result = "High fiber food options:\n"
        for _, row in high_fiber.iterrows():
            if row['Dietary Fiber'] > 0:
                result += f"• {row['food']}: {row['Dietary Fiber']}g fiber\n"
        return result

    def _handle_calorie_query(self, query: str) -> str:
        """Handle calorie-related queries"""
        if "low calorie" in query:
            low_cal = self.nutrition_df.nsmallest(10, 'Caloric Value')
        else:
            low_cal = self.nutrition_df.nlargest(10, 'Caloric Value')
        
        result = "Calorie information:\n"
        for _, row in low_cal.iterrows():
            result += f"• {row['food']}: {row['Caloric Value']} calories\n"
        return result

    def _general_food_search(self, query: str) -> str:
        """General food search"""
        # Extract potential food names from query
        words = query.split()
        food_matches = []
        
        for word in words:
            if len(word) > 3:  # Skip short words
                matches = self.nutrition_df[
                    self.nutrition_df['food'].str.contains(word, case=False, na=False)
                ]
                if not matches.empty:
                    food_matches.extend(matches.head(3).to_dict('records'))
        
        if food_matches:
            result = "Nutrition information:\n"
            for food in food_matches[:5]:  # Limit to 5 results
                result += (f"• {food['food']}: {food['Caloric Value']} cal, "
                          f"{food['Protein']}g protein, {food['Carbohydrates']}g carbs, "
                          f"{food['Fat']}g fat\n")
            return result
        
        return "I couldn't find specific nutrition data for that. Could you be more specific about the food or nutrient you're asking about?"

    def _save_to_memory(self, user_msg: str, bot_response: str):
        """Save conversation to both session memory and Pinecone"""
        timestamp = datetime.now().isoformat()
        
        # Save to session memory
        self.session_memory.append({
            "timestamp": timestamp,
            "user": user_msg,
            "assistant": bot_response
        })
        
        # Save to Pinecone for persistent storage and retrieval
        self._save_chat_to_pinecone(user_msg, bot_response)
        
        # Save important interactions to local persistent memory too
        if any(keyword in user_msg.lower() for keyword in ['my name is', 'i am', 'call me']):
            self.persistent_memory["recent_conversations"] = self.persistent_memory.get("recent_conversations", [])
            self.persistent_memory["recent_conversations"].append({
                "timestamp": timestamp,
                "user": user_msg,
                "assistant": bot_response
            })
            # Keep only last 5 important conversations
            if len(self.persistent_memory["recent_conversations"]) > 5:
                self.persistent_memory["recent_conversations"] = self.persistent_memory["recent_conversations"][-5:]
            self._save_persistent_memory()
        
        # Keep only last 10 exchanges to avoid memory bloat
        if len(self.session_memory) > 10:
            self.session_memory = self.session_memory[-10:]

    def _get_llm_response(self, user_input: str) -> str:
        """Get response from Groq LLM with context from both session and Pinecone"""
        # Build context from recent session conversation
        session_context = ""
        if self.session_memory:
            session_context = "\nRecent session conversation:\n"
            for entry in self.session_memory[-3:]:  # Last 3 exchanges
                session_context += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n"
        
        # Search for relevant past conversations in Pinecone
        past_conversations = self._search_chat_history(user_input, top_k=2)
        history_context = ""
        if past_conversations:
            history_context = "\nRelevant past conversations:\n"
            for conv in past_conversations:
                history_context += f"Previous User: {conv['user_msg']}\nPrevious Assistant: {conv['assistant_msg']}\n"
        
        # Build complete prompt with all context
        prompt = f"""You are a helpful nutrition assistant. Answer questions about food, nutrition, calories, and healthy eating.

{session_context}

{history_context}

Current question: {user_input}

Provide a helpful, concise response about nutrition. If you recognize this question from past conversations, you can reference that context. If you don't have specific data, give general nutrition advice."""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Fast model
                temperature=0.7,
                max_tokens=300  # Keep responses concise
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "I'm having trouble connecting right now. Please try asking about specific foods or nutrients."

    def chat(self, user_input: str) -> str:
        """Main chat method - optimized for speed with Pinecone memory"""
        if not user_input.strip():
            return "Please ask me about nutrition, foods, or healthy eating!"
        
        # 1. Check for quick responses first (instant)
        quick_response = self._check_quick_response(user_input)
        if quick_response:
            self._save_to_memory(user_input, quick_response)
            return quick_response
        
        # 2. Try nutrition data search (fast pandas operations)
        nutrition_response = self._search_nutrition_data(user_input)
        if "couldn't find" not in nutrition_response:
            self._save_to_memory(user_input, nutrition_response)
            return nutrition_response
        
        # 3. Fall back to LLM with Pinecone context for complex queries
        llm_response = self._get_llm_response(user_input)
        self._save_to_memory(user_input, llm_response)
        return llm_response

    def run_interactive_chat(self):
        """Run interactive chat session"""
        print("\n🥗 Fast Nutrition Agent Ready!")
        print("Ask me about nutrition, foods, calories, or healthy eating.")
        print("Type 'quit' to exit.")
        if self.user_name:
            print(f"Hello again, {self.user_name}! 👋")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nNutritionist: Goodbye! Stay healthy! 🌟\n")
                    break
                
                # Get response (optimized for speed)
                start_time = time.time()
                response = self.chat(user_input)
                response_time = time.time() - start_time
                
                print(f"\nNutritionist: {response}")
                print(f"⚡ Response time: {response_time:.2f}s\n")
                
            except KeyboardInterrupt:
                print("\n\nNutritionist: Goodbye! Stay healthy! 🌟\n")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                continue

# Usage example
if __name__ == "__main__":
    agent = FastNutritionAgent()
    agent.run_interactive_chat()