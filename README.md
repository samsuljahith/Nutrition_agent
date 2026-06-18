# Nutrition Agent — RAG-Powered Food & Diet Assistant

A conversational nutrition assistant with persistent memory across sessions. Uses a USDA food nutrition CSV as its knowledge base, Pinecone for semantic chat history retrieval, and Groq (LLaMA 3.1) for natural language responses.

## Features

- **Nutrition queries** — protein, calories, sodium, carbs, fibre lookups from real food data
- **Fast pandas path** — common queries (high-protein foods, low-sodium soups) hit pandas directly without an LLM call
- **Persistent memory** — remembers your name and past conversations across sessions via Pinecone + local JSON
- **LLM fallback** — complex queries that don't match structured patterns go to Groq with conversation context injected

## How It Works

```
User query
  ├─ Quick response patterns (regex, instant)
  ├─ Pandas nutrition search (fast, structured)
  └─ Groq LLM with Pinecone context injection (complex queries)
```

Chat history is embedded and stored in Pinecone; relevant past exchanges are retrieved and injected into the LLM prompt on each turn.

## Setup

```bash
pip install groq pinecone sentence-transformers pandas python-dotenv
```

Create a `.env` file:

```env
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=nutrition-agent
```

Place `FOOD-DATA-GROUP1.csv` in the project root (or update the path in `nutrition.py`).

```bash
python nutrition.py
```

## Tech Stack

- **LLM** — Groq `llama-3.1-8b-instant`
- **Vector memory** — Pinecone (namespace: `chat_history`)
- **Embeddings** — SentenceTransformers `all-MiniLM-L6-v2`
- **Data** — pandas over USDA nutrition CSV

## Example Prompts

```
What are the highest protein foods?
Low sodium soup options?
My name is Alex
What low carb foods can I eat?
```
