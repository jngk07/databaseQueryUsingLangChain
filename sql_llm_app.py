# sql_llm_app.py

import os
import re
import ast
import streamlit as st

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from pydantic import BaseModel, Field
from typing import List
from operator import itemgetter


# Set API Keys securely
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else st.text_input("🔑 Google API Key", type="password")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔑 OpenAI API Key", type="password")

st.title("🧠 Natural Language SQL Querying")
st.caption("Ask questions like 'Show me the total sales by artist'")

# Connect to SQLite database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
st.success("✅ DB Connected: " + ", ".join(db.get_usable_table_names()))

# Pydantic Tool
class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")

# Initialize Gemini LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Category → Tables Chain
categories_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return the names of any SQL tables that are relevant to the user question.\nThe tables are:\n\nMusic\nBusiness\n"),
    ("human", "{input}")
])

llm_with_tools = llm.bind_tools([Table])
category_parser = PydanticToolsParser(tools=[Table])
category_chain = categories_prompt | llm_with_tools | category_parser

def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(["Album", "Artist", "Genre", "MediaType", "Playlist", "PlaylistTrack", "Track"])
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables

table_chain = category_chain | get_tables

# Gather proper nouns for vector search
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(str(res)) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res

proper_nouns = query_as_list(db, "SELECT Name FROM Artist")
proper_nouns += query_as_list(db, "SELECT Title FROM Album")
proper_nouns += query_as_list(db, "SELECT Name FROM Genre")

retriever = FAISS.from_texts(proper_nouns, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 15})

# Prompt Template
sql_prompt = ChatPromptTemplate.from_template(
    """You are a SQLite expert. Your job is to generate a syntactically correct and executable SQLite query.

### Objective:
Generate a SQL query that answers the user's natural language question using the Chinook database.

### Instructions:
- Only return the SQL query with **no markdown**, no explanation.
- Return **no more than {top_k} rows**, unless explicitly told otherwise.
- Do NOT guess any column or table relationships. Use only the valid joins listed below.
- If filtering on a value (e.g., artist, album, genre), validate its spelling against the values listed in the table_info.

### Valid Tables and Columns:

Artist(ArtistId, Name)  
Album(AlbumId, Title, ArtistId)  
Track(TrackId, Name, AlbumId, GenreId)  
Genre(GenreId, Name)  
Playlist(PlaylistId, Name)  
PlaylistTrack(PlaylistId, TrackId)  
MediaType(MediaTypeId, Name)

### Valid Join Paths:

- Album.ArtistId = Artist.ArtistId  
- Track.AlbumId = Album.AlbumId  
- Track.GenreId = Genre.GenreId  
- PlaylistTrack.TrackId = Track.TrackId  
- PlaylistTrack.PlaylistId = Playlist.PlaylistId  

### Table Info:
{table_info}

### Question:
{input}
"""
)

query_chain = create_sql_query_chain(llm, db, prompt=sql_prompt)

retriever_chain = (
    itemgetter("question")
    | retriever
    | (lambda docs: "\n".join(doc.page_content for doc in docs))
)

final_chain = (
    RunnablePassthrough.assign(
        proper_nouns=retriever_chain,
        table_names_to_use={"input": itemgetter("question")} | table_chain
    )
    | query_chain
)

def clean_sql(sql_text):
    cleaned = re.sub(r"```sql|```", "", sql_text, flags=re.IGNORECASE).strip()
    cleaned = re.split(r"(?i)\b(SELECT|WITH)\b", cleaned, maxsplit=1)
    if len(cleaned) == 3:
        cleaned = cleaned[1] + cleaned[2]
    else:
        cleaned = cleaned[0]
    return cleaned.strip()

# Question UI
question = st.text_input("❓ Ask a question about the database")

if question:
    with st.spinner("Generating SQL..."):
        sql = final_chain.invoke({"question": question, "input": question, "top_k": 5})
        cleaned_sql = clean_sql(sql)
        st.code(cleaned_sql, language="sql")

        try:
            result = db.run(cleaned_sql)
            st.dataframe(result)
        except Exception as e:
            st.error(f"Execution Error: {e}")
