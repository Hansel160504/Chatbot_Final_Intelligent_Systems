import streamlit as st
import pandas as pd
import json
import requests
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
from difflib import get_close_matches

# ---------------------------
# 1️⃣ CONFIG & GEMINI SETUP
# ---------------------------
st.set_page_config(page_title="AI Movie Assistant", layout="centered")

# Load environment variables
load_dotenv()

# Try to get API Key from Streamlit Secrets or Environment Variable
api_key = os.getenv("GEMINI_API_KEY") 

# Configure Gemini if key exists
GEMINI_AVAILABLE = False
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash") # Flash is faster/cheaper for large data
        GEMINI_AVAILABLE = True
    except Exception as e:
        st.error(f"Gemini Configuration Error: {e}")

OMDB_API_KEY = "ee2cec22"

# ---------------------------
# 2️⃣ DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tmdb_5000_movies.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    def parse_genres(x):
        try: return [item['name'].lower() for item in json.loads(x)]
        except: return []
            
    df['parsed_genres'] = df['genres'].apply(parse_genres)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    
    # Clean numeric columns
    cols = ['budget', 'revenue', 'vote_average', 'popularity', 'runtime']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['title'] = df['title'].astype(str)
    
    # Clean text columns
    df['overview'] = df['overview'].fillna("No description available.")
    df['tagline'] = df['tagline'].fillna("")
    return df

df = load_data()

# Prepare simple lists for Regex search
if not df.empty:
    movie_titles = df["title"].tolist()
    all_genres = set(g for sublist in df['parsed_genres'] for g in sublist)
else:
    movie_titles = []
    all_genres = set()

# ---------------------------
# 3️⃣ HELPER FUNCTIONS
# ---------------------------
def get_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        if "Poster" in data and data["Poster"] != "N/A":
            return data["Poster"]
    except:
        pass
    return None

def get_json_names(json_str):
    try: return ", ".join([item['name'] for item in json.loads(json_str)])
    except: return "N/A"

def ask_gemini(question, dataframe):
    if not GEMINI_AVAILABLE:
        return "⚠️ Gemini API Key not found."

    # 1. OPTIMIZATION: Filter data based on the question BEFORE sending to Gemini
    target_df = dataframe.copy()
    
    # Simple keyword filtering to make the context smarter
    if "horror" in question.lower():
        mask = target_df['parsed_genres'].apply(lambda x: "horror" in x)
        target_df = target_df[mask]
    elif "action" in question.lower():
        mask = target_df['parsed_genres'].apply(lambda x: "action" in x)
        target_df = target_df[mask]
    elif "comedy" in question.lower():
        mask = target_df['parsed_genres'].apply(lambda x: "comedy" in x)
        target_df = target_df[mask]

    # 2. SORTING FIX:
    # We must include 'popularity' in the list so we can sort by it.
    cols_to_keep = ['title', 'year', 'budget', 'revenue', 'vote_average', 'popularity']
    
    # Sort first, then take the top 100
    slim_df = target_df[cols_to_keep].sort_values(by='popularity', ascending=False).head(100)
    
    # Convert to text
    data_context = slim_df.to_string(index=False)
    
    prompt = f"""
    You are an expert movie data analyst. 
    Use the dataset sample below to answer the question.
    
    DATA SAMPLE:
    {data_context}
    
    QUESTION: {question}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error asking Gemini: {e}"
# ---------------------------
# 4️⃣ SEARCH LOGIC ("The Hybrid Brain")
# ---------------------------
def natural_language_search(query):
    query_lower = query.lower()
    
    # --- 1. SPECIAL LOGIC: COMPLEX QUESTIONS -> GEMINI ---
    # If the user types a long sentence (> 3 words) and it doesn't contain a year, 
    # assume it's a question for AI, not a movie title search.
    if len(query.split()) > 3 and not re.search(r'\b(19|20)\d{2}\b', query_lower):
        return pd.DataFrame(), "gemini"

    # --- 2. FILTERS (Year & Genre) ---
    year_match = re.search(r'\b(19|20)\d{2}\b', query_lower)
    year_filter = int(year_match.group()) if year_match else None
    genre_filters = [g for g in all_genres if g in query_lower]

    # --- 3. SORTING KEYWORDS ---
    sort_map = {
        'budget': ['budget', 'expensive', 'cost'],
        'revenue': ['revenue', 'gross', 'box office'],
        'vote_average': ['best', 'top', 'rating', 'worst'],
        'popularity': ['popular', 'trending']
    }
    
    target_col = 'popularity'
    ascending = False
    
    for col, keywords in sort_map.items():
        if any(k in query_lower for k in keywords):
            target_col = col
            break
            
    if any(k in query_lower for k in ['worst', 'lowest', 'least']):
        ascending = True

    # --- 4. EXECUTE LOCAL FILTER/SORT ---
    if year_filter or genre_filters or any(k in query_lower for k in ['best', 'top', 'budget', 'popular']):
        filtered = df.copy()
        if year_filter: filtered = filtered[filtered['year'] == year_filter]
        if genre_filters: 
            mask = filtered['parsed_genres'].apply(lambda x: any(g in x for g in genre_filters))
            filtered = filtered[mask]
        
        filtered = filtered.sort_values(by=target_col, ascending=ascending)
        return filtered.head(10), "list"

    # --- 5. FALLBACK TITLE SEARCH (Local) ---
    # increased cutoff to 0.6 to avoid "Sharkboy" errors
    matches = get_close_matches(query, movie_titles, n=1, cutoff=0.6) 
    if matches:
        return df[df['title'] == matches[0]], "title"
    
    # --- 6. FINAL FALLBACK -> GEMINI ---
    return pd.DataFrame(), "gemini"

# ---------------------------
# 5️⃣ CHAT INTERFACE
# ---------------------------
st.title("🤖 AI Movie Assistant")
if not GEMINI_AVAILABLE:
    st.warning("⚠️ Gemini is not active. Add GEMINI_API_KEY to .env to enable AI answers.")

st.caption("Ask specific queries (e.g. 'Action 2015') for Lists, or general questions (e.g. 'Compare Avatar and Titanic') for AI.")

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can search the database to answer questions."}]

# --- RENDER HISTORY ---
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        
        # TYPE A: LIST (UI Buttons)
        if isinstance(msg["content"], list) and msg.get("type") == "movie_list":
            st.write(f"Found {len(msg['content'])} movies:")
            for idx, item in enumerate(msg["content"]):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**{idx+1}. {item['title']}** ({int(item['year'])})")
                    st.caption(f"Rating: {item['vote_average']}/10")
                with c2:
                    btn_key = f"btn_{msg_index}_{item['id']}"
                    if st.button("View Details", key=btn_key):
                        st.session_state.messages.append({"role": "user", "content": f"Show details for {item['title']}"})
                        st.session_state.messages.append({"role": "assistant", "type": "movie_card", "content": {"data": item}})
                        st.rerun()

        # TYPE B: MOVIE CARD (Detailed UI)
        elif isinstance(msg["content"], dict) and msg.get("type") == "movie_card":
            m = msg["content"]["data"]
            st.subheader(f"{m['title']}")
            if m['tagline']: st.caption(f"_{m['tagline']}_")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                poster = get_poster(m['title'])
                if poster: st.image(poster, use_container_width=True)
            with c2:
                st.write(f"📅 **Year:** {int(m['year'])} | ⭐ **Rating:** {m['vote_average']}/10")
                st.write(f"⏱️ **Runtime:** {int(m['runtime'])} min")
                st.write(f"🗣️ **Language:** {str(m['original_language']).upper()}")
                
            st.markdown("### 📝 Overview")
            st.write(m['overview'])
            with st.expander("💰 Financials & Production"):
                st.write(f"**Budget:** ${m['budget']:,.0f}")
                st.write(f"**Revenue:** ${m['revenue']:,.0f}")
                st.write(f"**Prod:** {get_json_names(m['production_companies'])}")
                
        # TYPE C: TEXT (Standard or Gemini)
        else:
            st.write(msg["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Ask about movies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # 1. Try Local Search First (Fast + UI Buttons)
        results, search_type = natural_language_search(prompt)
        
        if search_type == "title":
            movie = results.iloc[0].to_dict()
            st.write("Here are the details:")
            st.session_state.messages.append({"role": "assistant", "type": "movie_card", "content": {"data": movie}})
            st.rerun() 

        elif search_type == "list":
            st.write(f"Found {len(results)} movies:")
            movies_data = results.to_dict('records')
            st.session_state.messages.append({"role": "assistant", "type": "movie_list", "content": movies_data})
            st.rerun()
            
        elif search_type == "gemini":
            # 2. Fallback to Gemini (Smart Text Answer)
            with st.spinner("🤖 Asking Gemini..."):
                ai_response = ask_gemini(prompt, df)
                st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})