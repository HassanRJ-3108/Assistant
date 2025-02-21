import os
import warnings
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import sqlite3

# --- Load environment variables ---
load_dotenv()

# Configure Gemini API Key
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
    st.success("API Key configured successfully! 😊")
else:
    st.error("No API key found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Optional: Suppress specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

try:
    from first.crew import PersonalAIAssistantCrew
except ImportError as e:
    if "pkg_resources" in str(e):
        st.error("Error importing pkg_resources. This might be due to a deployment issue. Please check your requirements.txt and ensure setuptools is installed.")
    else:
        st.error(f"Module not found: {e}. Ensure that 'crew.py' is inside the 'first' folder and __init__.py exists there. 🚫")
    st.stop()

def create_database():
    conn = sqlite3.connect('hassan_data.db')
    c = conn.cursor()
    c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS hassan_info 
                 USING fts5(content, tokenize='porter')''')
    conn.commit()
    conn.close()

def load_and_index_personal_data(filename="first/knowledge/user_preference.txt"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            personal_data = f.read().strip()
        
        # Create database and table if not exists
        create_database()
        
        # Insert data into the database
        conn = sqlite3.connect('hassan_data.db')
        c = conn.cursor()
        c.execute("DELETE FROM hassan_info")  # Clear existing data
        for line in personal_data.split('\n'):
            c.execute("INSERT INTO hassan_info (content) VALUES (?)", (line,))
        conn.commit()
        conn.close()
        
        st.success("Personal data loaded and indexed successfully! 😊")
        return personal_data
    except Exception as e:
        st.error(f"Error reading or indexing personal data: {e} 🚫")
        return None

def search_personal_data(query):
    conn = sqlite3.connect('hassan_data.db')
    c = conn.cursor()
    c.execute("SELECT content FROM hassan_info WHERE content MATCH ? ORDER BY rank LIMIT 5", (query,))
    results = c.fetchall()
    conn.close()
    return [result[0] for result in results]

def process_query(personal_data, user_query):
    if not personal_data:
        st.error("Personal data is empty. Please update the file.")
        return None
    
    try:
        # Search for relevant information
        relevant_info = search_personal_data(user_query)
        relevant_context = "\n".join(relevant_info)
        
        crew = PersonalAIAssistantCrew().crew()
        inputs = {
            "topic": "Hassan RJ",
            "current_year": str(datetime.now().year),
            "personal_data": relevant_context,
            "user_query": user_query
        }
        result = crew.kickoff(inputs=inputs)
        return result
    except Exception as e:
        st.error(f"An error occurred while processing your query: {str(e)} 🚫")
        return None

def train_model():
    st.write("### Training Parameters 🚀")
    topic = st.text_input("Enter the training topic (e.g., 'Hassan RJ')", key="train_topic")
    n_iterations = st.text_input("Enter the number of training iterations (e.g., 5)", key="train_iterations")
    filename = st.text_input(
        "Enter the filename for training input (e.g., first/knowledge/user_preference.txt)",
        value="first/knowledge/user_preference.txt",
        key="train_filename"
    )
    
    if st.button("Start Training 🚀"):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                personal_data = f.read().strip()
        except Exception as e:
            st.error(f"Error reading training file: {e} 🚫")
            return
        
        if not personal_data:
            st.error("The training file is empty. Please add your personal data to the file. 🚫")
            return
        
        try:
            crew = PersonalAIAssistantCrew().crew()
            crew.train(
                n_iterations=int(n_iterations),
                filename=filename,
                inputs={"topic": topic, "personal_data": personal_data}
            )
            st.success("Training completed successfully! 😊")
        except Exception as e:
            st.error(f"An error occurred while training the crew: {e} 🚫")

def main():
    st.title("Personal AI Assistant for Hassan RJ 🚀😊")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Ask Query", "Train Model"])
    
    personal_data = load_and_index_personal_data()
    
    if app_mode == "Ask Query":
        st.header("Ask a Question about Hassan RJ")
        user_query = st.text_area("Enter your question:", height=100)
        if st.button("Submit Query 🚀"):
            if user_query.strip() == "":
                st.error("Please enter a question.")
            else:
                with st.spinner("Processing query..."):
                    result = process_query(personal_data, user_query)
                if result is not None:
                    st.subheader("AI Assistant's Response:")
                    try:
                        parsed = json.loads(result)
                        if "raw" in parsed:
                            st.markdown(parsed["raw"])
                        else:
                            st.markdown(result)
                    except Exception:
                        st.markdown(result)
    
    elif app_mode == "Train Model":
        st.header("Train the Personal AI Assistant")
        train_model()

if __name__ == "__main__":
    main()