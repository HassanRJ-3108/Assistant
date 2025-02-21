import os
import warnings
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
import re

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


def tokenize(text):
    return re.findall(r'\w+', text.lower())

def vectorize(text):
    return Counter(tokenize(text))

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = (sum1 * sum2)**0.5
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def load_and_vectorize_personal_data(filename="first/knowledge/user_preference.txt"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            personal_data = f.read().strip()
        
        # Vectorize the personal data
        sentences = personal_data.split('\n')
        vectorized_sentences = [vectorize(sentence) for sentence in sentences]
        
        st.success("Personal data loaded and vectorized successfully! 😊")
        st.write(f"Number of sentences vectorized: {len(sentences)}")
        return personal_data, vectorized_sentences, sentences
    except Exception as e:
        st.error(f"Error reading or vectorizing personal data: {e} 🚫")
        return None, None, None

def process_query(personal_data, user_query, vectorized_sentences, sentences):
    if not personal_data or not vectorized_sentences:
        st.error("Personal data or vectorization components are missing.")
        return None
    
    try:
        # Vectorize the user query
        query_vector = vectorize(user_query)
        
        # Perform similarity search
        similarities = [cosine_similarity(query_vector, sent_vector) for sent_vector in vectorized_sentences]
        
        # Get top 5 most similar sentences
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
        relevant_sentences = [sentences[i] for i in top_indices]
        relevant_context = "\n".join(relevant_sentences)
        
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
    
    personal_data, vectorized_sentences, sentences = load_and_vectorize_personal_data()
    
    if app_mode == "Ask Query":
        st.header("Ask a Question about Hassan RJ")
        user_query = st.text_area("Enter your question:", height=100)
        if st.button("Submit Query 🚀"):
            if user_query.strip() == "":
                st.error("Please enter a question.")
            else:
                with st.spinner("Processing query..."):
                    result = process_query(personal_data, user_query, vectorized_sentences, sentences)
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