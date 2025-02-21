import os
import warnings
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()  # Loads .env from the current directory

# Check for Gemini API Key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("No API key found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()
else:
    # Set it directly as GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = gemini_key
    st.success("Gemini API Key configured successfully! 😊")

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

# --- Function to load personal data ---
def load_personal_data(filename="first/knowledge/user_preference.txt"):
    """
    Load personal data from the specified file.
    The file is now located in the 'first/knowledge' folder relative to app.py.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            personal_data = f.read().strip()
        st.success("Personal data loaded successfully! 😊")
        st.write(f"Length of personal data: {len(personal_data)} characters")
        return personal_data
    except Exception as e:
        st.error(f"Error reading personal data from file: {e} 🚫")
        return None

# --- Function to process user queries ---
def process_query(personal_data, user_query):
    """
    Process a user query using the PersonalAIAssistantCrew.
    """
    if not personal_data:
        st.error("Personal data is empty. Please update the file.")
        return None
    
    # Initialize the crew and prepare inputs
    try:
        crew = PersonalAIAssistantCrew().crew()
        inputs = {
            "topic": "Hassan RJ",
            "current_year": str(datetime.now().year),
            "personal_data": personal_data,
            "user_query": user_query
        }
        result = crew.kickoff(inputs=inputs)
        return result
    except Exception as e:
        st.error(f"An error occurred while processing your query: {str(e)} 🚫")
        return None

# --- Function to train the model ---
def train_model():
    """
    Provide UI inputs for training and train the crew.
    """
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

# --- Main Streamlit application ---
def main():
    st.title("Personal AI Assistant for Hassan RJ 🚀😊")
    
    # Sidebar navigation for switching between modes
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Ask Query", "Train Model"])
    
    # Load personal data once
    personal_data = load_personal_data()
    
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