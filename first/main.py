import warnings
from datetime import datetime
import streamlit as st
from first.crew import PersonalAIAssistantCrew

# Suppress specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def load_personal_data(filename="knowledge/user_preference.txt"):
    """
    Load personal data from a file.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            personal_data = f.read().strip()
        st.success("Personal data loaded successfully! 😊")
        st.write(f"Length of personal data: {len(personal_data)} characters")
        return personal_data
    except Exception as e:
        st.error(f"Error reading personal data from file: {e} 🚫")
        return None

def process_query(personal_data, user_query):
    """
    Process a user query using the PersonalAIAssistantCrew.
    """
    if not personal_data:
        st.error("Personal data is empty. Please update the file.")
        return None

    crew = PersonalAIAssistantCrew().crew()
    inputs = {
        "topic": "Hassan RJ",
        "current_year": str(datetime.now().year),
        "personal_data": personal_data,
        "user_query": user_query
    }
    try:
        result = crew.kickoff(inputs=inputs)
        return result
    except Exception as e:
        st.error(f"An error occurred while processing your query: {e} 🚫")
        return None

def train_model():
    """
    Provide UI inputs for training and train the crew.
    """
    st.write("### Training Parameters 🚀")
    topic = st.text_input("Enter the training topic (e.g., 'Hassan RJ')", key="train_topic")
    n_iterations = st.text_input("Enter the number of training iterations (e.g., 5)", key="train_iterations")
    filename = st.text_input("Enter the filename for training input (e.g., knowledge/user_preference.txt)", value="knowledge/user_preference.txt", key="train_filename")
    
    if st.button("Start Training 🚀"):
        # Read training file
        try:
            with open(filename, "r", encoding="utf-8") as f:
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
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Ask Query", "Train Model"])
    
    # Load personal data once at the start
    personal_data = load_personal_data()
    
    if app_mode == "Ask Query":
        st.header("Ask a Question about Hassan RJ")
        user_query = st.text_area("Enter your question:", height=100)
        if st.button("Submit Query 🚀"):
            if user_query.strip() == "":
                st.error("Please enter a question.")
            else:
                result = process_query(personal_data, user_query)
                if result is not None:
                    st.subheader("AI Assistant's Response:")
                    st.write(result)
                    
    elif app_mode == "Train Model":
        st.header("Train the Personal AI Assistant")
        train_model()

if __name__ == "__main__":
    main()
