from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# =========================================TOOLS=======================================================

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    print("Tool Message: Addition Tool is Called!")
    print("=" * 40)
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    print("Tool Message: Subtraction Tool is Called!")
    print("=" * 40)
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    print("Tool Message: Multiplication Tool is Called!")
    print("=" * 40)
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two integers."""
    print("Tool Message: Division Tool is Called!")
    print("=" * 40)
    if b == 0:
        raise ValueError("Error: Division by zero is not allowed.")
    return a / b

@tool
def intro(input_str: str = "") -> str:
    """Provide Hasnain's introduction."""
    print("Tool Message: Introduction Tool is Called!")
    print("=" * 40)
    return (
        """Hasnain Ali is a skilled web developer and programmer with a passion for creating dynamic, user-focused projects.
        Here is His LinkedIn Profile: https://www.linkedin.com/in/hasnain-ali-developer/
        His portfolio features innovative Python projects and AI solutions, including a chatbot built with LangChain and Google Gemini LLM.
        He is exploring Agentic AI and intelligent agents, driving advancements in automation and artificial intelligence."""
    )

@tool
def creator(input_str: str = "") -> str:
    """Provide Hasnain's introduction."""
    print("Tool Message: Developer Details Tool is Called!")
    print("=" * 40)
    return (
        """I am a Calculator Agent Developed By Hasnain Ali.
        If You Want to know About Hasnain Ali, Then Enter 'Who Is Hasnain?'.
        """
    )

@tool
def goodbye(input_str: str = "") -> str:
    """Stop the Agent."""
    print("Tool Message: Goodbye Tool is Called!")
    print("=" * 40)
    return "Goodbye! Thanks for your visit. Come again!"

@tool
def give_social_accounts(input_str: str = "") -> str:
    """Provide Hasnain's social accounts."""
    print("Tool Message: Contact Details Tool is Called!")
    print("=" * 40)
    return (
        """
        Hasnain's LinkedIn: https://www.linkedin.com/in/hasnain-ali-developer/ \n
        Hasnain's GitHub: https://github.com/HasnainCodeHub \n
        Hasnain's Instagram: https://www.instagram.com/i_hasnainaliofficial/ \n
        Hasnain's Facebook Profile: https://www.facebook.com/hasnainazeem.hasnainazeem.1 \n
        Hasnain's Email Address: husnainazeem048@gmail.com \n
        Hasnain's Contact Number: 03702537927
        """
    )

tools = [
    add,
    subtract,
    multiply,
    divide,
    intro,
    creator,
    goodbye,
    give_social_accounts
]

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key  = GOOGLE_API_KEY, model="gemini-2.0-flash-exp", verbose=True)

# =========================================AGENT=======================================================
# Initialize the agent
agent = initialize_agent(
    tools,                         # Provide the tools
    llm,                           # LLM for fallback
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=50
)

# Streamlit App
st.title("AI-Based Calculator Agent")
st.write("Welcome to Hasnain's Coding World!")

user_query = st.text_input("Ask your query:")
if st.button("Submit"):
    if user_query.strip():  # Check if input is not empty
        try:
            response = agent.invoke({"input": user_query})  # 'input' key in lowercase
            st.write(response.get('output', 'No output available'))  # Safely access the response
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to proceed.")
