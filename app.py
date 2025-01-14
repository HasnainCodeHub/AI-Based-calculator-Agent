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

# Suggested queries
suggested_queries = [
    "What is 5 + 3?",
    "Can you subtract 10 from 20?",
    "Who is Hasnain?",
    "Who is Founder/Developer/Creator?",
    "Give me Hasnain's social accounts.",
    "Multiply 7 and 8.",
    "Divide 100 by 4.",
    "Perform multiple operations like add 5 and 3, then multiply by 2."
]

st.write("### Suggested Queries:")
for query in suggested_queries:
    st.write(f"- {query}")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # Store user queries and responses

# Input with Arrow Button
# Input with Arrow Button
col1, col2 = st.columns([4, 1])  # Adjust column sizes to previous proportions
with col1:
    user_query = st.text_input( "Enter your query and press ➡️")
    if st.button("➡️"):  # Arrow button for submission
        if user_query.strip():  # Check if input is not empty
            try:
                # Invoke the agent with the user's query
                response = agent.invoke({"input": user_query})
                agent_response = response.get('output', 'No output available')

                # Update conversation history
                st.session_state.conversation.append((user_query, agent_response))

                # Display the conversation history
                st.write("### Conversation History:")
                for i, (query, reply) in enumerate(st.session_state.conversation, 1):
                    st.write(f"Human Message: {query}")
                    st.write(f"Agent Response:  {reply}")
                    st.write("---")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query to proceed.")
