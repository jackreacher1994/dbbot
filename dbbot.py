import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import OpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents import initialize_agent
from langchain.agents import AgentType

st.set_page_config(page_title="DbBot", page_icon="📊")
st.header('📊 Welcome to DbBot, your companion for working with SQL databases.')

db_types = {"PostgreSQL": "postgresql+psycopg2", "MySQL": "mysql+pymysql", "SQLite": "sqlite"}
dialect_types = {"PostgreSQL": "postgresql", "MySQL": "mysql", "SQLite": "sqlite"}
st.sidebar.title("DbBot")
selected_db_type = st.sidebar.selectbox("Please select the type of your database:", options=db_types.keys())
db_type = db_types[selected_db_type]
db_uri = st.sidebar.text_area(label="Please enter the uri your database:", placeholder="postgres:postgres@localhost:5432/pagila")

load_dotenv()

if db_type and db_uri:
    db = SQLDatabase.from_uri(db_type + "://" + db_uri)
else:
    st.error("Please select the type of your database and enter the uri.")
    st.stop()

llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
# llm = OpenAI()

working_directory = os.getcwd()
tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["read_file", "write_file", "list_directory"],).get_tools()
tools.append(PythonREPLTool())
tools.extend(SQLDatabaseToolkit(db=db, llm=llm).get_tools())

prompt_prefix = """ 
##Instructions:
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
As part of your final answer, ALWAYS include an explanation of how to got to the final answer, including the SQL query you run. Include the explanation and the SQL query in the section that starts with "Explanation:".

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don\'t know" as the answer.

##Tools:

"""

prompt_format_instructions = """ 
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

Explanation:

<===Beging of an Example of Explanation:

I joined the invoices and customers tables on the customer_id column, which is the common key between them. This will allowed me to access the Total and Country columns from both tables. Then I grouped the records by the country column and calculate the sum of the Total column for each country, ordered them in descending order and limited the SELECT to the top 5.

```sql
SELECT c.country AS Country, SUM(i.total) AS Sales
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY Country
ORDER BY Sales DESC
LIMIT 5;
```

===>End of an Example of Explanation
"""

agent = initialize_agent(
    tools, 
    llm, 
    agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True,
    output_parser=ReActSingleInputOutputParser(),
    agent_kwargs={
        "prefix": prompt_prefix,
        "format_instructions": prompt_format_instructions
    }
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        # try:
        response = agent.run(
            {
                "input": user_query,
                "dialect": dialect_types[selected_db_type],
                "top_k": 10
            }, callbacks = [st_cb])
        # except ValueError as e:
        #     response = f"⚠️ Error parsing the response: {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

