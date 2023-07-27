import os
from dotenv import load_dotenv
import streamlit as st
import requests
from typing import List, Dict

# from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent

st.set_page_config(layout="centered", page_title="AIJOBS")
st.title("GPT- Job Assistant")
system_message = SystemMessage(
    content="You are very custom designed helpful assistant in finding relevent jobs, you can use the tools when needed, tone should be friendly, and don't use the tools if the query is not related to jobs or carrer related info, please remember you should not answer if the query is not relevent to jobs or carrer related information"
)
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)

load_dotenv()


def custom_tool(query: str) -> List[Dict[str, str]]:
    url = "https://ai.joblab.ai/get_job_matches"
    query_params = {
        "query": query,
        "page": 1,
        "size": 7,  # You can adjust the size as needed to get more job matches
    }

    headers = {"accept": "application/json"}

    response = requests.post(url, params=query_params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        total_jobs = data["total"]
        page_size = data["size"]

        print(f"Total jobs matching the query: {total_jobs}")

        # Create a list to store all the job match data
        job_matches_data = []

        # Loop through each job match and capture the data dynamically
        for job_match in data["items"]:
            job_data = {
                "job_id": job_match["job_id"],
                "job_title": job_match["job_title"],
                "job_company": job_match["job_company"],
                "job_location": job_match["job_location"],
            }

            job_matches_data.append(job_data)

        return job_matches_data

    else:
        print(f"Request failed with status code: {response.status_code}")
        return []


st.sidebar.header("OpenAI API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
else:
    llm = ChatOpenAI(
        temperature=0.2,
        streaming=True,
        openai_api_key=openai_api_key,
    )

    # Create the tool object for your custom tool
    custom_tool_object = Tool.from_function(
        func=custom_tool,
        name="Job_FInder",
        description="use This tool to find job matches based on a given query, and You can use this tool to explore job opportunitiesthe field of your interest and do not use this tool if the query is not related to jobs or carrer",
    )

    tools = [custom_tool_object]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Initialize the agent with the tools, LLMMathChain, and agent type
    # agent = initialize_agent(
    #     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    # )

    # Use the agent to run a query
    # response = agent.run("hey help me to find some jobs for me in Data science?")
    # print(response)

    # llm = OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
    # tools = load_tools(["ddg-search"])
    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     # verbose=True
    # )

    # try: "what are the names of the kids of the 44th president of america"
    # try: "top 3 largest shareholders of nvidia"
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.write("🧠 thinking...")
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.run(prompt, callbacks=[st_callback])
            st.write(response)
