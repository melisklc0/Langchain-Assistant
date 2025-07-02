from datetime import datetime
from langchain_core.tools import tool
from modules.models import get_llm
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

@tool
def get_current_time() -> str:
    """Geçerli zamanı döndürür. Örnek: 14:30"""
    return datetime.now().strftime("%H:%M")

def run_react_agent():
    tools = [get_current_time]

    prompt = hub.pull("hwchase17/react")
    llm = get_llm()

    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,    
        stop_sequence=True
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        verbose=True
    )

    response = agent_executor.invoke({"input": "Şuanda saat kaç?"})
    print(response["output"])