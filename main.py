import time
import streamlit as st
from app.planning.subtasks import plan_actions
from app.planning.executor import AgentExecutor, agent_task

from app.utils.bedrock import WrapperBedrock
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_bedrock() -> WrapperBedrock:
    return WrapperBedrock()


@agent_task("SEARCH_DOCS")
def search_docs(exec: AgentExecutor, args: dict) -> None:
    print("SEARCH_DOCS")
    time.sleep(1)
    return {"DOCS": ["salut.pdf", "bite.lol"]}


@agent_task("ANALYZE_DOCS")
def analyze_docs(exec: AgentExecutor, args: dict) -> None:
    documents = exec.outputs[args["in"]]
    print(f"ANALYZE_DOCS: {documents}")
    return None


@agent_task("DATAVIZ")
def dataviz(exec: AgentExecutor, args: dict) -> None:
    print("DATAVIZ")
    time.sleep(3)
    return None


@agent_task("SYNTHESIZE")
def synthesize(exec: AgentExecutor, args: dict) -> None:
    print("SYNTHESIZE")
    return None


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Bonjour, je suis Sfil RiskDoc! Prompte moi ce que tu veux ou utilise un bouton pour essayer rapidement"}]

chat_container = st.container(height=800)
chat_messages_container = chat_container.container(height=720)

# affichage des précédents messages
for msg in st.session_state["messages"]:
    chat_messages_container.chat_message(msg["role"]).write(msg["content"])

prompt = chat_container.chat_input(placeholder="Entrez votre prompt")

if prompt:
    # add the typed in message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    chat_messages_container.chat_message("user").write(prompt)

    bot_reply = chat_messages_container.chat_message("assistant")

    planning_status = bot_reply.status("Plannification")

    # récupération du planning de l'agent
    planning = plan_actions(get_bedrock(), validation_model_id="mistral.mistral-large-2407-v1:0",
                            planning_model_id="mistral.mistral-large-2407-v1:0", user_request=prompt)

    if "tasks" in planning:
        planning_status.write(planning["tasks"])
        planning_status.update(state="complete")

    if "error" in planning:
        planning_status.update(state="error")
        st.session_state["messages"].append(
            {"role": "assistant", "content": "🛑 " + planning["error"]})
        bot_reply.write("🛑 " + planning["error"])

    if "tasks" in planning:
        exec = AgentExecutor()
        exec.register_task(search_docs)
        exec.register_task(analyze_docs)
        exec.register_task(dataviz)
        exec.register_task(synthesize)

        for task in exec.execute_tasks(planning["tasks"]):
            status = bot_reply.status(label=task)
            status.update(state="complete")
