import time
import streamlit as st
from app.planning.subtasks import plan_actions
from app.planning.executor import AgentExecutor, agent_task
from app.planning.tasks import search_docs, analyze_documents, synth

from app.utils.bedrock import WrapperBedrock
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.image("sfil.png", width=100)

@st.cache_resource
def get_bedrock() -> WrapperBedrock:
    return WrapperBedrock()


@agent_task("DATAVIZ")
def dataviz(exec: AgentExecutor, args: dict) -> None:
    print("DATAVIZ")
    time.sleep(3)
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

    execution_status = bot_reply.status("Préparation")

    # récupération du planning de l'agent
    planning = plan_actions(get_bedrock(), validation_model_id="mistral.mistral-large-2407-v1:0",
                            planning_model_id="mistral.mistral-large-2407-v1:0", user_request=prompt)

    if "error" in planning:
        execution_status.update(state="error")
        st.session_state["messages"].append(
            {"role": "assistant", "content": "🛑 " + planning["error"]})
        bot_reply.write("🛑 " + planning["error"])

    else:
        if "tasks" in planning:
            execution_status.update(
                label="Execution des tâches ...", state="running")

            exec = AgentExecutor(get_bedrock())
            exec.register_task(search_docs)
            exec.register_task(analyze_documents)
            exec.register_task(dataviz)
            exec.register_task(synth)

            for id, task in enumerate(exec.execute_tasks(planning["tasks"])):
                execution_status.update(
                    label="{} ({} / {})".format(task, id + 1, len(planning["tasks"])))

        if "synthesize_output" in exec.outputs:
            bot_reply.write(exec.get_inputs("synthesize_output"))

        execution_status.update(state="complete")
