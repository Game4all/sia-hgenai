import time
import pandas as pd
import streamlit as st
from app.planning.subtasks import plan_actions
from app.planning.executor import AgentContext, agent_task
from app.planning.tasks import search_docs, analyze_documents, synth, dataviz
from app.dataviz import recommend_dataviz_suggestion

import folium
import streamlit_folium as stf

from app.utils.bedrock import WrapperBedrock
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.image("sfil.png", width=100)

@st.cache_resource
def get_bedrock() -> WrapperBedrock:
    return WrapperBedrock()


@st.cache_resource
def get_agent_context() -> AgentContext:
    ag = AgentContext(get_bedrock())
    ag.register_task(search_docs)
    ag.register_task(analyze_documents)
    ag.register_task(dataviz)
    ag.register_task(synth)
    return ag


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Bonjour, je suis Sfil RiskDoc! Prompte moi ce que tu veux ou utilise un bouton pour essayer rapidement"}]

chat_container = st.container(height=800)
chat_messages_container = chat_container.container(height=720)

# affichage des précédents messages
for msg in st.session_state["messages"]:
    msg_c = chat_messages_container.chat_message(msg["role"])
    msg_c.write(msg["content"])
    if "embed" in msg:
        with msg_c:
            stf.st_folium(msg["embed"], width=500)

    if "fig" in msg:
        msg_c.plotly_chart(msg["fig"])

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

    if "tasks" in planning:
        execution_status.update(
            label="Execution des tâches ...", state="running")

        exec = get_agent_context()
        exec.reset()

        for id, task in enumerate(exec.execute_tasks(planning["tasks"])):
            execution_status.update(
                label="{} ({} / {})".format(task, id + 1, len(planning["tasks"])))

    # optionnal data viz
    viz = exec.get_inputs(
        "dataviz_output") if "dataviz_output" in exec.outputs else None

    if "synthesize_output" in exec.outputs:
        msg = {"role": "assistant",
               "content": exec.get_inputs("synthesize_output")}
        if viz:
            if type(viz) == type(folium.Map):
                msg["embed"] = viz
            else:
                msg["fig"] = viz

        st.session_state["messages"].append(msg)
        bot_reply.write(exec.get_inputs("synthesize_output"))

        if viz:
            with bot_reply:
                if type(viz) == type(folium.Map):
                    stf.st_folium(viz, width=500)
                else:
                    st.plotly_chart(viz)

    execution_status.update(state="complete")
