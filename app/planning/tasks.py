import fitz
from ..utils.bedrock import ConverseMessage
from .executor import agent_task, AgentExecutor
from ..analysis import analyze_doc_risks, RiskAnalysisOutput
from ..utils.scrapper import scrapper


@agent_task("SEARCH_DOCS")
def search_docs(exec: AgentExecutor, args: dict) -> dict:
    sc = scrapper(num_results=5)
    # TODO: harcode sur le georisque pour l'instant
    results = [sc.repport_geoRisk(city=args["lieux"])]
    return results


@agent_task("ANALYZE_DOCS")
def analyze_documents(exec: AgentExecutor, args: dict) -> dict:
    if "in" not in args:
        return {}
    else:
        files = exec.get_inputs(args["in"])
        analysis = []

        for f in files:
            doc = fitz.open(stream=f["pdf"])
            doc_text = "\n".join(page.get_text("text") for page in doc)

            analysis.append(analyze_doc_risks(
                exec.bedrock, doc_text, f["url"]))

            doc.close()

    return analysis


@agent_task("SYNTHESIZE")
def synth(exec: AgentExecutor, args: dict) -> dict:
    data = [d.model_dump_json() for d in exec.get_inputs(args["in"])]

    test_prompt = f"Fais une synthèse globale des risques à partir des données de risques qui sont au format JSON: \n {data}. Pour chaque risque identifié, nomme le risque, et le plan de mitigation si il y'en a un."
    c = exec.bedrock.converse(model_id="mistral.mistral-large-2407-v1:0",
                              messages=[ConverseMessage.make_user_message(test_prompt)], max_tokens=4096)
    return c.content[0].text
