import fitz
from ..utils.bedrock import ConverseMessage
from .executor import agent_task, AgentExecutor
from ..analysis import analyze_doc_risks, RiskAnalysisOutput
from ..utils.scrapper import scrapper
import dotenv
import os

dotenv.load_dotenv()

@agent_task("SEARCH_DOCS")
def search_docs(exec: AgentExecutor, args: dict) -> dict:
    sc = scrapper(num_results=1, pipe="mistral.mistral-7b-instruct-v0:2", googlecred=os.environ["SCRAPPER_API"], googleidengin=os.environ["SCRAPPER_ENGINE"])
    print("begin search georisques")
    geo_result = sc.repport_geoRisk(city=args["lieux"].split(",")[0])
    print("begin search docs")
    docs_results = sc.find_doc(args["lieux"].split(",")[0], args["docs"])
    if geo_result:
        docs_results.append(geo_result)
    return docs_results


@agent_task("ANALYZE_DOCS")
def analyze_documents(exec: AgentExecutor, args: dict) -> dict:
    if "in" not in args:
        return {}
    else:
        files = exec.get_inputs(args["in"])
        analysis = []
        for f in files:
            if len(f["pdf"]) > 200000:
                f["pdf"] = f["pdf"][:200000]
                print(f"Le fichier {f['url']} est trop grand, il a été tronqué à 200000 caractères")
            print("Analyse de ", f["url"])
            analysis.append(analyze_doc_risks(
                exec.bedrock, f["pdf"], f["url"], "anthropic.claude-3-5-sonnet-20241022-v2:0", args.get("risques", None)))

    return analysis


@agent_task("DATAVIZ")
def dataviz(exec: AgentExecutor, args: dict) -> dict:
    pass


@agent_task("SYNTHESIZE")
def synth(exec: AgentExecutor, args: dict) -> dict:
    data = [d.model_dump_json() for d in exec.get_inputs(args["in"])]

    test_prompt = f"Fais une synthèse globale des risques à partir des données de risques qui sont au format JSON: \n {data}. Pour chaque risque identifié, nomme le risque, et le plan de mitigation si il y'en a un."
    c = exec.bedrock.converse(model_id="mistral.mistral-large-2407-v1:0",
                              messages=[ConverseMessage.make_user_message(test_prompt)], max_tokens=4096)
    return c.content[0].text
