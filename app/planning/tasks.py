import fitz
from ..utils.bedrock import ConverseMessage
from .executor import agent_task, AgentExecutor
from ..analysis import analyze_doc_risks, RiskAnalysisOutput
from ..utils.scrapper import scrapper
from ..dataviz import generate_visualization
import matplotlib.pyplot as plt
import io
import base64


@agent_task("SEARCH_DOCS")
def search_docs(exec: AgentExecutor, args: dict) -> dict:
    sc = scrapper(num_results=5)
    # TODO: harcode sur le georisque pour l'instant
    results = [sc.repport_geoRisk(city=args["lieux"])]
    return results


@agent_task("ANALYZE_DOCS")
def analyze_documents(exec: AgentExecutor, args: dict) -> list:
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


@agent_task("DATAVIZ")
def dataviz(exec: AgentExecutor, args: dict) -> dict:
    # Récupération des données analysées    
    risques = args.get("risques", [])
    lieu = args.get("lieux", "").split(",")[0]

    print(risques)
    print(lieu)

    # Génération du code de visualisation via le modèle
    visualization = generate_visualization(exec.bedrock, risques, lieu)
    visualization_code = visualization.get("visualization_code")

    # Exécution du code Python généré
    try:
        # Création d'un espace d'exécution isolé
        local_env = {}
        exec(visualization_code, {"plt": plt}, local_env)
        
        # Sauvegarde de l'image dans un buffer mémoire
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Encodage de l'image en base64 pour l'intégration dans la réponse
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        img_data = f"data:image/png;base64,{img_base64}"

        # Retour de la réponse avec l'image intégrée
        return {
            "visualization": img_data,
            "description": visualization.get("description", ""),
            "insights": visualization.get("insights", "")
        }
    
    except Exception as e:
        return {"error": f"Erreur lors de la génération de la visualisation : {str(e)}"}


@agent_task("SYNTHESIZE")
def synth(exec: AgentExecutor, args: dict) -> dict:
    data = [d.model_dump_json() for d in exec.get_inputs(args["in"])]

    test_prompt = f"Fais une synthèse globale des risques à partir des données de risques qui sont au format JSON: \n {data}. Pour chaque risque identifié, nomme le risque, et le plan de mitigation si il y'en a un."
    c = exec.bedrock.converse(model_id="mistral.mistral-large-2407-v1:0",
                              messages=[ConverseMessage.make_user_message(test_prompt)], max_tokens=4096)
    return c.content[0].text
