import fitz
import folium.plugins
from ..utils.bedrock import ConverseMessage
from .executor import agent_task, AgentContext
from ..analysis import analyze_doc_risks, RiskAnalysisOutput
from ..utils.scrapper import scrapper
from ..dataviz import generate_visualization, recommend_dataviz_suggestion, slotfill_viz
import folium as folium
import plotly.express as px
import os


@agent_task("SEARCH_DOCS")
def search_docs(exec: AgentContext, args: dict) -> dict:
    sc = scrapper(num_results=1, pipe="mistral.mistral-7b-instruct-v0:2",
                  googlecred=os.environ["SCRAPPER_API"], googleidengin=os.environ["SCRAPPER_ENGINE"])
    print("begin search georisques")
    geo_result = sc.repport_geoRisk(city=args["lieux"].split(",")[0])
    print("begin search docs")
    docs_results = sc.find_doc(args["lieux"].split(",")[0], args["docs"])
    if geo_result:
        docs_results.append(geo_result)
    return docs_results


@agent_task("ANALYZE_DOCS")
def analyze_documents(exec: AgentContext, args: dict) -> list:
    if "in" not in args:
        return {}
    else:
        files = exec.get_inputs(args["in"])
        analysis = []
        for f in files:
            if len(f["pdf"]) > 200000:
                f["pdf"] = f["pdf"][:200000]
                print(
                    f"Le fichier {f['url']} est trop grand, il a été tronqué à 200000 caractères")
            print("Analyse de ", f["url"])
            analysis.append(analyze_doc_risks(
                exec.bedrock, f["pdf"], f["url"], "anthropic.claude-3-5-sonnet-20241022-v2:0", args.get("risques", None)))

    return analysis


@agent_task("DATAVIZ")
def dataviz(exec: AgentContext, args: dict) -> dict:
    # Récupération des données analysées
    # risques = args.get("risques", [])
    lieu = args.get("lieux", "").split(",")[0]
    analyzed_risks = exec.get_inputs(args["in"])

    sc = scrapper()

    # recuperation de la suggestion de dataviz
    source = recommend_dataviz_suggestion(
        exec.bedrock, choices=["histo", "carte"], risks=analyzed_risks[0].risques)

    # select de la data source
    datasource = sc.get_accident_history(
        lieu)["df"] if source["source"] == 'CATNAT-GEORISQUE' else None

    match source["visualization"]:
        case "histo":
            colonnes = datasource.columns.tolist()
            best_colonne = slotfill_viz(
                exec.bedrock, source["visualization"], colonnes)

            fig = px.histogram(
                datasource[best_colonne["col"]], nbins=30, title=best_colonne["titre"])
            return fig
        case "carte":
            insee = sc.get_insee_code(city_name=lieu)
            long_lalt = sc.get_city_coordinates(insee)

            fmap = folium.Map(
                location=[long_lalt["latitude"], long_lalt["longitude"]], zoom_start=8)

            fmap.add_child(folium.Marker(
                location=[long_lalt["latitude"], long_lalt["longitude"]]))

            fmap.add_child(folium.Circle(
                location=[long_lalt["latitude"], long_lalt["longitude"]],
                radius=1000,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.3
            ))

            return fmap

    # print(risques)
    # print(lieu)

    # # Génération du code de visualisation via le modèle
    # visualization = generate_visualization(exec.bedrock, risques, lieu)
    # visualization_code = visualization.get("visualization_code")

    # # Exécution du code Python généré
    # try:
    #     # Création d'un espace d'exécution isolé
    #     local_env = {}
    #     exec(visualization_code, {"plt": plt}, local_env)

    #     # Sauvegarde de l'image dans un buffer mémoire
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.close()
    #     buf.seek(0)

    #     # Encodage de l'image en base64 pour l'intégration dans la réponse
    #     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    #     img_data = f"data:image/png;base64,{img_base64}"

    #     # Retour de la réponse avec l'image intégrée
    #     return {
    #         "visualization": img_data,
    #         "description": visualization.get("description", ""),
    #         "insights": visualization.get("insights", "")
    #     }

    # except Exception as e:
    #     return {"error": f"Erreur lors de la génération de la visualisation : {str(e)}"}


@agent_task("SYNTHESIZE")
def synth(exec: AgentContext, args: dict) -> dict:
    data = [d.model_dump_json() for d in exec.get_inputs(args["in"])]

    test_prompt = f"Fais une synthèse globale des risques à partir des données de risques qui sont au format JSON: \n {data}. Pour chaque risque identifié, nomme le risque, et le plan de mitigation si il y'en a un, et la source du risque."
    c = exec.bedrock.converse(model_id="mistral.mistral-large-2407-v1:0",
                              messages=[ConverseMessage.make_user_message(test_prompt)], max_tokens=4096)
    return c.content[0].text
