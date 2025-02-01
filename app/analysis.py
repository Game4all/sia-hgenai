from .utils.format import prompt_template
from .utils.bedrock import WrapperBedrock, ConverseMessage


from app.utils.format import prompt_template, parse_json_response
from app.utils.bedrock import WrapperBedrock, ConverseMessage
from pydantic import BaseModel

RISQUES = [
    # Risques physiques aigus :
    "Inondation",
    "Feu de forêt"
    "Événements cycloniques (tempêtes)",
    "Tremblement de terre",
    "Sécheresse",
    "Vague de chaleur",

    # Risques physiques chroniques :
    "Retrait-gonflement des argiles",
    "Érosion du littoral",
    "Élévation du niveau de la mer",
    "Perte d’enneigement",

    # Risques environnementaux :
    "Stress hydrique",
    "Perte de biodiversité",
    "Pollution de l’air, des sols, de l’eau",
    "Gestion des déchets",
]


class AnalyzedRisk(BaseModel):
    nom_risque: str
    identification_risque: str
    plan_adaptation_risque: str | None


class RiskAnalysisOutput(BaseModel):
    risques: list[AnalyzedRisk]
    note: float
    explication: str
    url: str | None = None


@prompt_template
def risk_analysis_prompt(liste_risques: list[str], doc: str) -> str:
    """
    Tu es un assistant de recherche spécialisé dans l'évaluation des risques climatiques des collectivités françaises. Ta tâche est de soutenir Sfil une banque d'investissement, dans la recherche et l'analyse des documents relatifs à l'adaptation des collectivités au risque climatique.

    Liste des risques:
    {% for risk in liste_risques -%}
        - {{risk}}
    {% endfor %}

    À partir du document ci-dessous, extrait les risques qui sont mentionnés dans le texte, ainsi que que le passage qui identifie ces risques, et le passage qui montre si les collectivités ont un plan d'adapation face à ce risque (laisse a vide si il n'y a pas de mention de plan).
    Enfin, évalue la pertinence du document pour l'analyse de risques avec une note de 1 à 10 en fonction de si le document montre que les risques trouvés sont bien identifiés et ont un plan de mitigation.

    Réponds en utilisant le schéma JSON suivant:
    {
        "risques": [
            {
                "nom_risque": "<nom du risque ici>",
                "identification_risque": "<passage qui mentionne l'identification du risque>",
                "plan_adaptation_risque: "<passage qui mentionne le plan d'adapation face au risque si il existe, sinon mets a null>"
            }
        ],
        "note": 1.0,
        "explication": "<ton raisonnement sur la note indiquée>"
    }

    Le document à analyser:

    {{doc}}
    """
    pass


def analyze_doc_risks(bedrock: WrapperBedrock, doc: str, doc_url: str) -> RiskAnalysisOutput:
    """
    Effectue l'analyse de risques sur le document spécifié.
    """
    analysis_response = bedrock.converse(model_id="mistral.mistral-large-2407-v1:0", messages=[
                                         ConverseMessage.make_user_message(risk_analysis_prompt(RISQUES, doc))], max_tokens=4096)
    out = RiskAnalysisOutput.model_validate(
        parse_json_response(analysis_response.content[0].text))

    out.url = doc_url
    return out