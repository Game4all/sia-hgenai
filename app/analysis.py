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
    identification_risque: str | None
    plan_adaptation_risque: str | None


class RiskAnalysisOutput(BaseModel):
    risques: list[AnalyzedRisk]
    note: float
    explication: str
    url: str | None = None


@prompt_template
def risk_analysis_prompt(liste_risques: list[str], doc: str) -> str:
    """
    Tu es un assistant de recherche spécialisé dans l'évaluation des risques climatiques des collectivités françaises. 
    Ta mission est d'aider SFIL, une banque d'investissement, dans l'analyse des documents liés à l'adaptation des collectivités aux risques climatiques.

    **Liste des risques à analyser** :
    {% for risk in liste_risques -%}
        - {{ risk }}
    {% endfor %}

    **Tâches à effectuer** :
    - Identifier les passages du document mentionnant un ou plusieurs des risques listés.
    - Extraire les passages qui identifient les risques.
    - Extraire les passages qui mentionnent un éventuel plan d’adaptation pour ces risques (laisser `null` si aucun plan n'est mentionné).
    - Attribuer une note de pertinence de 1 à 10 en fonction de la qualité des informations fournies sur les risques et les stratégies d’adaptation.
    - Justifier la note attribuée.
    
    Veuillez analyser le document suivant :
    {{ doc }}

    Après avoir analysé le document, retourne UNIQUEMENT un JSON contenant les informations suivantes avec ce format sans texte additionnel :
    {
        "risques": [
            {
                "nom_risque": "<nom du risque>",
                "identification_risque": "<passage du document identifiant le risque>",
                "plan_adaptation_risque": "<passage mentionnant un plan d'adaptation ou null si inexistant>"
            }
        ],
        "note": <valeur entre 1 et 10>,
        "explication": "<raisonnement expliquant la note>"
    }

    Exemple de réponse :
    {
        "risques": [
            {
                "nom_risque": "Inondation",
                "identification_risque": "Le document mentionne que la commune est située en zone inondable.",
                "plan_adaptation_risque": "Un plan de prévention des inondations est en cours d'élaboration."
            },
            {
                "nom_risque": "Feu de forêt",
                "identification_risque": "Le document mentionne que la commune est située en zone à risque de feu de forêt.",
                "plan_adaptation_risque": null
            }
        ],
        "note": 8,
        "explication": "Les risques sont bien identifiés mais les plans d'adaptation ne sont pas toujours mentionnés."
    }
    
    """
    pass


def analyze_doc_risks(
        bedrock: WrapperBedrock,
        doc: str,
        doc_url: str,
        analyse_model_id: str,
        risques: list[str] = None,
        max_retires: int = 3
    ) -> RiskAnalysisOutput:
    """ 
    Effectue l'analyse de risques sur le document spécifié.

    Args:
        bedrock (WrapperBedrock): Instance Bedrock pour effectuer l'analyse.
        doc (str): Contenu du document à analyser.
        doc_url (str): URL du document.
        analyse_model_id (str): ID du modèle d'analyse.
        risques (list[str], optional): Liste des risques à analyser. Defaults to None.
        max_retires (int, optional): Nombre maximal de tentatives pour effectuer l'analyse. Defaults to 3.

    Returns:
        RiskAnalysisOutput: Résultat de l'analyse.
    """
    retries = 0
    risques = RISQUES if risques is None else risques
    prompt = risk_analysis_prompt(risques, doc)
    while retries < max_retires:
        try:
            analysis_response = bedrock.converse(model_id=analyse_model_id, messages=[
                                                ConverseMessage.make_user_message(prompt)],
                                                max_tokens=8192)
            out = RiskAnalysisOutput.model_validate(
                parse_json_response(analysis_response.content[0].text))
            out.url = doc_url
            break
        except Exception as e:
            retries += 1
            if retries >= max_retires:
                raise e
            else:
                # bail out prompt
                instruction = (
                    "Veuillez vérifier le format du prompt ainsi que le nombre de tokens utilisés dans la requête. "
                    "Assurez-vous que la réponse respecte strictement le format JSON suivant :\n"
                    "{\n"
                    '    "risques": [\n'
                    '        {\n'
                    '            "nom_risque": "<nom du risque>",\n'
                    '            "identification_risque": "<passage identifiant le risque>",\n'
                    '            "plan_adaptation_risque": "<passage mentionnant un plan ou null>"\n'
                    '        }\n'
                    '    ],\n'
                    '    "note": <valeur entre 1 et 10>,\n'
                    '    "explication": "<raisonnement expliquant la note>"\n'
                    "}\n"
                    "Si la réponse dépasse la limite de tokens, essayez de réduire la longueur du document ou la liste des risques analysés."
                )
                prompt += instruction
        
    return out