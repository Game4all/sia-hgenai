from app.utils.bedrock import WrapperBedrock, ConverseMessage
from app.utils.format import prompt_template, parse_json_response
from app.analysis import AnalyzedRisk


@prompt_template
def recommend_dataviz_template(choices: list[str], risks: list[AnalyzedRisk]) -> str:
    """
        Choisis parmi les options disponibles ci-dessous la visualisation de données la plus adaptée pour les risques prédominants suivants: 

        Risques prédominants:
        {% for risk in risks -%}
        - {{risk.nom_risque}}
        {% endfor %}


        **Options:**
        {% for choice in choices -%}
        - {{choice}}
        {% endfor %}

        Réponds en utilisant le schéma JSON suivant:
        {
            "visualization": "<visualisation choisie de la liste au dessus>"
        }

        Donne uniquement ton choix.
    """


def recommend_dataviz(bedrock: WrapperBedrock, choices: list[str], risks: list, model_id: str = "mistral.mistral-large-2407-v1:0") -> str:
    resp = parse_json_response(bedrock.converse(model_id=model_id, messages=[
        ConverseMessage.make_user_message(recommend_dataviz_template(choices, risks))]).content[0].text)
    return resp["visualization"]
