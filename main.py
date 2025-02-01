from dotenv import load_dotenv
from app.utils.format import prompt_template
from app.utils.bedrock import WrapperBedrock, ConverseMessage
import json

load_dotenv()

wrapper = WrapperBedrock()

@prompt_template
def risk_analysis_prompt(location: str, text: str, risk_list: str) -> str:
    """
    Tu es un assistant de recherche spécialisé dans l'évaluation des risques climatiques des collectivités françaises. 
    Ta tâche est de soutenir Sfil, une banque d'investissement, dans la recherche et l'analyse des documents relatifs à 
    l'adaptation des collectivités au risque climatique.

    Région : {{ location }}
    
    Liste des risques possibles : {{ risk_list }}
    
    Texte source :
    {{ text }}
    
    À partir des informations fournies, tu dois retourner les éléments suivants sous format JSON :
    {
        "risques_identifiés": ["risque1", "risque2"],
        "synthèse": "Synthèse des informations...",
        "degré_identification": "Phrase expliquant le degré d'identification du risque...",
        "actions_adaptation": ["Action1", "Action2"],
        "niveau_confiance": {
            "score": 1-10,
            "commentaire": "Commentaire expliquant le niveau de confiance..."
        }
    }
    """

def analyze_risk(text_parts: list, location: str) -> None:
    """
    Génère un fichier JSON avec une analyse du risque basé sur les données fournies.
    
    :param text_parts: Liste des morceaux de texte sur le risque.
    :param location: Nom du lieu (ex: "Paris").
    """
    risk_list = """
    Risques physiques aigus :
    - Inondation
    - Feu de forêt
    - Événements cycloniques (tempêtes)
    - Tremblement de terre
    - Sécheresse
    - Vague de chaleur
    
    Risques physiques chroniques :
    - Retrait-gonflement des argiles
    - Érosion du littoral
    - Élévation du niveau de la mer
    - Perte d’enneigement
    
    Risques environnementaux :
    - Stress hydrique
    - Perte de biodiversité
    - Pollution de l’air, des sols, de l’eau
    - Gestion des déchets
    """
    
    # Extraire le nom et le lien du document depuis la première ligne du texte
    first_line = text_parts[0].strip()
    doc_info = first_line.split(" - ", 1)
    document_name = doc_info[0] if len(doc_info) > 0 else "Non trouvé"
    document_link = doc_info[1] if len(doc_info) > 1 else "Non disponible"
    
    full_text = "\n".join(text_parts[1:])
    prompt = risk_analysis_prompt(location=location, text=full_text, risk_list=risk_list)
    
    response = wrapper.converse(messages=[ConverseMessage.make_user_message(prompt)], 
                                model_id="mistral.mistral-7b-instruct-v0:2")
    
    response_text = response.text if hasattr(response, "text") else str(response)
    
    result_json = {
        "lieu": location,
        "document": {
            "nom": document_name,
            "lien": document_link
        },
        "analyse": response_text
    }
    
    output_path = "output_risk.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    
    print(f"Analyse sauvegardée dans {output_path}")

if __name__ == "__main__":
    test_text_parts = [
        "DICRIM Paris - https://georisques.gouv.fr/dicrim/paris.pdf",
        "Le DICRIM de Paris décrit les risques majeurs, dont les inondations.",
        "Il précise les types d'inondations possibles : crues lentes et remontées de nappe.",
        "Des mesures d’adaptation sont mentionnées, comme la construction de digues et des plans d’évacuation.",
        "Ce document est fiable avec des sources officielles et une mise à jour récente."
    ]
    test_location = "Paris"

    analyze_risk(test_text_parts, test_location)