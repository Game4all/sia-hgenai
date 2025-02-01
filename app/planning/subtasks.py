from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Optional, Any
from ..utils.format import prompt_template, parse_json_response
import json
from ..utils.bedrock import WrapperBedrock, ConverseMessage
import logging
import re

class SubTask(BaseModel):
    id: int = Field(..., description="Identifiant unique de la sous-tâche")
    description: str = Field(..., description="Description de la sous-tâche")
    args: Optional[Dict[str, Any]] = Field({}, description="Arguments supplémentaires pour la sous-tâche")


@prompt_template
def validate_prompt_template(prompt: str) -> str:
    """
    Vérifie que la requête utilisateur a pour sujet l'analyse ou l'adaptaion des collectivites francasies face aux  des risques environnementaux. \n
    Vérifie que le requête utilisateur contient au moins un des risques suivants ou catégories de risques :\n"
    **Risques physiques aigus** :\n"
    - Inondation\n"
    - Feu de forêt\n"
    - Événements cycloniques (tempêtes)\n"
    - Tremblement de terre\n"
    - Sécheresse\n"
    - Vague de chaleur\n\n"
    **Risques physiques chroniques** :\n"
    - Retrait-gonflement des argiles\n"
    - Érosion du littoral\n"
    - Élévation du niveau de la mer\n"
    - Perte d’enneigement\n\n"
    **Risques environnementaux** :\n"
    - Stress hydrique\n"
    - Perte de biodiversité\n"
    - Pollution de l’air, des sols, de l’eau\n"
    - Gestion des déchets\n\n"
    De plus, assure-toi qu'un lieu en France est mentionné. Indique également son niveau administratif (commune, département, région, groupement de communes, etc.).\n"
    Si les deux critères sont remplis, corrige l'orthographe et la grammaire si nécessaire et trouve les mots-clés pertinents en plus, comme par exemple la période de temps, le type de lieu (hôpital, école, etc.) \n"
    Si un critère manque, indique précisément ce qui est absent et donne un exemple de requête valide basée sur le prompt. \n"
    ⚠️ IMPORTANT : Retourne uniquement un JSON **valide** et **parsable**.\n"
    - AUCUN texte avant ou après le JSON.\n"
    - NE PAS entourer la réponse avec ```json ou tout autre balisage.\n"
    - Vérifie que toutes les clés et valeurs sont bien formatées.\n"
    - Chaque sous-tâche doit inclure les champs obligatoires.\n\n"
    Retourne un JSON sous la forme suivante :\n"
    ✅ Si valide : {\n"
        'requete_valide': true,\n"
        'message': '<requête reformulée>',\n"
        'risques': ['<risque1 ou catégorie1>', '<risque2 ou catégorie2>', ...],\n"
        'lieu': ['<lieu1>', '<lieu2>'],\n"
        'niv_admin': '<niveau_administratif>',\n"
        'kwargs': {'<clé1>': '<valeur1>', '<clé2>': '<valeur2>', ...}\n"
    }\n"
    ❌ Sinon : {\n"
        'requete_valide': false,\n"
        'message': '<explication du problème>'\n"
    }\n\n"
    ⚠️ Assure-toi que :\n"
    - Toutes les clés sont entourées de guillemets doubles (`\"`) pour respecter le format JSON.\n"
    - Chaque valeur de liste est dans des crochets `[]`.\n"
    - Pas de virgule en trop après le dernier élément d'un objet ou d'une liste.\n"
    - Chaque texte de la réponse est bien encodé en UTF-8.\n"
    
    Voici la requête à traiter : \n
    **Requête utilisateur :** {{ prompt }}
    """
    pass


def validate_prompt(prompt: str, client: WrapperBedrock, model_id: str) -> Dict[str, Any]:
    """
    Valide une requête utilisateur pour s'assurer qu'elle contient les informations nécessaires.

    :param prompt: Prompt pour guider le modèle lors de la correction.
    :param client: Instance de WrapperBedrock pour interagir avec le LLM.
    :param model_id: ID du modèle Bedrock à utiliser.

    :return: Dictionnaire contenant les informations validées ou un message d'erreur.
    """
    instruction = validate_prompt_template(prompt)
    messages = [ConverseMessage.make_user_message(instruction)]
    response = client.converse(model_id=model_id, messages=messages)
    
    try:
        parsed_response = parse_json_response(response.content[0].text)
        if isinstance(parsed_response, dict) and {"requete_valide", "message", "risques", "lieu", "niv_admin"} <= parsed_response.keys():
            return parsed_response
        else:
            raise json.JSONDecodeError
    except Exception as e:
        print(e)
        pass
    
    return {"requete_valide": False, "message": "Oups... Nous n'avons pas pu valider la requête. Veuillez réessayer."}


@prompt_template
def _generate_prompt_template(user_request: str, few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Vous êtes un planificateur de tâches avancé au sein de SFIL, une banque d'investissement spécialisée dans la recherche et l'analyse des documents relatifs à l'adaptation des collectivités aux risques climatiques. Votre tâche consiste à analyser la requête utilisateur et à diviser le processus en sous-tâches clairement définies. Ces sous-tâches doivent être organisées en fonction de l'ordre d'exécution et des dépendances. Elles doivent couvrir les étapes suivantes, qui sont génériques et peuvent varier légèrement selon le cas spécifique :

    1. **Recherche** : Identification et collecte des documents et données pertinentes relatives aux risques climatiques et à l’adaptation des collectivités.
    2. **Analyse** : Analyse des documents pour comprendre l'impact des risques climatiques sur les collectivités et évaluer les mesures d’adaptation existantes.
    3. **Visualisation** : Création de visualisations (graphiques, cartes, etc.) permettant de mieux comprendre et communiquer les résultats de l’analyse.
    4. **Synthèse** : Élaboration d’un rapport détaillé, résumant les résultats de l’analyse et proposant des recommandations adaptées pour les collectivités.

    Chaque sous-tâche doit inclure :
    - Une **description claire** et précise de l’étape à réaliser.
    - Un **ordre d'exécution** définissant dans quel ordre les sous-tâches doivent être réalisées.
    - Une **liste de dépendances** (si applicable) spécifiant les sous-tâches dont celle-ci dépend avant de pouvoir être exécutée.

    La structure du format JSON pour chaque sous-tâche est la suivante :
    {
        "id": int,                      # Identifiant unique de la sous-tâche
        "description": str,             # Description détaillée de la sous-tâche
    }

    {% if few_shot_examples %}
    Voici quelques exemples de tâches correctement divisées :

    {% for example in few_shot_examples %}
    ---
    **Requête:** {{ example['request'] }}
    **Sous-tâches générées :**
    {% for subtask in example['subtasks'] %}
    - **id :** {{ subtask['id'] }}
    - **description :** {{ subtask['description'] }}
    - **dependencies :** {{ subtask.get('dependencies', []) }}
    - **order :** {{ subtask['order'] }}
    {% endfor %}
    {% endfor %}
    ---
    {% endif %}

    Voici la requête à traiter : \n

    **Requête utilisateur :**
    {{ user_request }} \n

    Retournez UNIQUEMENT un JSON contenant la liste des sous-tâches sans autre texte.
    """
    pass


def divide_task(prompt: str,
            client: WrapperBedrock,
            model_id: str,
            max_retries: int) -> Optional[List[SubTask]]:
    """
    Logique pour valider les sous-tâches générées par le modèle Bedrock.
    Si une erreur est détectée, guide le modèle pour qu'il se corrige et relance la demande.

    :param prompt: Prompt pour guider le modèle lors de la correction.
    :param client: Instance de WrapperBedrock pour interagir avec le LLM.
    :param model_id: ID du modèle Bedrock à utiliser.
    :param max_retries: Nombre maximum

    :return: Liste de sous-tâches validées ou une la tàche initiale si l'échec persiste.
    """
    attempt = 0
    messages = [ConverseMessage.make_system_message(prompt)]
    while attempt < max_retries:
        try:
            llm_response = client.converse(model_id, messages=messages)
            subtasks_data = parse_json_response(llm_response.content[0].text)
            validated_subtasks = [SubTask(**task) for task in subtasks_data]
            return validated_subtasks
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt < max_retries:
                attempt += 1
                logging.error(f"❌ Erreur lors de la validation des sous-tâches (Tentative {attempt}/{max_retries}): {e}")
                response = ConverseMessage.make_assistant_message("⚠️ Erreur détectée. Reformulez la réponse au format JSON valide.")
                messages.append(response)
                instruction = (
                    "\n\n⚠️ Erreur détectée. Reformulez la réponse au format JSON valide :\n"
                    "- Chaque sous-tâche doit inclure `id`, `description`, `dependencies` et `order`.\n"
                    "- Corrigez toute erreur de formatage et réessayez.\n"
                )
                messages.append(ConverseMessage.make_system_message(instruction))
            else:
                logging.critical("💥 Échec après plusieurs tentatives. Impossible de diviser la tâche.")
                raise e


def get_subtasks(
    client: WrapperBedrock,
    model_id: str,
    user_request: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3,
    **kwargs: dict
) -> List[SubTask]:
    """
    Divise une tâche utilisateur en sous-tâches cohérentes en utilisant WrapperBedrock.
    Si une erreur est détectée, guide le modèle pour qu'il se corrige et relance la demande.

    :param client: Instance de WrapperBedrock pour interagir avec le LLM.
    :param model_id: ID du modèle Bedrock à utiliser.
    :param user_request: Requête utilisateur à diviser en sous-tâches.
    :param few_shot_examples: Exemples pour le Few-Shot Prompting (optionnel).
    :param max_retries: Nombre maximum de tentatives en cas d'erreur.
    :param kwargs: Arguments supplémentaires à transmettre au modèle.

    :return: Liste de sous-tâches validées ou une la tàche initiale si l'échec persiste.
    """
    # Vérification de la requête utilisateur
    output = validate_prompt(user_request, client, model_id)
    if not output['requete_valide']:
        return client.converse(model_id=model_id, messages=[ConverseMessage.make_user_message(output['message'])])
        
    # Génération du prompt initial
    prompt = _generate_prompt_template(user_request=output, few_shot_examples=few_shot_examples)

    print(prompt)

    # Validation de la réponse
    subtasks = divide_task(prompt, client, "mistral.mistral-large-2407-v1:0", max_retries)

    return subtasks