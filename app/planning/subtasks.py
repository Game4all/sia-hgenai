from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Optional, Any
from ..utils.format import prompt_template, parse_json_response
import json
from ..utils.bedrock import WrapperBedrock, ConverseMessage
import logging


class SubTask(BaseModel):
    task: str = Field(..., description="Type de la tâche")
    description: str = Field(...,
                             description="Description textuelle de la sous-tâche")
    args: Optional[Dict[str, Any]] = Field(
        {}, description="Arguments supplémentaires pour la sous-tâche")
    out: str | None = Field(
        None, description="Nom de la variable pour la sortie de la sous-tâche")


@prompt_template
def validate_user_request_template(user_request: str, few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Vérifie que la requête utilisateur a pour sujet l'analyse ou l'adaptaion des collectivites francasies face aux  des risques environnementaux. \n
    Vérifie que le requête utilisateur parle des risques physique ou de transition environnementaux en général ou alors contient au moins un des risques ou catégories de risques suivants :\n"
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
    Retourne un JSON sous la forme suivante si la requête est valide : \n"

    {\n"
        "requete_valide": true,\n"
        "message": "<requête reformulée>",\n"
        "risques": ["<risque1 ou catégorie1>", "<risque2 ou catégorie2>", ...],\n"
        "lieux": ["<lieu1>", "<lieu2>"],\n"
        "niv_admin": "<niveau_administratif>",\n"
    }\n"
    Si la requête n'est pas valide, retourne un JSON sous la forme suivante : \n"
    {\n"
        'requete_valide': false,\n"
        'message': '<explication du problème>'\n"
    }\n\n"

    {% if few_shot_examples %}
    Voici quelques exemples de requêtes avec leurs résultats de validation : \n
    {% for example in few_shot_examples %}
    Requête utilisateur : {{ example["requete"] }}\n
    Résultat :
    {
        "requete_valide": {{ example["requete_valide"] }},
        "message": "{{ example["message"] }}",
        "risques": "{{ example["risques"] }}",
        "lieux": "{{ example["lieux"] }}",
        "niv_admin": "{{ example["niv_admin"] }}"
    }\n
    {% endfor %}
    {% endif %}

    Voici la requête à traiter : \n
    **Requête utilisateur :** {{ user_request }} \n
    """

    pass


def validate_user_request(user_request: str,
                          client: WrapperBedrock,
                          model_id: str) -> Dict[str, Any]:
    """
    Valide une requête utilisateur pour s'assurer qu'elle contient les informations nécessaires.

    :param prompt: Prompt pour guider le modèle lors de la correction.
    :param client: Instance de WrapperBedrock pour interagir avec le LLM.
    :param model_id: ID du modèle Bedrock à utiliser.

    :return: Dictionnaire contenant les informations validées ou un message d'erreur.
    """
    few_shot_examples = [
        {
            "requete": "Quels sont les plans d'adaptation de la region Provence-Alpes-Côte d'Azur face aux vague de chaleur et ala sécheresse ?",
            "requete_valide": "true",
            "message": "Quels sont les plans d'adaptation de la région Provence-Alpes-Côte d'Azur face aux vagues de chaleur et à la sécheresse ?",
            "risques": ["Vague de chaleur", "Sécheresse"],
            "lieux": ["Provence-Alpes-Côte d'Azur"],
            "niv_admin": "région"
        },
        {
            "requete": "Comment la France gère-t-elle le stress hydrique ?",
            "requete_valide": "false",
            "message": "La requête mentionne un risque environnemental (stress hydrique), mais ne précise pas de localisation en France. Veuillez spécifier un lieu comme une commune, un département ou une région. Par exemple : 'Comment la région Occitanie gère-t-elle le stress hydrique ?'"
        },
        {
            "requete": "Quels sont les projets d'urbanisation prévus à Lyon ?",
            "requete_valide": "false",
            "message": "La requête mentionne un lieu en France (Lyon, commune) mais ne fait référence à aucun risque environnemental. Veuillez inclure un risque environnemental pertinent. Par exemple : 'Quels sont les projets d'urbanisation prévus à Lyon pour faire face aux risques d'inondation ?'"
        },
        {
            "requete": "Fais moi une synthèse des risques environnementaux à Paris.",
            "requete_valide": "true",
            "message": "Fais-moi une synthèse des risques environnementaux à Paris.",
            "risques": [],
            "lieux": ["Paris"],
            "niv_admin": "commune"
        },
         {
            "requete": "Rédige un rapport sur les risques dans la région de la Bretagne.",
            "requete_valide": "true",
            "message": "Rédige un rapport sur les risques dans la région de la Bretagne.",
            "risques": [],
            "lieux": ["Bretagne"],
            "niv_admin": "région"
        },
        {
            "requete": "Quelles sont les mesures prises contre les inondtaions à Paris, Bordeaux et Lyon ?",
            "requete_valide": "true",
            "message": "Quelles sont les mesures prises contre les inondations à Paris, Bordeaux et Lyon ?",
            "risques": ["Inondation"],
            "lieux": ["Paris", "Bordeaux", "Lyon"],
            "niv_admin": "commune"
        },
        {
            "requete": "Comment la métropole de Lyon s'adapte-t-elle à la pollution de l'air ?",
            "requete_valide": "true",
            "message": "Comment la métropole de Lyon s'adapte-t-elle à la pollution de l'air ?",
            "risques": ["Pollution de l’air"],
            "lieux": ["Métropole de Lyon"],
            "niv_admin": "groupement de communes"
        }
    ]
    instruction = validate_user_request_template(
        user_request=user_request, few_shot_examples=few_shot_examples)
    messages = [ConverseMessage.make_user_message(instruction)]
    response = client.converse(
        model_id=model_id, messages=messages, max_tokens=512)
    response_text = response.content[0].text

    try:
        parsed_response = parse_json_response(response_text)
        print(parsed_response)
        if parsed_response and isinstance(parsed_response, dict) and {"requete_valide", "message", "risques", "lieux", "niv_admin"} <= parsed_response.keys():
            return parsed_response
        else:
            raise Exception
    except Exception:
        return {"requete_valide": False, "message": parsed_response["message"]}
@prompt_template
def subtask_prompt_template(user_request: str, few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Vous êtes un planificateur de tâches avancé au sein de SFIL, une banque d'investissement spécialisée dans la recherche et l'analyse des documents relatifs à l'adaptation des collectivités aux risques climatiques. Votre tâche consiste à analyser la requête utilisateur et à diviser le processus en sous-tâches clairement définies. Ces sous-tâches doivent être organisées en fonction de l'ordre d'exécution et des dépendances. Elles doivent couvrir les étapes suivantes, qui sont génériques et peuvent varier légèrement selon le cas spécifique :

    1. **SEARCH_DOCS** : Identification et collecte des documents et données pertinentes relatives aux risques climatiques et à l’adaptation des collectivités.
    2. **ANALYZE_DOCS** : Analyse des documents pour comprendre l'impact des risques climatiques sur les collectivités et évaluer les mesures d’adaptation existantes.
    3. **DATAVIZ** : Création de visualisations (graphiques, cartes, etc.) permettant de mieux comprendre et communiquer les résultats de l’analyse.
    4. **SYNTHESIZE** : Élaboration d’un rapport détaillé, résumant les résultats de l’analyse et proposant des recommandations adaptées pour les collectivités.

    La structure du format JSON pour chaque sous-tâche est la suivante :
    {
        "task": "<type de la sous-tâche>",
        "description": "<description textuelle de la sous-tâche>",
        "args": {
            "<argument1>": "<valeur1>",
            "<argument2>": "<valeur2>",
            ...
        },
        "out": "<identifiant unique pour la sortie de la tâche>"
    }

    Chaque valeur d'attribut doit être encadrée de guillemets doubles. Très important pour le parser JSON.


    {%- set doc_mapping = {
        "commune": {
            "docs": ["DICRIM", "Plan Local d'Urbanisme", "Plan Communal de Sauvegarde"],
            "sources": ["Geoportail", "Georisques", "Gaspar"]
        },
        "groupement": {
            "docs": ["Plan Local d'Urbanisme Intercommunal", "Plan Intercommunal De Sauvegarde", "Plan d'Action de Prévention des Inondations"],
            "sources": ["Geoportail Urbanisme", "Gaspar", "Ademe"]
        },
        "departement": {
            "docs": ["Dossier départemental des risques majeurs", "Plan Départemental de Protection des Forêts Contre les Incendies"],
            "sources": ["Gaspar", "Préfecture"]
        },
        "région": {
            "docs": ["SRADDET", "SDAGE"],
            "sources": ["Ademe", "Régions de France"]
        }
    } %}

    [
        {
            "task": "SEARCH_DOCS",
            "description": "Recherche de documents pour {{ user_request["lieux"] | join(', ') }}",
            "args": {
                "docs": {{ doc_mapping[user_request['niv_admin']]['docs'] | tojson }},
                "sources": {{ doc_mapping[user_request['niv_admin']]['sources'] | tojson }},
                "lieux" : {{ user_request["lieux"] | join(', ') }}
            },
            "out": "search_output"
        },
        {
            "task": "ANALYZE_DOCS",
            "description": "Analyse des documents pour {{ user_request['lieux'] | join(', ') }}",
            "args": {
                "in": "search_output",
                "risques": {{ user_request["risques"] | tojson }} if user_request["risques"] else []
            },
            "out": "analyze_output"
        },
        {
            "task": "DATAVIZ",
            "description": "Création de visualisations pour {{ user_request['lieux'] | join(', ') }}",
            "args": {
                "lieux": {{ user_request["lieux"] | join(', ') }}
                "risques": "risques": {{ user_request["risques"] | tojson }} if user_request["risques"] else []
            },
            "out": "dataviz_output"
        },
        {
            "task": "SYNTHESIZE",
            "description": "Synthèse des données de risques pour {{ user_request['lieux'] | join(', ') }}",
            "args": {
                "in": "analyze_output",
                "dataviz": "dataviz_output"
            },
            "out": "synthesize_output"
        }
    ]

    {% if few_shot_examples %}
    Voici quelques exemples de tâches correctement divisées : \n
    {% for example in few_shot_examples %}
    Requête : {{ example["requete"] }}\n
    Résultat: {{ example["subtasks"] | tojson }}\n

    {% endfor %}
    {% endif %}

    Voici la requête à traiter : \n

    **Requête utilisateur :**
    {{ user_request["message"] }} \n

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
            llm_response = client.converse(
                model_id, messages=messages,  max_tokens=1024)
            subtasks_data = parse_json_response(llm_response.content[0].text)
            validated_subtasks = [SubTask(**task) for task in subtasks_data]
            return validated_subtasks
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt < max_retries:
                attempt += 1
                logging.error(
                    f"❌ Erreur lors de la validation des sous-tâches (Tentative {attempt}/{max_retries}): {e}")
                response = ConverseMessage.make_assistant_message(
                    "⚠️ Erreur détectée. Reformulez la réponse au format JSON valide.")
                messages.append(response)
                instruction = (
                    "\n\n⚠️ Erreur détectée. Reformulez la réponse au format JSON valide :\n"
                    "La structure du format JSON pour chaque sous-tâche est la suivante :\n"
                    "{\n"
                    "    'task': '<type de la sous-tâche>',\n"
                    "    'description': '<description textuelle de la sous-tâche>',\n"
                    "    'args': {\n"
                    "        '<argument1>': '<valeur1>',\n"
                    "        '<argument2>': '<valeur2>',\n"
                    "        ...\n"
                    "    },\n"
                    "    'out': '<identifiant unique pour la sortie de la tâche>'\n"
                    "}\n"
                )
                messages.append(
                    ConverseMessage.make_system_message(instruction))
            else:
                logging.critical(
                    "💥 Échec après plusieurs tentatives. Impossible de diviser la tâche.")
                raise e


def plan_actions(
    client: WrapperBedrock,
    validation_model_id: str,
    planning_model_id: str,
    user_request: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3,
) -> List[SubTask]:
    """
    Divise une tâche utilisateur en sous-tâches cohérentes en utilisant WrapperBedrock.
    Si une erreur est détectée, guide le modèle pour qu'il se corrige et relance la demande.

    :param client: Instance de WrapperBedrock pour interagir avec le LLM.
    :param validation_model_id: ID du modèle Bedrock pour la validation de la requête utilisateur.
    :param planning_model_id: ID du modèle Bedrock pour la division de la tâche en sous-tâches.
    :param user_request: Requête utilisateur à diviser en sous-tâches.
    :param few_shot_examples: Exemples pour le Few-Shot Prompting (optionnel).
    :param max_retries: Nombre maximum de tentatives en cas d'erreur.
    :param kwargs: Arguments supplémentaires à transmettre au modèle.

    :return: Liste de sous-tâches validées ou une la tàche initiale si l'échec persiste.
    """
    # Vérification de la requête utilisateur
    output = validate_user_request(user_request, client, validation_model_id)
    if not output['requete_valide']:
        return {"error": output["message"]}

    # Génération du prompt initial
    few_shot_examples = [
        {
            "requete": "Fais une synthèse des risques d'inondation pour la ville de Paris.",
            "subtasks": [
                            {
                                "task": "SEARCH_DOCS",
                                "description": "Recherche de documents sur les inondations à Paris",
                                "args": {
                                "docs": ["DICRIM", "Plan Local d'Urbanisme", "Plan Communal de Sauvegarde"],
                                "sources": ["Geoportail", "Georisques", "Gaspar"],
                                "lieux": "Paris",
                                },
                                "out": "search_output"
                            },
                            {
                                "task": "ANALYZE_DOCS",
                                "description": "Analyse des documents pour extraire les informations sur les inondations à Paris",
                                "args": {
                                "in": "search_output",
                                "risques": ["Inondation"]
                                },
                                "out": "analyze_output"
                            },
                            {
                                "task": "DATAVIZ",
                                "description": "Création de visualisations pour illustrer les risques d'inondation à Paris",
                                "args": {
                                "lieux": "Paris"
                                },
                                "out": "dataviz_output"
                            },
                            {
                                "task": "SYNTHESIZE",
                                "description": "Synthèse des données et formulation des recommandations sur les inondations à Paris",
                                "args": {
                                "in": "analyze_output",
                                "dataviz": "dataviz_output"
                                },
                                "out": "synthesize_output"
                            }
                        ]
        },
        {
            "requete": "Analyse les risques de sécheresse et de vague de chaleur dans la région Provence-Alpes-Côte d'Azur.",
            "subtasks": [
                            {
                                "task": "SEARCH_DOCS",
                                "description": "Recherche de documents concernant la sécheresse et les vagues de chaleur en région Provence-Alpes-Côte d'Azur",
                                "args": {
                                "docs": ["SRADDET", "SDAGE"],
                                "sources": ["Ademe", "Régions de France"],
                                "lieux": "Provence-Alpes-Côte d'Azur",
                                },
                                "out": "search_output"
                            },
                            {
                                "task": "ANALYZE_DOCS",
                                "description": "Analyse des documents pour identifier les impacts de la sécheresse et des vagues de chaleur dans la région",
                                "args": {
                                    "in": "search_output",
                                    "risques": ["Sécheresse", "Vague de chaleur"]
                                },
                                "out": "analyze_output"
                            },
                            {
                                "task": "SYNTHESIZE",
                                "description": "Synthèse des données et recommandations pour l'adaptation face à la sécheresse et aux vagues de chaleur",
                                "args": {
                                    "in": "analyze_output"
                                },
                                "out": "synthesize_output"
                            }
                        ]
        },
        {
            "requete": "Prépare un rapport sur la pollution de l'air dans le département de l'Isère.",
            "subtasks": [
                        {
                            "task": "SEARCH_DOCS",
                            "description": "Recherche de documents sur la pollution de l'air dans le département de l'Isère",
                            "args": {
                            "docs": ["Dossier départemental des risques majeurs", "Plan Départemental de Protection des Forêts Contre les Incendies"],
                                "sources": ["Gaspar", "Préfecture"],
                                "lieux": "Isère",
                            },
                            "out": "search_output"
                        },
                        {
                            "task": "ANALYZE_DOCS",
                            "description": "Analyse des documents pour identifier les sources et conséquences de la pollution de l'air dans l'Isère",
                            "args": {
                                "in": "search_output",
                                "risques": ["Pollution de l’air"]
                            },
                            "out": "analyze_output"
                        },
                        {
                            "task": "DATAVIZ",
                            "description": "Création de visualisations pour illustrer les données sur la pollution de l'air",
                            "args": {
                                "in": "Isère",
                                "risques": ["Pollution de l’air"]
                            },
                            "out": "dataviz_output"
                        },
                        {
                            "task": "SYNTHESIZE",
                            "description": "Synthèse des informations et élaboration d'un rapport sur la pollution de l'air dans l'Isère",
                            "args": {
                                "in": "analyze_output",
                                "dataviz": "dataviz_output"
                            },
                            "out": "synthesize_output"
                        }
                        ]   
        },
        {
            "requete": "Réalise une étude des risques de feux de forêt pour le groupement de communes de l'Essonne.",
            "subtasks": [
                        {
                            "task": "SEARCH_DOCS",
                            "description": "Recherche de documents sur les feux de forêt pour le groupement de communes de l'Essonne",
                            "args": {
                            "docs": ["Plan Local d'Urbanisme Intercommunal", "Plan Intercommunal De Sauvegarde", "Plan d'Action de Prévention des Inondations"],
                                "sources": ["Geoportail Urbanisme", "Gaspar", "Ademe"],
                                "lieux": "Essonne"
                            },
                            "out": "search_output"
                        },
                        {
                            "task": "ANALYZE_DOCS",
                            "description": "Analyse des documents pour évaluer l'impact et les risques de feux de forêt dans l'Essonne",
                            "args": {
                                "in": "search_output",
                                "risques": ["Feu de forêt"]
                            },
                            "out": "analyze_output"
                        },
                        {
                            "task": "DATAVIZ",
                            "description": "Création de visualisations pour représenter les risques de feux de forêt dans l'Essonne",
                            "args": {
                                "lieux": "Essonne",
                                "risques": ["Feu de forêt"]
                            },
                            "out": "dataviz_output"
                        },
                        {
                            "task": "SYNTHESIZE",
                            "description": "Synthèse des données et formulation de recommandations pour prévenir les feux de forêt",
                            "args": {
                                "in": "analyze_output",
                                "dataviz": "dataviz_output"
                            },
                            "out": "synthesize_output"
                        }
                        ]
        },
        {
            "requete": "Élabore un rapport complet incluant des visualisations sur l'impact des risques environnementaux dans la ville de Lyon.",
            "subtasks": [
                        {
                            "task": "SEARCH_DOCS",
                            "description": "Recherche de documents sur les risques environnementaux à Lyon",
                            "args": {
                            "docs": ["DICRIM", "Plan Local d'Urbanisme", "Plan Communal de Sauvegarde"],
                            "sources": ["Geoportail", "Georisques", "Gaspar"],
                            "lieux": "Lyon",
                            },
                            "out": "search_output"
                        },
                        {
                            "task": "ANALYZE_DOCS",
                            "description": "Analyse des documents pour identifier les causes et impacts des risques environnementaux à Lyon",
                            "args": {
                            "in": "search_output",
                            "risques": "[]"
                            },
                            "out": "analyze_output"
                        },
                        {
                            "task": "DATAVIZ",
                            "description": "Création de visualisations pour illustrer les données sur les risques environnementaux à Lyon",
                            "args": {
                                "lieux": "Lyon",
                                "risques": "[]"
                            },
                            "out": "dataviz_output"
                        },
                        {
                            "task": "SYNTHESIZE",
                            "description": "Synthèse des résultats d'analyse et intégration des visualisations dans un rapport complet sur les risques environnementaux à Lyon",
                            "args": {
                                "in": "dataviz_output",
                                "dataviz": "dataviz_output"
                            },
                            "out": "synthesize_output"
                        }
                        ]
        }
    ]
    subtask_prompt = subtask_prompt_template(
        user_request=output, few_shot_examples=few_shot_examples)

    # Validation de la réponse
    subtasks = divide_task(subtask_prompt, client,
                           planning_model_id, max_retries)

    return {"tasks": subtasks}