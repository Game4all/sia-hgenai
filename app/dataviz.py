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


@prompt_template
def generate_visualization_template(risks: list[str], lieu: str) -> str:
    """
    Tu es un expert en visualisation de données environnementales. Ta mission est de rechercher et de générer des visualisations adaptées aux risques environnementaux identifiés ci-dessous pour le lieu : {{lieu}}.

    Instructions :
    1. Recherche des données statistiques pertinentes pour chaque risque en lien avec {{lieu}} à partir de sources fiables (bases de données spécialisées, rapports officiels, etc.).
    2. Choisis le type de visualisation le plus pertinent en fonction de la nature des données (carte géographique, histogramme, graphique en ligne, etc.).
    3. Crée une visualisation qui met en évidence les éléments suivants :
        - La fréquence et l'intensité des risques.
        - La répartition géographique des risques si applicable.
        - Les tendances temporelles ou saisonnières des risques si pertinentes.

    Liste des risques :
    {% for risk in risks -%}
    - {{risk}}
    {% endfor %}

    Assure-toi que la visualisation soit claire, informative, et facilement compréhensible pour le public cible.
    Réponds en fournissant un schéma JSON structuré comme suit :
    {
        "visualization_type": "<type de visualisation>",
        "description": "Description succincte de la visualisation et des données utilisées.",
        "visualization_code": {{ Code Python (ou autre langage) nécessaire pour générer la visualisation. | json }},
        "insights": "Principaux enseignements et conclusions tirés de la visualisation."
    }

    Exemples :

    1. Carte montrant la répartition des risques climatiques en Picardie :
    {
        "visualization_type": "carte",
        "description": "Carte illustrant la répartition des risques climatiques en Picardie, basée sur les données de l'Observatoire national sur les effets du réchauffement climatique.",
        "visualization_code": "import folium\nm = folium.Map(location=[49.8941, 2.2957], zoom_start=7)\nfolium.Marker([49.8941, 2.2957], popup='Amiens').add_to(m)\nm",
        "insights": "Les zones côtières de la Picardie sont plus exposées aux risques climatiques, notamment les inondations et les tempêtes."
    }

    2. Histogramme de la fréquence des risques climatiques en France :
    {
        "visualization_type": "histogramme",
        "description": "Histogramme montrant la fréquence des principaux risques climatiques en France, basé sur les données de Météo-France.",
        "visualization_code": "import matplotlib.pyplot as plt\nrisques = ['Inondations', 'Sécheresse', 'Canicule', 'Tempêtes']\noccurrences = [25, 30, 15, 20]\nplt.bar(risques, occurrences)\nplt.xlabel('Risques climatiques')\nplt.ylabel('Nombre d\'occurrences annuelles')\nplt.title('Fréquence des risques climatiques en France')\nplt.show()",
        "insights": "Les sécheresses et les inondations sont les risques les plus fréquents en France."
    }

    3. Graphique en ligne des tendances des inondations en France :
    {
        "visualization_type": "graphique en ligne",
        "description": "Graphique illustrant l'évolution des risques d'inondations en France sur la dernière décennie, basé sur les données de la DREAL.",
        "visualization_code": "import matplotlib.pyplot as plt\nannees = list(range(2010, 2020))\ninondations = [12, 15, 20, 18, 25, 22, 28, 30, 27, 35]\nplt.plot(annees, inondations, marker='o')\nplt.xlabel('Année')\nplt.ylabel('Nombre d\'inondations')\nplt.title('Évolution des inondations en France (2010-2019)')\nplt.grid(True)\nplt.show()",
        "insights": "Les inondations en France ont montré une tendance à la hausse au cours de la dernière décennie, avec des pics significatifs en 2015 et 2019."
    }
    """
    pass

def generate_visualization(bedrock: WrapperBedrock, risks: list[str], lieu:str, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0") -> str:
    resp = parse_json_response(bedrock.converse(model_id=model_id, messages=[
        ConverseMessage.make_user_message(generate_visualization_template(risks, lieu))], max_tokens=8192).content[0].text)
    return resp