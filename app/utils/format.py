from functools import wraps
from inspect import signature, cleandoc
from jinja2 import Template
from typing import Callable, Any
import re
import json

def prompt_template(func) -> Callable[..., Any]:
    """
    Marque la docstring d'une fonction comme étant un template de prompt utilisant jinja

    Exemple:

    @prompt_template
    def mon_prompt(nom) ->:
    \"\"\"
        Salut par ici {{nom}} !!!
    \"\"\"
    pass

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # signature de la fonction
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # création template avec jinja
        template = Template(cleandoc(func.__doc__))
        context = bound_args.arguments
        context['enumerate'] = enumerate
        return template.render(bound_args.arguments)

    return wrapper


def parse_json_response(response: str):
    """
    Parse une réponse en JSON, qu'il s'agisse d'un dictionnaire ou d'un tableau.

    :param response: Réponse à parser.
    :return: Réponse parsée en JSON (dict ou list).
    """
    json_pattern = r'\[.*\]|\{.*\}'  # Adjusted to capture both array and object patterns
    json_match = re.search(json_pattern, response, re.DOTALL)
    
    if json_match:
        response_text = json_match.group(0)
    else:
        response_text = response
    
    try:
        parsed_response = json.loads(response_text)
        return parsed_response
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Failed to parse response as JSON: {response_text}", response_text, 0)