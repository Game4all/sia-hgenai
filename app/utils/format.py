from functools import wraps
from inspect import signature, cleandoc
from jinja2 import Template
from typing import Callable, Any

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