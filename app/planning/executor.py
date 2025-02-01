from .subtasks import SubTask
from typing import Callable
from functools import wraps
from ..utils.bedrock import WrapperBedrock


def agent_task(name):
    """
    Marque une fonction comme étant une tâche éxecutable par un agent.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return name, func
        return wrapper
    return decorator


class AgentExecutor:
    def __init__(self, wrapper: WrapperBedrock):
        # les sorties des différentes tâches
        self.outputs: dict[str, dict] = {}
        # les tâches disponibles dans l'agent
        self.tasks: dict[str, Callable] = {}
        self.bedrock = wrapper
        pass

    def register_task(self, task_callable: Callable) -> None:
        assert (hasattr(task_callable, "__wrapped__"))
        task_name, og_func = task_callable()
        self.tasks[task_name] = og_func
        pass

    def execute_task(self, task: SubTask) -> None:
        """
            Execute une seule tâche et stocke le résultat dans les sorties si une sortie est configurée.
        """
        output = self.tasks[task.task](self, task.args)
        if task.out is not None and output is not None:
            self.outputs[task.out] = output
        pass

    def get_inputs(self, name: str) -> dict:
        return self.outputs[name]

    def execute_tasks(self, tasks: list[SubTask]):
        for task in tasks:
            self.execute_task(task)
            yield task.description
