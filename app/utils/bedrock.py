import boto3
import json
import numpy as np
from pydantic import BaseModel
from typing import Literal, List


class ConverseMessageContent(BaseModel):
    """
    Contenu d'un message de l'API Bedrock Converse.
    """
    text: str


class ConverseMessage(BaseModel):
    """
    Un wrapper pour un message dans l'API bedrock Converse.
    """
    role: Literal["user", "assistant", "system"]
    content: list[ConverseMessageContent]

    @classmethod
    def make_user_message(cls, message: str):
        return cls(role="user", content=[ConverseMessageContent(text=message)])

    @classmethod
    def make_assistant_message(cls, message: str):
        return cls(role="assistant", content=[ConverseMessageContent(text=message)])
    
    @classmethod
    def make_system_message(cls, message: str):
        return cls(role="user", content=[ConverseMessageContent(text=message)])


class WrapperBedrock:
    def __init__(self, service_name='bedrock-runtime', region: str = "us-west-2"):
        """
        Wrapper pour les APIs Bedrock.

        :param service_name: Nom du service Bedrock.
        :param region: Région AWS.
        """
        self.session = boto3.Session()
        self.bedrock = self.session.client(
            service_name=service_name, region_name=region)
        pass

    def converse_raw(self,
                     model_id: str,
                     messages: List[ConverseMessage],
                     max_tokens: int = 100,
                     temperature: float = 0,
                     **kwargs: dict) -> dict:
        """
        Converse avec un modèle Bedrock.

        :param model_id: ID du modèle Bedrock.
        :param messages: Liste de messages pour la conversation.
        :param max_tokens: Nombre maximum de tokens à générer.
        :param temperature: Température pour l'échantillonnage.
        :param top_p: Seuil pour le top-p sampling.
        :param kwargs: Arguments supplémentaires.

        :return: Réponse du modèle Bedrock. (dict)
        """

        # verif des arguments
        valid_keys = ['presencePenalty', 'frequencyPenalty', 'top_k', 'top_p']
        for key in kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Argument invalide : {key}")

        # validation des parametres
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"'max_tokens' doit être un entier positif, reçu : {max_tokens}")

        if not (0 <= temperature <= 1):
            raise ValueError(
                f"'temperature' doit être entre 0 et 1, reçu : {temperature}")

        if 'presencePenalty' in kwargs and not (-2 <= kwargs['presencePenalty'] <= 2):
            raise ValueError(
                f"'presencePenalty' doit être entre -2 et 2, reçu : {kwargs['presencePenalty']}")

        if 'frequencyPenalty' in kwargs and not (-2 <= kwargs['frequencyPenalty'] <= 2):
            raise ValueError(
                f"'frequencyPenalty' doit être entre -2 et 2, reçu : {kwargs['frequencyPenalty']}")

        if 'top_k' in kwargs and not (0 <= kwargs['top_k'] <= 100):
            raise ValueError(
                f"'top_k' doit être entre 0 et 100, reçu : {kwargs['top_k']}")

        if 'top_p' in kwargs and not (0 <= kwargs['top_p'] <= 1):
            raise ValueError(
                f"'top_p' doit être entre 0 et 1, reçu : {kwargs['top_p']}")

        response = self.get_client().converse(
            modelId=model_id,
            messages=[message.model_dump() for message in messages],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        )

        # to return str response use response['output']['messages']['content']
        return response

    def converse(self,
                 model_id: str,
                 messages: List[ConverseMessage],
                 max_tokens: int = 100,
                 temperature: float = 0,
                 **kwargs: dict) -> ConverseMessage:

        return ConverseMessage.model_validate_json(json.dumps(self.converse_raw(model_id, messages, max_tokens, temperature, **kwargs)["output"]["message"]))

    def get_embedding(self, text: str, embed_model_id: str = "amazon.titan-embed-text-v2:0") -> np.ndarray:
        """
        Récupère l'embedding d'un texte avec Bedrock.

        :param text: Texte à encoder.
        :param embed_model_id: ID du modèle d'embedding.

        :return: Embedding du texte.
        """
        # Appel au modèle Bedrock pour obtenir l'embedding
        response = self.get_client().invoke_model(
            body=json.dumps({"inputText": text}),
            modelId=embed_model_id,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body['embedding']

    def get_session(self) -> boto3.Session:
        return self.session

    def get_client(self) -> boto3.client:
        return self.bedrock
