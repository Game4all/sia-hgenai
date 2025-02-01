import dotenv
from app.utils.bedrock import WrapperBedrock, ConverseMessage
from app.planning.subtasks import get_subtasks

dotenv.load_dotenv()

bedrock_client = WrapperBedrock()
user_request = "Fais moi une fiche synthèse des risques d'innondation et de sécheresse pour la ville de Marseille et de Paris"
messages = [ConverseMessage.make_user_message(user_request)]

validate_output = get_subtasks(bedrock_client, validation_model_id="mistral.mistral-7b-instruct-v0:2", planning_model_id="mistral.mistral-large-2402-v1:0", user_request=user_request)

print(validate_output)