import dotenv
from app.utils.bedrock import WrapperBedrock, ConverseMessage
from app.planning.subtasks import get_subtasks, validate_user_request

dotenv.load_dotenv()

bedrock_client = WrapperBedrock()
user_request = "Fais moi une fiche synthèse des risques d'innondation pour la ville de Marseille"
messages = [ConverseMessage.make_user_message(user_request)]

validate_output = get_subtasks(bedrock_client, validation_model_id="mistral.mistral-large-2402-v1:0", planning_model_id="mistral.mistral-large-2402-v1:0", user_request=user_request)

print(validate_output)