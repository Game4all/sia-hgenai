import dotenv
from app.utils.bedrock import WrapperBedrock, ConverseMessage
from app.planning.subtasks import get_subtasks

dotenv.load_dotenv()

bedrock_client = WrapperBedrock()
user_request = "Analyse moi les risques d'inondation dans la ville de Marseille, et l'adaptation de la ville face à ces risques"
messages = [ConverseMessage.make_user_message(user_request)]
subtasks = get_subtasks(bedrock_client, "mistral.mistral-large-2407-v1:0", user_request)

print(subtasks)