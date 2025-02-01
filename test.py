import dotenv
from app.utils.bedrock import WrapperBedrock, ConverseMessage
from app.planning.subtasks import get_subtasks, validate_user_request

dotenv.load_dotenv()

bedrock_client = WrapperBedrock()
user_request = "Comment le groupement de communes du Grand Périgueux s’adapte-t-il aux risques d’inondation?"
messages = [ConverseMessage.make_user_message(user_request)]
validate_output = validate_user_request(user_request, bedrock_client, "mistral.mistral-large-2402-v1:0")

print(validate_output)