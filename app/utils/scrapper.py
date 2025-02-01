import requests
import os
from googlesearch import search
from difflib import SequenceMatcher
import fitz
import utils.bedrock as bedrock
from duckduckgo_search import DDGS
import json
from googleapiclient.discovery import build
# import app.utils.format as format

class scrapper:
    def __init__(self, num_results=1,pipe=None,googlecred=None, googleidengin=None):
        self.num_results = num_results
        self.pipe = pipe
        self.cred = googlecred
        self.idengin = googleidengin
    
    def get_insee_code(self,city_name):
        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()
        def find_insee_code(data, target="Paris"):
            best_match = None
            highest_score = 0.0

            for entry in data:
                score = similarity(target.lower(), entry["nom"].lower())
                if score > highest_score:
                    highest_score = score
                    best_match = entry["code"]

            return best_match

        url = f"https://geo.api.gouv.fr/communes?nom={city_name}&fields=code"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data:
                return find_insee_code(data,city_name)
            else:
                return "City not found"
        else:
            return "Error fetching data"
    
    def pdf_to_text(self,pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)  # Extract text only
        return text
    
    def word_count(self,text):
        words = text.split()  # Split text by whitespace
        return len(words)
    
    def truncate_string(self,text, x):
        words = text.split()
        return ' '.join(words[:x])
    
    def repport_geoRisk(self,city,v=False):
        code_insee = self.get_insee_code(city)
        if(v):
            print(code_insee)
        response = requests.get(f"https://georisques.gouv.fr/api/v1/rapport_pdf?code_insee={code_insee}", stream=True)
        return {"url":f"https://georisques.gouv.fr/api/v1/rapport_pdf?code_insee={code_insee}","pdf":response}
    
    def check_revelence(self,subject,pathpdf,logs=False,v=False):
        if(v):
            print(f"checking {pathpdf} revelence...")
        text = self.pdf_to_text(pathpdf)
        if(self.word_count(text) < 1000):
            if(v):
                print("not enough words, bailout")
            return False
        sample = self.truncate_string(text,10)
        messages = [bedrock.ConverseMessage.make_user_message( f"tu vas recevoir un echantillons de text et tu devra me dire seulement \"Oui\" ou \"Non\" si le text est du non sens tel que par exemple <wsefwsefgvygf \n\n voici l'echantillons: {sample}")]
        bedrockapi = bedrock.WrapperBedrock()
        outputs = bedrockapi.converse(self.pipe,messages,4,0)
        if(v):print(outputs.content[0].text)
        if("Non" in outputs.content[0].text):
            if(v):
                print("gibbriche")
            return False
        if(v):
            print(f"{outputs.content[0].text}")
        text = self.truncate_string(text,2000)
        if(logs):
            with open(f"{pathpdf}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        messages = [bedrock.ConverseMessage.make_user_message(f"tu vas recevoir un text et tu devra me dire seulement \"Oui\" ou \"Non\" si le sujet parle bien de {subject} et non pas par exemple d'outils\n\n voici le text: {text}")]
        outputs = bedrockapi.converse(self.pipe,messages,4,0)
        if(v):print(outputs.content[0].text)
        if("Non" in outputs.content[0].text):
            if(v):
                print("found not revelent")
            return False
        elif("Oui" in outputs.content[0].text):
            if(v):
                print("found revelent")
            return True

    def find_doc(self,region:str,documents:list,v=False,logs=False) -> list:
        files= []
        for document in documents:
            query = f"{region} {document} \"{document}\" filetype:pdf"
            # results = list(search(query, num_results=self.num_results*2+5))
            service = build(
            "customsearch", "v1", developerKey=self.cred
            )
            res = (
                service.cse()
                .list(
                    q=query,
                    cx=self.idengin,
                )
                .execute()
            )
            results = [link["link"] for link in res["items"]]
            counter_result=0
            for result in results:
                if result.endswith(".pdf"):
                    try:
                        response = requests.get(result, stream=True)
                        files.append({"url":result,"pdf":response})
                        os.makedirs("Temp", exist_ok=True)
                        filename = "Temp/"+os.path.basename(result)
                        with open(filename, "wb") as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        stat = self.check_revelence(document,filename,v=v,logs=logs)
                        if(stat):
                            print(f"PDF downloaded as {filename}")
                            counter_result+=1
                        else:
                            os.remove(filename)
                        if(counter_result >= self.num_results):
                            break
                    except Exception as e:
                       print(f"error: {e}")
                       os.remove(filename)
            return files

# import torch
# from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# scrap = scrapper(3,pipe=pipe)
# scrap.find_doc("picardi","SRADDET",True,True)
# scrap.repport_geoRisk("Chateau-thierry",True)
