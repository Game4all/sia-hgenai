from utils.scrapper import scrapper
# import streamlit as st

if __name__ == '__main__':
    # st.title("Test")
    scrap = scrapper(3,"mistral.mistral-7b-instruct-v0:2","AIzaSyD0EjP3Z9YChWnGpnKhDAc1Ithm2Npdpdk","2711c54ca8c8e4c35")
    doc = scrap.find_doc("Picardie",["SRADDET", "SDAGE"],True,True)
    print([docs["url"] for docs in doc])