from utils.scrapper import scrapper
# import streamlit as st

if __name__ == '__main__':
    # st.title("Test")
    scrap = scrapper(3,"mistral.mistral-7b-instruct-v0:2","AIzaSyDX4QfKTMx84FbslTVyoHm_yYSMpyl5HEI","2711c54ca8c8e4c35")
    doc = scrap.find_doc("Paris",["DICRIM"],True,True)
    print([docs["url"] for docs in doc])