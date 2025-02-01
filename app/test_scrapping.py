from utils.scrapper import scrapper
# import streamlit as st

if __name__ == '__main__':
    # st.title("Test")
    scrap = scrapper(5,"mistral.mistral-7b-instruct-v0:2","AIzaSyDX4QfKTMx84FbslTVyoHm_yYSMpyl5HEI","2711c54ca8c8e4c35")
    scrap.find_doc("picardi","SRADDET",True,True)