import streamlit as st
import pandas as pd
import numpy as np
from position_estimation import PositionEst
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.instance = PositionEst()
    
    def display_menu(self):
        st.title("📌 3D Position Estimation using WiFi Fingerprinting")

        st.markdown("""
        ## 🛰️ Project Overview  
        This project estimates **3D positions** based on WiFi fingerprinting signal strengths using machine learning models:  
        - 🌲 **Random Forest Regressor**
        - 📍 **K-Nearest Neighbors (KNN)**
        - 🚀 **XGBoost Regressor (XGBR)**  

        The system predicts the **x, y, z** coordinates using RSSI signal data, and provides model evaluation through 3D visualizations and feature importance analysis.
        """)


    def main(self):
        st.sidebar.title("Menu")
        page = st.sidebar.radio(
            "Please Select a Model",
            ["Menu", " Random Forest Process", 
             " XGBR Process ", 
             " KNN Process "]
        )

        if page == "Menu":
            self.display_menu()
        elif page == " Random Forest Process":
            st.title(" Random Forest Process")
            self.instance.RandomForestProcess()
        elif page == " XGBR Process ":
            st.title(" XGBR Process ")
            self.instance.XGBRProcess()
        elif page == " KNN Process ":
            st.title(" KNN Process ")
            self.instance.KNNProcess()

if __name__ == "__main__":
    x = Main()
    x.main()
