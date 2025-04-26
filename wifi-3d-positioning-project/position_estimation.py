import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio

class PositionEst:
    def __init__(self):
        self.df = pd.read_excel("02-3D Position Estimation using WiFi Fingerprinting.xlsx")
        self.signal_columns = [col for col in self.df.columns if "signal" in col and "_signal" in col]
        self.X = self.df[self.signal_columns]
        self.y = self.df[["x", "y", "z"]]
        self.model = None
        self.importances = None
        self.top_indices = None
        self.y_pred = None
        self.y_test = None
            
    def preprocessing(self):
        dict_df = self.df.isnull().sum()
        len_df = len(dict_df)
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def findBestParametersforRandomForest(self, X_train, y_train):
        rforest_test = RandomForestRegressor(random_state= 42)
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250, 300]  
        }

        grid_search = GridSearchCV(estimator=rforest_test, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        return grid_search.best_params_['n_estimators']
    
    def RandomForestFit_Predict(self):
        X_train, X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators= self.findBestParametersforRandomForest(X_train, y_train), random_state= 42)
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

        euclidean_errors = np.linalg.norm(self.y_test.values - self.y_pred, axis=1)
        mean_error = np.mean(euclidean_errors)
        print("Mean Euclidean Error for Random Forest:", mean_error)

    
    def find_the_best_param_for_KNN(self,X_train, y_train):
        knn = KNeighborsRegressor()
        param_grid = {'n_neighbors': list(range(1, 31))}

        # Grid search
        grid = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)

        print("Best n_neighbors:", grid.best_params_['n_neighbors'])

        return grid.best_params_['n_neighbors']
    
    def KNNFit_Predict(self):
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        # Multioutput KNN modeli
        self.model = KNeighborsRegressor(n_neighbors=self.find_the_best_param_for_KNN(self.X_train, self.y_train))
        
        # Fit ve predict
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        # Performans ölçümü
        euclidean_errors = np.linalg.norm(self.y_test.values - self.y_pred, axis=1)
        mean_error = np.mean(euclidean_errors)
        print("Mean Euclidean Error for KNN (Without Importance):", mean_error)


    def XGBRFit_Predict(self):
        X_train, X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        xgbr = XGBRegressor()

        param_dist = {
            'n_estimators': np.arange(50, 200, 50), 
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  
            'max_depth': np.arange(3, 10, 1),  
            'subsample': [0.7, 0.8, 0.9, 1.0],  
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  
            'gamma': [0, 0.1, 0.2, 0.3]
        }


        random_search = RandomizedSearchCV(
            estimator=xgbr, 
            param_distributions=param_dist, 
            n_iter=50,  
            scoring='neg_mean_squared_error',  
            cv=3,  
            verbose=2,  
            random_state=42,  
            n_jobs=-1  
        )

        random_search.fit(X_train, y_train)


        self.model = random_search.best_estimator_
        self.y_pred = self.model.predict(X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f'Mean Square Error (MSE): {mse}')
    
    def calculate_importance(self):
        self.importances = self.model.feature_importances_
        self.top_indices = np.argsort(self.importances)[-20:]  
        try:
            self.X = self.X.iloc[:, self.top_indices]
        except:
            self.X = self.X[:, self.top_indices]

        fig, ax = plt.subplots(figsize=(12,4))
        ax.bar(range(len(self.importances)), self.importances)
        ax.set_title("Access Point (RSSI) Priority Ranking")
        st.pyplot(fig)
        fig.savefig("Importance.png", bbox_inches='tight')
        plt.close(fig)

    

    def importance_for_KNN(self):
        importances_list = []
        X_np = self.X if isinstance(self.X, np.ndarray) else self.X.values  # Orijinal X değişmesin!

        for i in range(3):  # x, y, z
            knn = KNeighborsRegressor(n_neighbors=self.model.n_neighbors)
            knn.fit(X_np, self.y.iloc[:, i])

            result = permutation_importance(
                knn,
                X_np,
                self.y.iloc[:, i],
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_squared_error'
            )
            importances_list.append(result.importances_mean)

        self.importances = np.mean(importances_list, axis=0)
        self.top_indices = np.argsort(self.importances)[-20:]

        # 🟥 self.X = X_np[:, self.top_indices] BUNU SİL!
        # Çünkü bu X'i değiştirir → train/test split'teki X ile uyumsuzluk olur!

        # Görselleştirme:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(self.importances)), self.importances)
        ax.set_title("Access Point (RSSI) Priority Ranking (KNN via Permutation Importance)")
        st.pyplot(fig)
        fig.savefig("Importance_for_KNN.png", bbox_inches='tight')
        plt.close(fig)



    def RandomForestProcess(self):
        self.RandomForestFit_Predict()
        self.calculate_importance()  
        self.visualize("RandomForestProcess")


    def KNNProcess(self):
        self.KNNFit_Predict()
        self.importance_for_KNN()
        self.visualize("KNNProcess")


    def XGBRProcess(self):
        self.XGBRFit_Predict()
        self.calculate_importance()
        self.visualize("XGBRProcess")


    def visualize(self, process_name):
        fig = go.Figure()

        # 🟢 Her durumda y_test'i NumPy array'e çeviriyoruz:
        if isinstance(self.y_test, pd.DataFrame):
            y_test_array = self.y_test.to_numpy()
        else:
            y_test_array = self.y_test

        # 🎯 Gerçek (mavi) ve tahmin (kırmızı) verileri:
        x_real, y_real, z_real = y_test_array[:, 0], y_test_array[:, 1], y_test_array[:, 2]
        x_pred, y_pred, z_pred = self.y_pred[:, 0], self.y_pred[:, 1], self.y_pred[:, 2]

        # ✅ Gerçek Değerler (Mavi)
        fig.add_trace(go.Scatter3d(
            x=x_real,
            y=y_real,
            z=z_real,
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Real'
        ))

        # ✅ Tahmin Değerleri (Kırmızı)
        fig.add_trace(go.Scatter3d(
            x=x_pred,
            y=y_pred,
            z=z_pred,
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Predict '
        ))

        fig.update_layout(
            title=f"3D Prediction Visualization - {process_name}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=600
        )

        # 💾 PNG Olarak Kayıt (plots klasörüne kaydediyoruz):
        import os
        os.makedirs("plots", exist_ok=True)
        filename = os.path.join("plots", f"{process_name}.png")
        fig.write_image(filename)

        # 🖼️ Streamlit için interaktif gösterim:
        st.plotly_chart(fig, use_container_width=True)


        

if __name__ == "__main__":
    x = PositionEst()
    x.RandomForestProcess()
    x.XGBRProcess()
    x.KNNProcess()
    