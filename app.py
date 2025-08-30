import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)



def main():
    st.title('Breast Cancer Prediction Web App')
    st.sidebar.title("Web app")

    #upload file from user and predict the result
    upload_file=st.file_uploader("Upload a CSV file",type=["csv"])

    def load_data():
        data=pd.read_csv("C:/Users/abrup/Desktop/newproject/breast cancer.csv")
        data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
        return data

    
    @st.cache_data (persist=True)
    def split(df):
        y= df.iloc[:, 0]
        x= df.iloc[:, 1:]
        le = LabelEncoder()
        y = le.fit_transform(df["diagnosis"])
        x = df.drop(columns=["diagnosis"])
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
        return x_train,x_test,y_train,y_test,le, scaler
    

    def plot_metrics(metrics_list):
        if 'Confusion Matrix'  in metrics_list:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)


            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            y_prob=model.predict_proba(x_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve - KNN")
            ax.legend(loc="lower right")

            # 6. Display plot in Streamlit
            st.pyplot(fig)
        
        

 
    df=load_data()
    #class_names=['B','M']
    x_train,x_test,y_train,y_test,le,scaler=split(df)
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("k-Nearest Neighbors (KNN)","Linear Support Vector Machines (SVM)"))



    if classifier == "k-Nearest Neighbors (KNN)":
        st.sidebar.subheader("Model Hyperparameters")
        n_neighbors=int(st.sidebar.radio('n_neighbors',('3', '5', '7', '9', '11'),key='n_neighbors'))
        weight=st.sidebar.radio('weight',('uniform','distance'),key='weight')
        p = st.sidebar.radio('Distance Metric (p)', (1, 2), key='p')


        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve'))



        if st.sidebar.button("Classify",key='classify'):
            st.subheader("k-Nearest Neighbors (KNN)")
            model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weight,p=p)
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)

            st.session_state["model"] = model  
            
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred) * 100
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.text("Classification Report:\n"+classification_report(y_test,y_pred ))
            

            plot_metrics(metrics)


        if st.sidebar.button("Predict",key="predict"):
            if upload_file is not None:
                df=pd.read_csv(upload_file)

                st.write("📂 Uploaded Data:", df)

                try:
                    if "model" in st.session_state:    
                     model = st.session_state["model"]
                     predictions = model.predict(df)  # numeric (0/1)
                     predictions_labels = le.inverse_transform(predictions)  # back to B/M
                     df["Predictions"] = predictions_labels
                     st.write("✅ Predictions:", df)
   
                    else:
                        st.warning("⚠️ Please train the model first using 'Classify'")
                except Exception as e:
                    st.error(f"Error: {e}")


    if classifier == 'Linear Support Vector Machines (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Linear Support Vector Machines (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
           

            st.session_state["model"] = model  
            
               
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred) * 100
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.text("Classification Report:\n"+classification_report(y_test,y_pred ))
            
            plot_metrics(metrics)


        if st.sidebar.button("Predict",key="predict"):
            if upload_file is not None:
                df=pd.read_csv(upload_file)

                st.write("📂 Uploaded Data:", df)

                try:
                    if "model" in st.session_state:
                        model = st.session_state["model"]
                        predictions = model.predict(df)
                        predictions_labels = le.inverse_transform(predictions)  # back to B/M
                        df["Predictions"] = predictions_labels
                        st.write("✅ Predictions:", df)
                    else:
                        st.warning("⚠️ Please train the model first using 'Classify'")
                except Exception as e:
                    st.error(f"Error: {e}")
 

        
        







    if st.sidebar.checkbox('Show raw data',False):
        st.subheader('Breast Cancer Wisconsin (Diagnostic) Data Set')
        st.write(df)
        st.markdown("The Wisconsin Breast Cancer Data refers to datasets created at the University of Wisconsin to classify breast tumors as benign or malignant using features extracted from fine-needle aspirate (FNA) biopsies")


        

  





























if __name__ == '__main__':
    main()