import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class StudentPerformanceClassifier:
    def __init__(self):
        # Load the dataset
        self.df = pd.read_csv(r"C:\Users\K_har\Downloads\linear regression\StudentsPerformance.csv")
        
        # Prepare the data for classification
        self.prepare_data()
        
        # Train both regression and classification models
        self.train_models()
        
        # Create GUI
        self.create_gui()
    
    def prepare_data(self):
        """
        Prepare data for both regression and classification
        Steps:
        1. Perform linear regression to get predicted scores
        2. Create binary classification target based on performance
        """
        # Prepare features for regression
        X = self.df[['reading score', 'writing score']]
        y_regression = self.df['math score']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train linear regression model
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Create binary classification target
        # Let's classify students as 'high performers' or 'low performers'
        # We'll use the median math score as the threshold
        median_score = self.df['math score'].median()
        self.df['performance_class'] = (self.df['math score'] > median_score).astype(int)
        
        # Prepare features for classification
        X_class = self.df[['reading score', 'writing score']]
        y_class = self.df['performance_class']
        
        # Split classification data
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        
        # Store classification data
        self.X_train_class = X_train_class
        self.X_test_class = X_test_class
        self.y_train_class = y_train_class
        self.y_test_class = y_test_class
    
    def train_models(self):
        """
        Train logistic regression model for classification
        """
        # Scale features for classification
        X_train_scaled = self.scaler.transform(self.X_train_class)
        X_test_scaled = self.scaler.transform(self.X_test_class)
        
        # Train logistic regression
        self.logistic_model = LogisticRegression(random_state=42)
        self.logistic_model.fit(X_train_scaled, self.y_train_class)
        
        # Evaluate the model
        y_pred = self.logistic_model.predict(X_test_scaled)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(self.y_test_class, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test_class, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def create_gui(self):
        """
        Create GUI for predicting both regression and classification
        """
        # Create main window
        self.root = tk.Tk()
        self.root.title("Student Performance Predictor")
        self.root.geometry("500x400")
        
        # Input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=20)
        
        # Reading Score Input
        tk.Label(input_frame, text="Reading Score:").grid(row=0, column=0)
        self.reading_entry = tk.Entry(input_frame)
        self.reading_entry.grid(row=0, column=1)
        
        # Writing Score Input
        tk.Label(input_frame, text="Writing Score:").grid(row=1, column=0)
        self.writing_entry = tk.Entry(input_frame)
        self.writing_entry.grid(row=1, column=1)
        
        # Predict Button
        predict_button = tk.Button(
            self.root, 
            text="Predict Performance", 
            command=self.predict
        )
        predict_button.pack(pady=20)
        
        # Result Labels
        self.regression_label = tk.Label(self.root, text="")
        self.regression_label.pack()
        
        self.classification_label = tk.Label(self.root, text="")
        self.classification_label.pack()
        
        self.root.mainloop()
    
    def predict(self):
        """
        Predict both regression and classification results
        """
        try:
            # Get input values
            reading_score = float(self.reading_entry.get())
            writing_score = float(self.writing_entry.get())
            
            # Validate input
            if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
                messagebox.showerror("Invalid Input", "Scores must be between 0 and 100")
                return
            
            # Prepare input for prediction
            input_data = np.array([[reading_score, writing_score]])
            input_scaled = self.scaler.transform(input_data)
            
            # Regression Prediction (Predicted Math Score)
            predicted_math_score = self.regression_model.predict(input_scaled)[0]
            self.regression_label.config(
                text=f"Predicted Math Score: {predicted_math_score:.2f}"
            )
            
            # Classification Prediction (Performance Class)
            performance_pred = self.logistic_model.predict(input_scaled)[0]
            performance_prob = self.logistic_model.predict_proba(input_scaled)[0]
            
            # Interpret classification result
            performance_text = (
                "High Performer" if performance_pred == 1 else "Low Performer"
            )
            probability = performance_prob[performance_pred]
            
            self.classification_label.config(
                text=f"Performance Classification: {performance_text} "
                f"(Probability: {probability:.2%})"
            )
        
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric scores")

# Run the application
if __name__ == "__main__":
    app = StudentPerformanceClassifier()