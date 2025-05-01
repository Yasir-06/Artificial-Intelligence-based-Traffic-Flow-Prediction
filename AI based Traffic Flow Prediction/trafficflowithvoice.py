import pandas as pd
import numpy as np
import pyttsx3  # Text-to-speech library
import time
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Initialize the speech engine
engine = pyttsx3.init()

# Function for speech output
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to validate if input is a valid integer
def is_valid_number(input_str):
    try:
        # Try to convert the input to an integer
        num = int(input_str)
        return num
    except ValueError:
        return None

# Function to create a plot for a given junction and hours
def plot_junction_data(df, junction_input, look_back_input):
    # Filter the data for the selected junction and the required look-back period
    df_junction = df[df['Junction'] == junction_input]
    
    # We need to select the last 'look_back_input' hours of data for the plot
    df_junction_last_hours = df_junction.tail(look_back_input)

    # Plotting the vehicle data for the last 'look_back_input' hours
    plt.figure(figsize=(10, 6))
    plt.bar(df_junction_last_hours['DateTime'], df_junction_last_hours['Vehicles'], color='b', label=f'Junction {junction_input}')
    plt.title(f'Vehicles at Junction {junction_input} Over the Last {look_back_input} Hours')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Function to create a plot for the total vehicle count over time for all junctions
def plot_total_vehicles(df):
    plt.figure(figsize=(10, 6))
    junctions = df['Junction'].unique()
    
    for junction in junctions:
        df_junction = df[df['Junction'] == junction]
        df_junction_grouped = df_junction.groupby('DateTime')['Vehicles'].sum()
        plt.plot(df_junction_grouped.index, df_junction_grouped.values, label=f'Junction {junction}')
    
    plt.title('Total Vehicles Over Time for All Junctions')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to create a bar plot for total vehicles for each junction
def plot_all_junctions(df):
    plt.figure(figsize=(10, 6))
    junctions = df['Junction'].unique()
    
    total_vehicles = []
    for junction in junctions:
        total_vehicles.append(df[df['Junction'] == junction]['Vehicles'].sum())
    
    plt.bar(junctions, total_vehicles, color='r')
    plt.title('Total Vehicles by Junction')
    plt.xlabel('Junction')
    plt.ylabel('Total Number of Vehicles')
    plt.tight_layout()
    plt.show()

# Function to process the data and train the model
def run_traffic_prediction():
    try:
        # Load the dataset
        file_path = 'D:/traffic flow/traffic.csv'  # Update this path to your local file
        df = pd.read_csv(file_path)

        # Convert DateTime to datetime format
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

        # Extract time-related features
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['month'] = df['DateTime'].dt.month
        df['year'] = df['DateTime'].dt.year

        # Plot total vehicle data for all junctions and vehicles over time
        plot_all_junctions(df)
        plot_total_vehicles(df)

        # Get user inputs
        junction_input = is_valid_number(junction_var.get())  # Validate junction input
        if junction_input is None or junction_input not in [1, 2, 3, 4]:
            speak("Please enter a valid junction number between 1 and 4.")
            return  # Exit function if input is invalid

        look_back_input = is_valid_number(lookback_var.get())  # Validate look-back input
        if look_back_input is None or look_back_input <= 0:
            speak("Please enter a valid number for look-back hours.")
            return  # Exit function if input is invalid

        # Plot the data for the selected junction and hours
        plot_junction_data(df, junction_input, look_back_input)

        # Prepare data for the selected junction
        df.set_index('DateTime', inplace=True)
        df_junction = df[df['Junction'] == junction_input]['Vehicles']

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_junction_scaled = scaler.fit_transform(df_junction.values.reshape(-1, 1))

        # Train/test split
        train_size = int(len(df_junction_scaled) * 0.8)
        train, test = df_junction_scaled[:train_size], df_junction_scaled[train_size:]

        # Create dataset function
        def create_dataset(dataset, look_back=1):
            X, y = [], []
            for i in range(len(dataset) - look_back):
                X.append(dataset[i:(i + look_back), 0])
                y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train, look_back_input)
        X_test, y_test = create_dataset(test, look_back_input)

        # Reshape inputs for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # LSTM Model
        model = Sequential()
        model.add(LSTM(25, return_sequences=False, input_shape=(look_back_input, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stop])

        # Make predictions for the next time period
        last_hours = df_junction_scaled[-look_back_input:].reshape(1, look_back_input, 1)
        predicted_vehicles = model.predict(last_hours)
        predicted_vehicles = scaler.inverse_transform(predicted_vehicles)

        # Show prediction result
        result_text = f"The predicted number of vehicles for the next hour is: {predicted_vehicles[0][0]:.2f}"
        speak(result_text)  # Announce the result using voice
        messagebox.showinfo("Prediction Result", result_text)  # Show the result in a message box

    except Exception as e:
        # If an error occurs, just continue without breaking the flow.
        print(f"An error occurred: {e}")

# Create the Tkinter window
root = tk.Tk()
root.title("Traffic Volume Prediction")

# Welcome the user and explain the input process
speak("Welcome to the Traffic Volume Prediction System.")
time.sleep(2)  # Pause for 2 seconds

speak("Please enter the junction number and the look-back hours.")

# Label and Entry for Junction selection
junction_label = tk.Label(root, text="Select Junction (1, 2, 3, or 4):")
junction_label.pack()

junction_var = tk.StringVar()
junction_entry = tk.Entry(root, textvariable=junction_var)
junction_entry.pack()

# Label and Entry for Look-back period
lookback_label = tk.Label(root, text="Enter Look-back period (e.g., 24 for 24 hours):")
lookback_label.pack()

lookback_var = tk.StringVar()
lookback_entry = tk.Entry(root, textvariable=lookback_var)
lookback_entry.pack()

# Run button
run_button = tk.Button(root, text="Run Prediction", command=run_traffic_prediction)
run_button.pack()

# Start the Tkinter event loop
root.mainloop()
