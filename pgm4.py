with open("trainingdata.csv", "w") as f:
    f.write("""Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes""")

import csv

def find_s(filename):
    # 1. Read the dataset
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    
    # Remove the header row
    data = data[1:] 
    
    # 2. Initialize the hypothesis with the first positive instance
    hypothesis = None
    for row in data:
        if row[-1].lower() == "yes":
            hypothesis = row[:-1]
            break
            
    # If no positive instances are found, return a message or handle it
    if hypothesis is None:
        return "No positive training examples found."

    # 3. Update the hypothesis by generalizing based on other positive instances
    for row in data:
        if row[-1].lower() == "yes":
            for i in range(len(hypothesis)):
                if hypothesis[i] != row[i]:
                    hypothesis[i] = "?"
                    
    return hypothesis

# Execute the function
result = find_s("trainingdata.csv")
print("Final Hypothesis:")
print(result)