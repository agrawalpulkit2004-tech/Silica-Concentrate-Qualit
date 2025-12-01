from src.inference import predict_one

example_input = {
    "% Silica Feed": 16.95,
    "Ore Pulp Density": 1.74,
    "Starch Flow": 3200,
    "Amina Flow": 520,
    "Air Flow Column 03": 250,
    "pH": 10.2,
    "hour": 3,
    "dayofweek": 2
}

prediction = predict_one(example_input)
print(f"Predicted % Silica Concentrate: {prediction:.4f}")
