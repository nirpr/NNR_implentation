# NNR implentation
This project is an implementation of an NNR (Nearest Neighbor Rule) classifier. The classifier utilizes a JSON file for configuring the CSV sheets that contain the data. The JSON file should have the following format:
```
{
  "data_file_train": "data/<enter csv name here>_train.csv",
  "data_file_validation": "data/<enter csv name here>_validation.csv",
  "data_file_test": "data/<enter csv name here>_test.csv"
}  
```

In this implementation, the following ideas are utilized:

Data Initialization: The data is initialized by scaling the training, validation, and test data. Additionally, the target labels for the training and validation data are prepared.

Distance Calculation: Euclidean distances between each training instance and each validation instance are computed. Similarly, Euclidean distances between each training instance and each test instance are calculated.

Radius Range Determination: The range of radii is determined based on the distances between the training and validation instances. The minimum and maximum distances are found, and a range of radii is created by evenly spacing values within this range.

Optimal Radius Selection: The optimal radius is determined from the range of radii using the validation data and labels. The goal is to find the radius that yields the best performance.

Prediction: Using the determined radius and the distances between the training and test instances, the classifier makes predictions for the test instances.
