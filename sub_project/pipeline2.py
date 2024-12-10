import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from components.parser.parser import get_dataset, get_split

if __name__ == '__main__':
    # Your initial setup and pipeline definition
    dataset_id = "HIGGS"
    split_ratio = 0.3
    #Assuming get_dataset and get_split are predefined
    X, y = get_dataset(dataset_id)
    X_train, X_test, y_train, y_test = get_split(X, y, split_ratio)


    user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC', RandomForestClassifier())])

    # Correct the order of split data assignment
    X_train, X_test, y_train, y_test = get_split(X, y, split_ratio)

    # Training and initial evaluation
    start_time = time.time()
    user2_pipe.fit(X_train, y_train)
    ac = user2_pipe.score(X_test, y_test)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"{ac},{training_time},{ac / training_time}")

    # Extract feature importances
    classifier = user2_pipe.named_steps['RFC']
    feature_importances = classifier.feature_importances_

    # Iteratively retrain and evaluate with top N features
    for i in range(1, 29):
        top_n_indices = np.argsort(feature_importances)[-i:]

        X_train_filtered = X_train[:, top_n_indices]
        X_test_filtered = X_test[:, top_n_indices]

        # Optionally, reset the pipeline here if needed
        user2_pipe = Pipeline([('scaler', StandardScaler()), ('RFC', RandomForestClassifier())])

        start_time = time.time()
        user2_pipe.fit(X_train_filtered, y_train)
        ac = user2_pipe.score(X_test_filtered, y_test)
        end_time = time.time()

        training_time = end_time - start_time
        print(f"{i},{ac},{training_time},{ac / training_time}")