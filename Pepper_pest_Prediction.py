{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "G2i0Ol7Hr7Yv"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "M1IKAIw4uoMT"
      },
      "outputs": [],
      "source": [
        "# Load the dataset\n",
        "file_path = 'E:\\Pepper disease predictions with weather data\\CSV_Pepper_diseases_data_combined.csv'\n",
        "df = pd.read_csv(file_path)\n",
        "\n",
        "# Drop rows with missing values (optional: or fillna)\n",
        "df.dropna(inplace=True)\n",
        "\n",
        "# Define features (previous week's weather data)\n",
        "features = ['MaxTemp', 'MinTemp', 'RH1', 'RH2', 'SunShine', 'Rainfall']\n",
        "\n",
        "# Create lag features for weather (shift by 1 week)\n",
        "for col in features:\n",
        "    df[f'{col}_prev'] = df[col].shift(1)\n",
        "\n",
        "# Drop the current week columns and keep only previous week features\n",
        "df = df.dropna()  # Drop the first row after shift\n",
        "X = df[[f'{col}_prev' for col in features]]\n",
        "\n",
        "# Define target (Binary classification for disease occurrence)\n",
        "# Assuming ARP is the main target for demonstration\n",
        "y = df['ARP'].apply(lambda x: 1 if x > 0 else 0)\n",
        "\n",
        "# Split data\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Initialize models\n",
        "models = {\n",
        "    'Random Forest': RandomForestClassifier(random_state=42),\n",
        "    'Support Vector Machine': SVC(),\n",
        "    'Logistic Regression': LogisticRegression(max_iter=1000)\n",
        "}\n",
        "\n",
        "# Train and evaluate models\n",
        "results = []\n",
        "for name, model in models.items():\n",
        "    model.fit(X_train, y_train)\n",
        "    y_pred = model.predict(X_test)\n",
        "    results.append({\n",
        "        'Model': name,\n",
        "        'Accuracy': accuracy_score(y_test, y_pred),\n",
        "        'Precision': precision_score(y_test, y_pred),\n",
        "        'Recall': recall_score(y_test, y_pred),\n",
        "        'F1 Score': f1_score(y_test, y_pred)\n",
        "    })\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FJDVCML0FU8Y",
        "outputId": "ce6461f3-e46b-4941-94e5-3b691753f080"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model Evaluation Results:\n",
            "                    Model  Accuracy  Precision  Recall  F1 Score\n",
            "0           Random Forest    0.4375        0.4   0.250  0.307692\n",
            "1  Support Vector Machine    0.6250        1.0   0.250  0.400000\n",
            "2     Logistic Regression    0.5000        0.5   0.375  0.428571\n"
          ]
        }
      ],
      "source": [
        "# Convert results to DataFrame for display\n",
        "results_df = pd.DataFrame(results)\n",
        "print(\"Model Evaluation Results:\")\n",
        "print(results_df)\n",
        "\n",
        "#import ace_tools as tools; tools.display_dataframe_to_user(name=\"ML Model Evaluation Results\", dataframe=results_df)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8M6mKg2YSXrr",
        "outputId": "cb90f55e-dff3-4372-efd2-e89ac4caaf16"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['E:/Pepper disease predictions with weather data/black_pepper_rf_model.pkl']"
            ]
          },
          "execution_count": 5,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "import joblib\n",
        "\n",
        "# Train your best model (assuming Random Forest was the best)\n",
        "best_model = RandomForestClassifier(random_state=42)\n",
        "best_model.fit(X_train, y_train)\n",
        "\n",
        "# Save it\n",
        "joblib.dump(best_model, 'E:/Pepper disease predictions with weather data/black_pepper_rf_model.pkl')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c5hK51fhTW8S",
        "outputId": "e2827eaf-6fb9-469a-94e0-3f1531c006d6"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['/content/drive/MyDrive/black_pepper_rf_model.pkl']"
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "joblib.dump(model, '/content/drive/MyDrive/black_pepper_rf_model.pkl')\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
