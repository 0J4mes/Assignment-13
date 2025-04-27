#!/usr/bin/env python3
# mushroom_preprocessing.py
# Assignment 13 - Preprocessing Data for scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import os


def download_dataset():
    """Download the mushrooms dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    if not os.path.exists("mushrooms.data"):
        print("Downloading dataset...")
        urlretrieve(url, "mushrooms.data")
    else:
        print("Dataset already exists locally")


def preprocess_data():
    """Load and preprocess the mushrooms dataset"""
    # Download the dataset
    download_dataset()

    # Load the data
    print("\nLoading data...")
    df = pd.read_csv("mushrooms.data", header=None)

    # Column names mapping
    column_names = {
        0: "edibility",
        1: "cap_shape",
        2: "cap_surface",
        3: "cap_color",
        4: "bruises",
        5: "odor",
        6: "gill_attachment",
        7: "gill_spacing",
        8: "gill_size",
        9: "gill_color",
        10: "stalk_shape",
        11: "stalk_root",
        12: "stalk_surface_above_ring",
        13: "stalk_surface_below_ring",
        14: "stalk_color_above_ring",
        15: "stalk_color_below_ring",
        16: "veil_type",
        17: "veil_color",
        18: "ring_number",
        19: "ring_type",
        20: "spore_print_color",
        21: "population",
        22: "habitat"
    }

    # Rename columns
    df.rename(columns=column_names, inplace=True)

    # Select specific columns
    selected_columns = ["edibility", "odor", "cap_color"]
    df = df[selected_columns]

    # Value mappings
    value_mappings = {
        "edibility": {"e": 0, "p": 1},
        "odor": {
            "a": 0,  # almond
            "l": 1,  # anise
            "c": 2,  # creosote
            "y": 3,  # fishy
            "f": 4,  # foul
            "m": 5,  # musty
            "n": 6,  # none
            "p": 7,  # pungent
            "s": 8  # spicy
        },
        "cap_color": {
            "n": 0,  # brown
            "b": 1,  # buff
            "c": 2,  # cinnamon
            "g": 3,  # gray
            "r": 4,  # green
            "p": 5,  # pink
            "u": 6,  # purple
            "e": 7,  # red
            "w": 8,  # white
            "y": 9  # yellow
        }
    }

    # Convert categorical values to numeric
    print("\nConverting categorical values to numeric...")
    for column in selected_columns:
        df[column] = df[column].map(value_mappings[column])

    return df


def perform_eda(df):
    """Perform exploratory data analysis and create visualizations"""
    print("\nPerforming exploratory data analysis...")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Distribution plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.countplot(x='edibility', data=df)
    plt.title('Edibility Distribution (0=edible, 1=poisonous)')

    plt.subplot(1, 3, 2)
    sns.countplot(x='odor', data=df)
    plt.title('Odor Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.countplot(x='cap_color', data=df)
    plt.title('Cap Color Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('output/distributions.png')
    print("\nSaved distributions plot as 'output/distributions.png'")

    # Relationship plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.stripplot(x='edibility', y='odor', data=df, jitter=True, alpha=0.5)
    plt.title('Edibility vs Odor')
    plt.xlabel('Edibility (0=edible, 1=poisonous)')
    plt.ylabel('Odor Code')

    plt.subplot(1, 2, 2)
    sns.stripplot(x='edibility', y='cap_color', data=df, jitter=True, alpha=0.5)
    plt.title('Edibility vs Cap Color')
    plt.xlabel('Edibility (0=edible, 1=poisonous)')
    plt.ylabel('Cap Color Code')

    plt.tight_layout()
    plt.savefig('output/relationships.png')
    print("Saved relationship plots as 'output/relationships.png'")

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("\nData Description:")
    print(df.describe())

    print("\nCross-tabulation of Edibility and Odor:")
    print(pd.crosstab(df['edibility'], df['odor'], margins=True))

    print("\nCross-tabulation of Edibility and Cap Color:")
    print(pd.crosstab(df['edibility'], df['cap_color'], margins=True))

    print("\nPercentage by Odor:")
    print(pd.crosstab(df['odor'], df['edibility'], normalize='index').round(2))

    print("\nPercentage by Cap Color:")
    print(pd.crosstab(df['cap_color'], df['edibility'], normalize='index').round(2))


def save_processed_data(df):
    """Save the processed data to CSV"""
    df.to_csv('output/processed_mushrooms.csv', index=False)
    print("\nSaved processed data as 'output/processed_mushrooms.csv'")


def main():
    """Main function to execute the data preprocessing pipeline"""
    print("Mushroom Dataset Preprocessing")
    print("IS 362 Assignment - Preprocessing Data for scikit-learn")

    # Preprocess the data
    df = preprocess_data()

    # Perform EDA
    perform_eda(df)

    # Save processed data
    save_processed_data(df)

    # Print conclusions
    print("\nConclusions:")
    print("1. Odor is a strong predictor of edibility with clear patterns")
    print("2. Cap color shows some predictive power but is less reliable")
    print("3. For Project 4, odor should be the primary feature with cap color as secondary")
    print("\nProcessing complete. Check the 'output' directory for results.")


if __name__ == "__main__":
    main()