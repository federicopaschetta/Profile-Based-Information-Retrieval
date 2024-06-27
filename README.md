# Information Retrieval System for Profile-Based Document Retrieval

This repository contains scripts, data, and a detailed report on a project aimed at developing a Profile-Based Information Retrieval System. This system leverages advanced text processing and machine learning techniques to deliver personalized document recommendations to users.

## Project Overview

The project utilizes several methodologies to analyze and categorize text data, enabling the system to match documents with user profiles effectively. The system's core lies in its ability to adapt to the unique preferences of each user, enhancing user engagement and information consumption efficiency.

## Folder Contents

### Scripts
- `script_final.ipynb`: Jupyter notebook containing the Python code for all text processing, modeling, and analysis.
- `script_final.py`: Python script version of the notebook for deployment or more efficient execution environments.

### Data
- `bbc-text.csv`: Dataset used for training and evaluating the retrieval system, containing categorized news articles.
- `words.txt`: Contains key vocabulary terms used in text processing and model training to enhance retrieval accuracy.

### Report
- `Information_Retrieval_1th_Assignment_Report.pdf`: Comprehensive report detailing the project's methodologies, implementation, and key findings.

## Installation and Setup

Before running the scripts, ensure that Python 3.8+ is installed along with the necessary libraries:

bash
pip install -r requirements.txt


## Running the Scripts

To execute the system, you can run the Python script directly or use the Jupyter notebook:

bash
python script_final.py


Or,

bash
jupyter notebook script_final.ipynb


## Key Components

### Text Processing
- Cleaning and preprocessing text data to improve model performance.
- Utilizing techniques such as tokenization, lemmatization, and TF-IDF to prepare the data for analysis.

### Methodologies
- **Latent Dirichlet Allocation (LDA)** for topic modeling to categorize documents.
- **TF-IDF Similarity Calculation** to match documents with user profiles based on content relevance.

### User Profile Simulation
- Simulating user interaction to demonstrate how the system recommends documents based on individual profiles.

## Results

Results are discussed in the included report, highlighting the effectiveness of different methodologies in matching documents to user profiles and vice versa.

## Authors

- Federico Paschetta
- Cecilia Peccolo
- Nicola Maria D’Angelo

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Special thanks to Universidad Politécnica de Madrid and Prof. Antonio Canale for guidance and resources throughout the project.

## Contact

For any queries, please contact:
- federico.paschetta@alumnos.upm.es
- cecilia.peccolo@alumnos.upm.es
- nicolamaria.dangelo@alumnos.upm.es
