# AgriAssist 🌱

AgriAssist is a smart farming assistant built with Streamlit that helps farmers choose the best crops and fertilizers for their land using data-driven recommendations.

## Features

- **Crop Recommendation:** Suggests the most suitable crops based on soil nutrients and climate conditions.
- **Fertilizer Recommendation:** Recommends fertilizers and organic alternatives based on nutrient deficiencies for selected crops.
- **Data Visualization:** Visualizes nutrient levels and crop suitability.
- **Sample Data:** Includes sample datasets for crops and fertilizers.

## How It Works

1. **Crop Recommendation:**  
   Enter your soil's N, P, K values, temperature, humidity, pH, and rainfall. The app predicts the best crops using a machine learning model.
2. **Fertilizer Recommendation:**  
   Select your crop and input current soil N, P, K values. The app suggests chemical and organic fertilizers to address deficiencies.

## Getting Started

### Prerequisites

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies.

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/AgriAssist.git
    cd AgriAssist/Agri
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the app:
    ```sh
    streamlit run app.py
    ```

## Project Structure

```
Agri/
├── app.py
├── crop_data.csv
├── crop_model.pkl
├── fertilizer.csv
└── requirements.txt
```

- `app.py`: Main Streamlit application.
- `crop_data.csv`: Crop dataset for model training.
- `fertilizer.csv`: Ideal NPK values for crops.
- `crop_model.pkl`: Saved machine learning model (auto-generated).
- `requirements.txt`: Python dependencies.


## Future Enhancements

- Yield prediction
- Pest/disease alerts
- Market price forecasts
- Multi-language support
- **Embedded System Integration:** We plan to introduce an embedded system to automatically detect soil qualities and feed the data directly into the application for crop and fertilizer recommendations.
- **Full-stack Development:** In the future, we aim to develop a proper backend and frontend architecture for improved scalability, user management, and deployment.
