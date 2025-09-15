# Superstore Sales Data Analysis

This repository contains a comprehensive data analysis project on the Superstore sales dataset. The project includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Visualizations using Matplotlib, Seaborn, and Plotly
- Interactive dashboards with Streamlit
- Geographical analysis with Plotly and Chart Studio

## Project Structure

- `sales_analyis.ipynb` — Main Jupyter notebook for data analysis and visualization
- `Superstore.csv` — Primary dataset
- `USZipsWithLatLon_20231227.csv` — US zip code geolocation data
- `streamlit_superstore.py` — Streamlit app for interactive dashboard
- `config.py` — (Not tracked by git) Stores API keys and credentials
- `requirements.txt` — List of required Python packages
- `.gitignore` — Files and folders to be ignored by git

## Getting Started

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd superstore_sales
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up credentials:**
   - Create a `config.py` file in the project directory with the following content:
     ```python
     CHART_STUDIO_USERNAME = 'your_username'
     CHART_STUDIO_API_KEY = 'your_api_key'
     ```
   - Do not share or commit your API keys.

4. **Run the Jupyter notebook:**
   - Open `sales_analyis.ipynb` in Jupyter Notebook or JupyterLab.

5. **Run the Streamlit app:**
   ```
   streamlit run streamlit_superstore.py
   ```
   or
   ```
   python -m streamlit run streamlit_superstore.py
   ```

## Features
- Data cleaning and feature engineering
- Sales, profit, and delivery time analysis by segment, state, and shipping mode
- Brand and product analysis
- Interactive visualizations and dashboards
- Geographical mapping of sales and profit

## Notes
- The `config.py` file is excluded from version control for security.
- API keys should never be committed to the repository.
- For any issues or questions, please open an issue on GitHub.

## License
This project is for educational and demonstration purposes.
