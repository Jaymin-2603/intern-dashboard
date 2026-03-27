# Intern Intelligence Platform

Built for the **Kenexai Hackathon 2k26 · CHARUSAT · Data & AI Challenge**. 

The Intern Intelligence Platform is a comprehensive analytics dashboard and data warehouse solution designed to track, analyze, and manage intern progress and performance. It features role-based access for both minimal setup local use and robust production use.

## Features

- **Role-based Dashboards:** Dedicated experiences for Mentors (to oversee multiple interns, track LMS progress, activity hours) and Interns (to track personal progress).
- **Automated ETL Pipeline:** Pulls data from sources, cleans it, and loads it into a Star Schema Data Warehouse (Medallion Architecture) in PostgreSQL.
- **GenAI Data Assistant:** Features querying capabilities over the data powered by Large Language Models (via Groq API).
- **Synthetic Data Generation:** Includes tools to bootstrap the application with synthetic data for development and testing.
- **Sleek UI:** A tailored, modern Streamlit UI with custom CSS.

## Tech Stack

- **Frontend/App:** [Streamlit](https://streamlit.io/), custom CSS
- **Data Processing:** Python, Pandas, SQLAlchemy
- **Database:** PostgreSQL (with Docker Compose setup for local development)
- **AI/LLM:** Groq API integration

## Project Structure

- `app.py` / `app1.py`: The main Streamlit application entry points. 
- `warehouse_etl.py` / `daily_data_pipeline.py`: ETL scripts to process incoming data files and load them into the warehouse.
- `schema_design.py`: Defines the Star Schema architecture.
- `mentor_dashboard.py` / `intern_dashboard.py`: Logic and UI for the respective user roles.
- `genai_service.py`: Integration with the Groq API for the AI data assistant.
- `generate_synthetic_data.py`: Utilities to create mock datasets for initial setup and testing.
- `ml_models.py`: Machine learning utilities.
- `.streamlit/` & `style.css`: UI customization and styling.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd kenexai
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `streamlit`, `pandas`, `sqlalchemy`, `psycopg2-binary`, `groq`, etc. installed)*

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys/database credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Database Setup:**
   - Ensure you have a running PostgreSQL instance locally or via Docker.
   - Run the initial ETL or Data Pipeline script to populate the local data warehouse (e.g., `python daily_data_pipeline.py` or through the dashboard fallback).

6. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage

- Navigate to `http://localhost:8501`.
- Select your role (Mentor or Intern) and name to enter the associated dashboard.
- The platform will fetch analytics directly from the PostgreSQL star schema, falling back to flat-file ETL if the database isn't immediately available.
