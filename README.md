# Intern Intelligence Platform 🚀

Built for the **Kenexai Hackathon 2k26 · CHARUSAT · Data & AI Challenge**

---

## 🔗 Live Demo

🌐 **[Click here to use the app](https://intern-dashboard-ft3knuyentakubr8vbpxnt.streamlit.app/)**

---

## 💻 Source Code

👉 https://github.com/Jaymin-2603/intern-dashboard

---

## 📌 About Project

The Intern Intelligence Platform is a comprehensive analytics dashboard and data warehouse solution designed to track, analyze, and manage intern progress and performance. It features role-based access for both minimal setup local use and robust production use.

---

## ✨ Features

* **Role-based Dashboards:** Dedicated experiences for Mentors (to oversee multiple interns, track LMS progress, activity hours) and Interns (to track personal progress).
* **Automated ETL Pipeline:** Pulls data from sources, cleans it, and loads it into a Star Schema Data Warehouse (Medallion Architecture) in PostgreSQL.
* **GenAI Data Assistant:** Features querying capabilities over the data powered by Large Language Models (via Groq API).
* **Synthetic Data Generation:** Includes tools to bootstrap the application with synthetic data for development and testing.
* **Sleek UI:** A tailored, modern Streamlit UI with custom CSS.

---

## 🛠 Tech Stack

* **Frontend/App:** Streamlit, custom CSS
* **Data Processing:** Python, Pandas, SQLAlchemy
* **Database:** PostgreSQL (Docker Compose supported)
* **AI/LLM:** Groq API

---

## 📂 Project Structure

* `app.py`: Main Streamlit application
* `warehouse_etl.py` / `daily_data_pipeline.py`: ETL scripts
* `schema_design.py`: Star schema design
* `mentor_dashboard.py` / `intern_dashboard.py`: Role-based dashboards
* `genai_service.py`: Groq API integration
* `generate_synthetic_data.py`: Synthetic data generator
* `ml_models.py`: ML utilities
* `.streamlit/` & `style.css`: UI customization

---

## ⚙️ Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jaymin-2603/intern-dashboard.git
   cd intern-dashboard
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add environment variables:**
   Create `.env`

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run app:**

   ```bash
   streamlit run app.py
   ```

---

## 🚀 Usage

* Open `http://localhost:8501`
* Choose role (Mentor / Intern)
* Explore dashboards and analytics

---

## 🌟 Live Access

👉 **Use the deployed app here:**
https://intern-dashboard-ft3knuyentakubr8vbpxnt.streamlit.app/
