import pandas as pd
import json
import random
import time
import os
import streamlit as st

# ─────────────────────────────────────────────
#  LANGCHAIN & GROQ SERVICE
#  Uses ChatGroq if GROQ_API_KEY is found in st.secrets or os.environ.
#  Otherwise falls back to mock logic.
#
#  FIX: langchain_groq and langchain_core are imported lazily inside
#  _get_llm() so a missing/unresolved package never crashes the app
#  at startup. The rest of the dashboard works fine in mock mode.
# ─────────────────────────────────────────────

def _get_llm():
    """Helper to initialize the Groq LLM if API key is available."""
    # ── Lazy imports: only attempted when this function is called ──
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate  # noqa: F401 – imported here so callers can use it
    except ImportError:
        return None  # Package not available → fall through to mock mode

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            if 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not api_key:
        st.error("Missing GROQ_API_KEY! Please add it to your .env file or Streamlit Secrets.")
        st.stop()

    if api_key:
        try:
            return ChatGroq(
                api_key=api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.1
            )
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {e}")
            return None
    return None


def _get_prompt_template():
    """Lazy-load ChatPromptTemplate so import errors never crash the module."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate
    except ImportError:
        return None


def _simulate_typing_delay():
    """Simulate API latency for realism when in mock mode"""
    time.sleep(random.uniform(0.5, 1.2))


def query_dataframe(df: pd.DataFrame, query: str) -> str:
    """
    Uses Langchain Groq to answer natural language questions about the dataframe.
    Falls back to mock keyword matching if no API key is set.
    """
    llm = _get_llm()
    ChatPromptTemplate = _get_prompt_template()

    if llm and ChatPromptTemplate:
        try:
            summary = {
                "columns": list(df.columns),
                "total_rows": len(df),
                "sample_data": df.sample(min(15, len(df))).to_dict(orient="records") if not df.empty else []
            }
            if 'Course Name' in df.columns:
                summary["unique_courses"] = list(df['Course Name'].unique())
            if 'Intern_Name' in df.columns:
                summary["unique_interns_count"] = int(df['Intern_Name'].nunique())
            if 'Progress_Numeric' in df.columns:
                summary["avg_progress"] = float(df['Progress_Numeric'].mean())
            if 'Intern_Name' in df.columns and 'Progress_Numeric' in df.columns:
                summary["top_3_interns"] = df.groupby('Intern_Name')['Progress_Numeric'].mean().sort_values(ascending=False).head(3).to_dict()
                summary["bottom_3_interns"] = df.groupby('Intern_Name')['Progress_Numeric'].mean().sort_values().head(3).to_dict()

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI Data Assistant analyzing an intern cohort performance dataset. Use the provided JSON data summary to answer the user's question accurately. If you don't know the exact answer from the summary, state what you can infer. Output in markdown format without preambles."),
                ("user", "Data Summary: {summary}\n\nQuestion: {query}")
            ])
            chain = prompt | llm
            return chain.invoke({
                "summary": json.dumps(summary, indent=2),
                "query": query
            }).content
        except Exception as e:
            return f"⚠️ **Error executing query:** {e}\n\n*Falling back to mock mode...*"

    # --- MOCK FALLBACK ---
    _simulate_typing_delay()
    query_lower = query.lower()

    if "how many interns" in query_lower or "total" in query_lower:
        return f"Based on the current data, there are **{len(df['Intern_ID'].unique())}** unique interns in this cohort."

    if "average progress" in query_lower or "avg progress" in query_lower:
        if 'Progress_Numeric' in df.columns:
            avg_p = df['Progress_Numeric'].mean()
            return f"The average progress across the cohort is **{avg_p:.1f}%**."

    if "risk" in query_lower or "at risk" in query_lower:
        return ("To find at-risk interns, please refer to the **Risk Intelligence** tab "
                "which calculates real-time dropout probabilities using our Random Forest classifier.")

    if "worst" in query_lower or "lagging" in query_lower:
        if 'Progress_Numeric' in df.columns and 'Intern_Name' in df.columns:
            worst = df.groupby('Intern_Name')['Progress_Numeric'].mean().sort_values().head(3)
            res = "The interns currently lagging the furthest behind in average progress are:\n"
            for name, val in worst.items():
                res += f"- **{name}**: {val:.0f}%\n"
            return res

    if "top" in query_lower or "best" in query_lower:
        if 'Progress_Numeric' in df.columns and 'Intern_Name' in df.columns:
            best = df.groupby('Intern_Name')['Progress_Numeric'].mean().sort_values(ascending=False).head(3)
            res = "The top performing interns by average progress are:\n"
            for name, val in best.items():
                res += f"- **{name}**: {val:.0f}%\n"
            return res

    return ("I am a GenAI data assistant. I need a valid `GROQ_API_KEY` to execute actual Data Q&A queries against the DataFrame. "
            "Please provide an API key, or try asking about 'average progress' or 'top interns' to see a mock response.")


def generate_intervention_strategy(intern_name: str, risk_level: str, metrics: dict) -> str:
    """
    Simulates an LLM generating a mentorship action plan for an at-risk intern.
    """
    llm = _get_llm()
    ChatPromptTemplate = _get_prompt_template()

    prog = metrics.get('Avg Progress', 'N/A')
    hrs = metrics.get('Hours Logged', 'N/A')

    if llm and ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical mentor advising other mentors. Generate a concise, actionable 3-step intervention strategy for an intern based on their metrics. Output valid markdown without any preambles."),
            ("user", "Intern Name: {intern_name}\nRisk Level: {risk_level}\nAverage Progress: {prog}\nHours Logged: {hrs}\n\nPlease generate a customized mentorship action plan.")
        ])
        chain = prompt | llm
        try:
            return chain.invoke({
                "intern_name": intern_name,
                "risk_level": risk_level,
                "prog": prog,
                "hrs": hrs
            }).content
        except Exception as e:
            return f"⚠️ **Error generating strategy:** {e}"

    # --- MOCK FALLBACK ---
    _simulate_typing_delay()
    if risk_level == "High":
        return (f"**🚨 Urgent Action Plan for {intern_name}**\n\n"
                f"**Context:** {intern_name} is currently flagged as High Risk (Progress: {prog}, Hours: {hrs}).\n\n"
                "*(Provide a GROQ_API_KEY for dynamic generative strategies)*\n\n"
                "**Mock Recommended Steps:**\n"
                "1. **Immediate 1-on-1:** Schedule a 15-minute sync within the next 24 hours.\n"
                "2. **Micro-Goals:** Break down their next pending course into three smaller, easily achievable tasks.\n"
                "3. **Peer Pairing:** Consider pairing them with a 'Steady Worker' persona for joint study sessions this week.")
    elif risk_level == "Medium":
        return (f"**⚡ Nudge Plan for {intern_name}**\n\n"
                f"**Context:** {intern_name} is showing early signs of disengagement (Progress: {prog}, Hours: {hrs}).\n\n"
                "*(Provide a GROQ_API_KEY for dynamic generative strategies)*\n\n"
                "**Mock Recommended Steps:**\n"
                "1. **Asynchronous Check-in:** Send a quick Teams/Slack message asking if they need clarification.\n"
                "2. **Highlight Quick Wins:** Point them towards knowledge checks they can easily pass.\n"
                "3. **Monitor:** Review their daily hours again in 72 hours.")
    else:
        return (f"**💡 Engagement Strategy for {intern_name}**\n\n"
                f"{intern_name} is performing well (Progress: {prog}, Hours: {hrs}).\n\n"
                "**Recommendation:** Congratulate them on their steady progress. Suggest they help review code/assignments for peers.")


def get_study_assistant_response(query: str, intern_name: str, course_context: str = "") -> str:
    """
    Answers an Intern's query using ChatGroq.
    """
    llm = _get_llm()
    ChatPromptTemplate = _get_prompt_template()

    context_str = f" Context: The intern is currently studying {course_context}." if course_context else ""

    if llm and ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly, encouraging AI Study Assistant for technical interns learning Python, SQL, and Data Engineering. Provide concise, helpful code snippets or explanations. Never solve entire assignments for them, just guide them. The intern's name is {intern_name}. {context_str}"),
            ("user", "{query}")
        ])
        chain = prompt | llm
        try:
            return chain.invoke({
                "intern_name": intern_name,
                "context_str": context_str,
                "query": query
            }).content
        except Exception as e:
            return f"⚠️ **Error fetching response:** {e}"

    # --- MOCK FALLBACK ---
    _simulate_typing_delay()
    query_lower = query.lower()

    if "stuck" in query_lower or "help" in query_lower:
        return (f"Hang in there, {intern_name}! It's completely normal to get stuck. "
                "I recommend checking the 'Documentation/Resources' section of the module. "
                "*(Provide a GROQ_API_KEY for generative RAG answers)*")

    if "python" in query_lower:
        return ("**Python Tip:** Make sure you are using list comprehensions where possible for cleaner code! "
                "Example: `squares = [x**2 for x in range(10)]`. Do you have a specific Python error you are trying to solve?")

    if "sql" in query_lower or "database" in query_lower:
        return ("**SQL Tip:** When joining tables, remember that `INNER JOIN` only returns matching rows, "
                "while `LEFT JOIN` keeps all rows from your primary table. Want me to draft a query for you?")

    if "motivation" in query_lower or "tired" in query_lower:
        motivational_quotes = [
            "\u201cThe expert in anything was once a beginner.\u201d \u2014 Helen Hayes",
            "\u201cIt always seems impossible until it\u2019s done.\u201d \u2014 Nelson Mandela",
            "\u201cFirst, solve the problem. Then, write the code.\u201d \u2014 John Johnson"
        ]
        return f"You're doing great, {intern_name}. Remember:\n\n* {random.choice(motivational_quotes)}"

    return (f"Hi {intern_name}! I am your AI Study Assistant. I need a valid `GROQ_API_KEY` to provide actual answers. "
            "Please provide an API key, or ask me for a 'python tip' or 'motivation' to see a mock response.")