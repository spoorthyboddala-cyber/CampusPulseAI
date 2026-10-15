
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
# 1. Page Configuration
st.set_page_config(
    page_title="CampusPulse AI",
    page_icon="🏫",
    layout="wide"
)
# 2. Load Model and Vectorizer
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "campus_model.joblib"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)


try:
    model = load_model()
    vectorizer = load_vectorizer()

except Exception as error:
    st.error(
        "Could not load the model or vectorizer. "
        "Check that both files exist and were saved "
        "in a compatible format."
    )
    st.code(str(error))
    st.stop()
# 3. Store Reports
if "reports" not in st.session_state:
    st.session_state.reports = []
# 4. Determine Review Priority
def get_priority(complaint, category, previous_count):

    text = complaint.lower()

    urgent_terms = [
        "emergency",
        "exposed electrical wire",
        "fire",
        "dangerous",
        "blocked emergency exit",
        "broken railing"
    ]

    if category == "Safety" or any(
        term in text for term in urgent_terms
    ):
        return "Urgent review"

    if previous_count >= 3:
        return "Repeated category"

    return "Normal review"
# 5. Application Heading
st.title("🏫 CampusPulse AI")
st.subheader("Campus Problem Prediction & Intelligence")

st.write(
    "Report campus problems, classify them using machine "
    "learning, and view trends in submitted reports."
)

st.info(
    "Student prototype. Predictions need validation, and "
    "priority labels are suggestions—not emergency alerts."
)
# 6. Complaint Form
st.header("📝 Report a Campus Problem")

with st.form("complaint_form"):

    complaint = st.text_area(
        "Describe the problem",
        placeholder="Example: Wi-Fi keeps disconnecting in the lab."
    )

    location = st.selectbox(
        "Campus location",
        [
            "Academic Block",
            "Computer Lab",
            "Library",
            "Canteen",
            "Hostel",
            "Bus Area",
            "Other"
        ]
    )

    submitted = st.form_submit_button("Analyze Complaint")
# 7. Analyze Complaint
if submitted:

    if len(complaint.strip()) < 5:
        st.warning(
            "Please describe the problem in more detail."
        )

    else:
        complaint_vector = vectorizer.transform(
            [complaint.strip()]
        )

        category = model.predict(complaint_vector)[0]

        # Calculate confidence only if supported
        # by the trained model.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(
                complaint_vector
            )[0]

            model_score = float(max(scores)) * 100
        else:
            model_score = None

        previous_count = sum(
            report["Category"] == category
            for report in st.session_state.reports
        )

        priority = get_priority(
            complaint,
            category,
            previous_count + 1
        )

        report = {
            "Complaint": complaint.strip(),
            "Location": location,
            "Category": category,
            "Priority": priority
        }

        if model_score is not None:
            report["Model Score (%)"] = round(
                model_score, 2
            )

        st.session_state.reports.append(report)

        st.success("Complaint analyzed successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Category",
            str(category)
        )

        if model_score is not None:
            col2.metric(
                "Model Score",
                f"{model_score:.1f}%"
            )
        else:
            col2.metric("Model Score", "Not available")

        col3.metric(
            "Review Status",
            priority
        )

        st.caption(
            "The model score is not a guarantee of correctness."
        )
# 8. Campus Intelligence Dashboard
st.divider()
st.header("📊 Campus Intelligence Dashboard")

if st.session_state.reports:

    df = pd.DataFrame(st.session_state.reports)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reports",
        len(df)
    )

    col2.metric(
        "Categories",
        df["Category"].nunique()
    )

    col3.metric(
        "Urgent Reviews",
        int(
            (df["Priority"] == "Urgent review").sum()
        )
    )

    st.subheader("Reports by Category")

    st.bar_chart(
        df["Category"].value_counts()
    )

    st.subheader("Reports by Campus Location")

    st.bar_chart(
        df["Location"].value_counts()
    )

    st.subheader("Submitted Reports")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Reports as CSV",
        data=csv_data,
        file_name="campus_reports.csv",
        mime="text/csv"
    )

else:

    st.write("No reports submitted yet.")

    st.caption(
        "Submit a few sample complaints to populate the dashboard."
    )
# 9. Footer
st.divider()

st.caption(
    "CampusPulse AI | Student ML IDEATHON Prototype"
)
