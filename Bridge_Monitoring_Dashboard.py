import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Bridge Sensor Monitoring Dashboard",
    page_icon="📈",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(135deg, #102a43 0%, #1f6f8b 100%);
        padding: 20px 28px;
        border-radius: 14px;
        margin-bottom: 16px;
        color: white;
        box-shadow: 0 8px 22px rgba(16, 42, 67, 0.18);
    }

    .hero h1 {
        color: white;
        font-size: 1.95rem;
        margin: 0 0 4px 0;
    }

    .hero p {
        color: #d9edf5;
        font-size: 0.95rem;
        margin: 0;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e8ef;
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 4px 12px rgba(16, 42, 67, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #52606d;
    }

    [data-testid="stMetricValue"] {
        color: #102a43;
    }

    h2, h3 {
        color: #102a43;
    }

    .section-label {
        color: #627d98;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 10px;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        margin-bottom: 0.15rem;
    }

    h2 {
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
    }

    h3 {
        margin-top: 0.45rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Bridge Sensor Monitoring and Simulation Dashboard</h1>
        <p>Historical monitoring overview with an adjustable anomaly scenario</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data = pd.read_csv("dashboard_input.csv")
    data["Time"] = pd.to_datetime(data["Time"])
    data["Event_sequence"] = np.arange(1, len(data) + 1)
    return data


try:
    base_data = load_data()
except FileNotFoundError:
    st.error(
        "dashboard_input.csv was not found. "
        "Run the export cell in the notebook first and place the CSV "
        "in the same folder as dashboard.py."
    )
    st.stop()


with st.sidebar:
    st.header("Simulation scenario")

    event_sequence = st.slider(
        "Event sequence",
        min_value=1,
        max_value=len(base_data),
        value=min(20, len(base_data)),
        step=1,
    )

    downtime_hours = st.slider(
        "Downtime (hrs)",
        min_value=1.0,
        max_value=7.0,
        value=1.7,
        step=0.1,
    )

    anomaly_strength = st.slider(
        "Simulated anomaly strength",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )

    affected_events = st.slider(
        "Affected events",
        min_value=1,
        max_value=min(10, len(base_data)),
        value=min(3, len(base_data)),
        step=1,
    )


results = base_data.copy()

# Use the exact thresholds exported from the validated notebook analysis.
if "PCA_threshold" not in base_data.columns or "Baseline_threshold" not in base_data.columns:
    st.error(
        "The CSV does not contain the validated notebook thresholds. "
        "Rerun the notebook export cell and replace dashboard_input.csv."
    )
    st.stop()

pca_threshold = float(base_data["PCA_threshold"].iloc[0])
baseline_threshold = float(base_data["Baseline_threshold"].iloc[0])

start_position = event_sequence - 1
end_position = min(
    start_position + affected_events,
    len(results),
)

affected_index = results.index[
    (results.index >= start_position)
    & (results.index < end_position)
]

downtime_multiplier = downtime_hours / 1.7
simulated_increase = anomaly_strength * downtime_multiplier

results.loc[affected_index, "PCA_Mahalanobis"] += simulated_increase
results.loc[affected_index, "Univariate_baseline"] += simulated_increase

results["PCA_warning"] = (
    results["PCA_Mahalanobis"] >= pca_threshold
)

results["Baseline_warning"] = (
    results["Univariate_baseline"] >= baseline_threshold
)


st.markdown(
    '<div class="section-label">Condition analysis</div>',
    unsafe_allow_html=True,
)
st.subheader("Anomaly scores and warning events")

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 6.4),
    sharex=True,
)

axes[0].plot(
    results["Event_sequence"],
    results["PCA_Mahalanobis"],
    color="#0072B2",
    linewidth=2,
    label="PCA–Mahalanobis score",
)

axes[0].axhline(
    pca_threshold,
    color="#D55E00",
    linestyle="--",
    linewidth=2,
    label="Paper threshold",
)

axes[0].scatter(
    results.loc[results["PCA_warning"], "Event_sequence"],
    results.loc[results["PCA_warning"], "PCA_Mahalanobis"],
    color="red",
    s=50,
    label="Warning",
    zorder=3,
)

axes[0].axvspan(
    event_sequence,
    min(event_sequence + affected_events - 1, len(results)),
    color="orange",
    alpha=0.18,
    label="Simulated abnormal period",
)

axes[0].set_ylabel("PCA score")
axes[0].set_title("PCA–Mahalanobis detection", fontweight="bold")
axes[0].legend(loc="upper left")
axes[0].grid(axis="y", linestyle="--", alpha=0.35)

axes[1].plot(
    results["Event_sequence"],
    results["Univariate_baseline"],
    color="#009E73",
    linewidth=2,
    label="Univariate baseline score",
)

axes[1].axhline(
    baseline_threshold,
    color="#D55E00",
    linestyle="--",
    linewidth=2,
    label="Paper threshold",
)

axes[1].scatter(
    results.loc[results["Baseline_warning"], "Event_sequence"],
    results.loc[results["Baseline_warning"], "Univariate_baseline"],
    color="red",
    s=50,
    label="Warning",
    zorder=3,
)

axes[1].axvspan(
    event_sequence,
    min(event_sequence + affected_events - 1, len(results)),
    color="orange",
    alpha=0.18,
)

axes[1].set_xlabel("Event sequence")
axes[1].set_ylabel("Baseline score")
axes[1].set_title("Univariate baseline detection", fontweight="bold")
axes[1].legend(loc="upper left")
axes[1].grid(axis="y", linestyle="--", alpha=0.35)

for axis in axes:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.markdown(
    '<div class="section-label">Simulation scenario — hypothetical</div>',
    unsafe_allow_html=True,
)

st.info(
    "Adjust the sidebar controls to explore a hypothetical anomaly. "
    "These results do not modify the validated historical results below."
)

st.subheader("Effect of the simulated anomaly")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("PCA warnings", int(results["PCA_warning"].sum()))

with col2:
    st.metric("Baseline warnings", int(results["Baseline_warning"].sum()))

with col3:
    st.metric("Simulated downtime", f"{downtime_hours:.1f} hrs")

st.subheader("Simulation summary")

summary = pd.DataFrame({
    "Setting": [
        "Selected event sequence",
        "Downtime duration (hrs)",
        "Affected events",
        "PCA threshold from notebook",
        "Baseline threshold from notebook",
        "Anomaly strength",
        "PCA warning events",
        "Baseline warning events",
    ],
    "Value": [
        event_sequence,
        downtime_hours,
        affected_events,
        pca_threshold,
        baseline_threshold,
        anomaly_strength,
        int(results["PCA_warning"].sum()),
        int(results["Baseline_warning"].sum()),
    ],
})

st.dataframe(summary, hide_index=True, width="stretch")


st.markdown(
    '<div class="section-label">Monitoring overview</div>',
    unsafe_allow_html=True,
)

st.subheader("Historical monitoring status")

monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)

with monitor_col1:
    st.metric("Events analyzed", len(base_data))

with monitor_col2:
    st.metric(
        "Historical PCA warnings",
        int(base_data["PCA_Warning"].sum()),
    )

with monitor_col3:
    st.metric(
        "Historical baseline warnings",
        int(base_data["Baseline_Warning"].sum()),
    )

with monitor_col4:
    historical_status = (
        "Warning"
        if (
            base_data["PCA_Warning"].any()
            or base_data["Baseline_Warning"].any()
        )
        else "Normal"
    )
    st.metric("Historical status", historical_status)

with st.expander("View historical warning events"):
    historical_warning_table = base_data.loc[
        base_data["PCA_Warning"] | base_data["Baseline_Warning"],
        [
            "Event_sequence",
            "Event",
            "Time",
            "PCA_Mahalanobis",
            "PCA_Warning",
            "Univariate_baseline",
            "Baseline_Warning",
        ],
    ]
    st.dataframe(
        historical_warning_table,
        hide_index=True,
        width="stretch",
    )


st.markdown(
    '<div class="section-label">Historical method comparison — fixed reference</div>',
    unsafe_allow_html=True,
)

st.info(
    "These figures show the validated historical results reported in the "
    "notebook and paper. They remain fixed when the simulation controls change."
)

st.subheader("Historical score distributions and warning agreement")

# Use full-width containers so the charts remain readable on screen.
distribution_col = st.container()
agreement_col = st.container()

with distribution_col:
    distribution_fig, distribution_axes = plt.subplots(
        1,
        2,
        figsize=(12, 3.6),
    )

    distribution_axes[0].hist(
        base_data["PCA_Mahalanobis"],
        bins=12,
        color="#2f80ed",
        alpha=0.78,
        edgecolor="white",
    )

    distribution_axes[0].axvline(
        pca_threshold,
        color="#d55e00",
        linestyle="--",
        linewidth=2,
        label="Warning threshold",
    )

    distribution_axes[0].set_title("PCA score distribution")
    distribution_axes[0].set_xlabel("PCA–Mahalanobis score")
    distribution_axes[0].set_ylabel("Number of events")
    distribution_axes[0].legend(fontsize=8)

    distribution_axes[1].hist(
        base_data["Univariate_baseline"],
        bins=12,
        color="#27ae60",
        alpha=0.78,
        edgecolor="white",
    )

    distribution_axes[1].axvline(
        baseline_threshold,
        color="#d55e00",
        linestyle="--",
        linewidth=2,
        label="Warning threshold",
    )

    distribution_axes[1].set_title("Baseline score distribution")
    distribution_axes[1].set_xlabel("Baseline score")
    distribution_axes[1].set_ylabel("Number of events")
    distribution_axes[1].legend(fontsize=8)

    for axis in distribution_axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    distribution_fig.tight_layout()
    st.pyplot(distribution_fig)
    plt.close(distribution_fig)

with agreement_col:
    # Use the validated historical warning decisions for this fixed
    # method-comparison figure, not the adjustable simulation results.
    pca_warning = base_data["PCA_Warning"]
    baseline_warning = base_data["Baseline_Warning"]

    agreement_counts = pd.Series({
        "Both methods": int((pca_warning & baseline_warning).sum()),
        "PCA only": int((pca_warning & ~baseline_warning).sum()),
        "Baseline only": int((~pca_warning & baseline_warning).sum()),
        "Neither method": int((~pca_warning & ~baseline_warning).sum()),
    })

    agreement_fig, agreement_axis = plt.subplots(
        figsize=(8, 3.8),
    )

    agreement_bars = agreement_axis.bar(
        agreement_counts.index,
        agreement_counts.values,
        color=["#1f6f8b", "#2f80ed", "#27ae60", "#b8c4ce"],
        edgecolor="white",
    )

    agreement_axis.set_title("Warning agreement by method")
    agreement_axis.set_ylabel("Number of events")
    agreement_axis.tick_params(axis="x", rotation=15)
    agreement_axis.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in agreement_bars:
        agreement_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())}",
            ha="center",
            fontweight="bold",
        )

    agreement_axis.spines["top"].set_visible(False)
    agreement_axis.spines["right"].set_visible(False)
    agreement_fig.subplots_adjust(bottom=0.25)
    agreement_fig.tight_layout()
    st.pyplot(agreement_fig)
    plt.close(agreement_fig)


st.caption(
    "Course-project prototype using historical event-level results and simulated changes."
)
