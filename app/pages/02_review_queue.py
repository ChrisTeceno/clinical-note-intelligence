"""Review Queue — filter and browse notes awaiting review."""

import streamlit as st
from db import query_df
from tooltips import TOOLTIP_CSS, tt

st.title("Review Queue")
st.caption("Browse extractions by status and specialty. Click a note ID to open it in Note Detail.")

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    status_options = ["All", "pending", "in_review", "approved", "rejected"]
    selected_status = st.selectbox("Status", status_options, index=0)

with filter_col2:
    specialties = query_df(
        "SELECT DISTINCT medical_specialty AS s FROM clinical_notes ORDER BY s"
    )["s"].tolist()
    specialty_options = ["All"] + specialties
    selected_specialty = st.selectbox("Specialty", specialty_options, index=0)

# ---------------------------------------------------------------------------
# Build query
# ---------------------------------------------------------------------------
where_clauses: list[str] = []
params: list[str] = []

if selected_status != "All":
    where_clauses.append("e.status = ?")
    params.append(selected_status)

if selected_specialty != "All":
    where_clauses.append("cn.medical_specialty = ?")
    params.append(selected_specialty)

where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

queue_df = query_df(
    f"""
    SELECT
        cn.source_id,
        e.note_id,
        cn.medical_specialty AS specialty,
        cn.sample_name,
        e.chief_complaint,
        e.status,
        e.created_at,
        (SELECT COUNT(*) FROM diagnoses d WHERE d.extraction_id = e.id) AS dx,
        (SELECT COUNT(*) FROM procedures p WHERE p.extraction_id = e.id) AS px,
        (SELECT COUNT(*) FROM medications m WHERE m.extraction_id = e.id) AS rx,
        ROUND(
            (SELECT AVG(d2.match_score)
             FROM diagnoses d2
             WHERE d2.extraction_id = e.id AND d2.match_score IS NOT NULL), 1
        ) AS avg_match
    FROM extractions e
    JOIN clinical_notes cn ON cn.id = e.note_id
    {where_sql}
    ORDER BY
        CASE e.status
            WHEN 'pending' THEN 0
            WHEN 'in_review' THEN 1
            WHEN 'approved' THEN 2
            WHEN 'rejected' THEN 3
        END,
        e.created_at DESC
    """,
    tuple(params),
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
st.markdown(f"**{len(queue_df)}** extractions match the current filters.")

# ---------------------------------------------------------------------------
# Table with clickable note IDs
# ---------------------------------------------------------------------------
if not queue_df.empty:
    # Header row with tooltip explanations
    hdr = st.columns([1.5, 2, 2.5, 1, 0.5, 0.5, 0.5, 1.5])
    hdr[0].markdown("**Note**")
    hdr[1].markdown("**Specialty**")
    hdr[2].markdown("**Name**")
    hdr[3].markdown(
        tt(
            "<strong>Status</strong>",
            "Extraction review status: pending, "
            "in_review, approved, or rejected",
        ),
        unsafe_allow_html=True,
    )
    hdr[4].markdown(
        tt("<strong>Dx</strong>", "Number of diagnoses extracted from this clinical note"),
        unsafe_allow_html=True,
    )
    hdr[5].markdown(
        tt("<strong>Px</strong>", "Number of procedures extracted from this clinical note"),
        unsafe_allow_html=True,
    )
    hdr[6].markdown(
        tt("<strong>Rx</strong>", "Number of medications extracted from this clinical note"),
        unsafe_allow_html=True,
    )
    hdr[7].markdown(
        tt(
            "<strong>Match</strong>",
            "Average ICD-10 fuzzy match score across all "
            "diagnoses for this note (higher is better)",
        ),
        unsafe_allow_html=True,
    )
    st.divider()

    for _, row in queue_df.iterrows():
        with st.container():
            cols = st.columns([1.5, 2, 2.5, 1, 0.5, 0.5, 0.5, 1.5])
            with cols[0]:
                note_url = f"/note_detail?note_id={row['note_id']}"
                st.markdown(
                    f"<a href='{note_url}' target='_self' style='text-decoration:none'>"
                    f"&#128196; #{row['source_id']}</a>",
                    unsafe_allow_html=True,
                )
            cols[1].write(row["specialty"])
            cols[2].write(row["sample_name"] or row["chief_complaint"] or "—")
            status = row["status"]
            status_colors = {
                "pending": "orange",
                "in_review": "blue",
                "approved": "green",
                "rejected": "red",
            }
            status_tips = {
                "pending": "Awaiting human review",
                "in_review": "Currently being reviewed",
                "approved": "Confirmed accurate by reviewer",
                "rejected": "Flagged as inaccurate, needs correction",
            }
            badge_html = (
                f"<span style='background:{status_colors.get(status, 'gray')};color:white;"
                f"padding:2px 8px;border-radius:10px;font-size:0.8em'>{status}</span>"
            )
            cols[3].markdown(
                tt(badge_html, status_tips.get(status, "")),
                unsafe_allow_html=True,
            )
            cols[4].write(f"{row['dx']}")
            cols[5].write(f"{row['px']}")
            cols[6].write(f"{row['rx']}")
            match = row["avg_match"]
            if match and match > 0:
                cols[7].progress(match / 100, text=f"{match:.0f}%")
            else:
                cols[7].write("—")
else:
    st.warning("No extractions match the current filters.")
