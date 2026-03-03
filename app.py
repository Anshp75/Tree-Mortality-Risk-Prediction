import streamlit as st
import numpy as np
from PIL import Image
from src.image_processing import analyze_tree
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tree SentinelAI", page_icon="🌲", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stProgress > div > div > div > div { background-color: #2ecc71; }
    </style>
    """, unsafe_allow_html=True)

st.title("Tree SentinelAI 🌲")
st.markdown("### Stage 2: Environmental Health, Restoration & Growth Advisor")
st.divider()

# --- SIDEBAR INFO ---
st.sidebar.header("System Intelligence")
st.sidebar.write("**Model:** CNN-Hybrid MobilenetV2")
st.sidebar.write("**Features:**")
st.sidebar.write("- ✅ AI Image Restoration")
st.sidebar.write("- ✅ Thirsty-Tree Regression")
st.sidebar.write("- ✅ Sunlight Exposure Check")
st.sidebar.write("- ✅ Growth Suggestions")

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Tree Image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    
    if st.button("🚀 Run Full Diagnostic"):
        # 1. ANALYSIS STEP
        with st.status("Performing Multi-Stage Analysis...", expanded=True) as status_box:
            st.write("🛠️ Applying AI Restoration (Denoising)...")
            status, water, light, img_restored = analyze_tree(img_raw)
            time.sleep(0.8)
            st.write("🔬 Calculating Chlorophyll & Hydration (Regression)...")
            time.sleep(0.8)
            st.write("☀️ Analyzing Solar Exposure & Urban Threats...")
            status_box.update(label="Diagnostic Complete!", state="complete", expanded=False)

        # 2. IMAGE RESTORATION COMPARISON
        st.subheader("🖼️ Image Restoration Process")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img_raw, caption="User Uploaded (Raw/Edited)", use_container_width=True)
        with col_b:
            st.image(img_restored, caption="AI Restored (Cleaned for Analysis)", use_container_width=True)

        # 3. CORE DIAGNOSTIC METRICS
        st.divider()
        st.subheader("📊 Diagnostic Results")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            if status == "LIVE":
                st.success(f"**Life Status:** {status}")
            else:
                st.error(f"**Life Status:** {status}")
        with m2:
            st.metric("Hydration Level", f"{water}%")
            st.progress(water)
        with m3:
            light_label = "Optimal" if light > 100 else ("Low" if light > 50 else "Critical")
            st.metric("Sunlight Exposure", light_label)

        # 4. AI GROWTH SUGGESTION ENGINE
        st.divider()
        st.subheader("🌱 AI Maintenance & Growth Suggestions")
        
        # Logic for Advice
        if status == "DEAD / STUMP":
            st.error("⚠️ **Action Required:** Schedule for removal with NYC Parks.")
            st.write("💡 **Growth Tip:** Replant with resilient species like 'London Plane' or 'Ginkgo'.")
        
        elif water < 60:
            st.warning("💧 **Irrigation Alert:** Severe water stress detected.")
            st.markdown("""
            * **Action:** Provide 15-20 gallons of water at the base.
            * **Growth Tip:** Apply organic mulch to keep the soil cool and moist.
            """)
            
        if light < 80 and status == "LIVE":
            st.info("⛅ **Sunlight Alert:** Light deprivation detected.")
            st.markdown("""
            * **Action:** Prune blocking branches or relocate if it's a potted tree.
            * **Growth Tip:** Clean leaves of urban dust/soot to improve photosynthesis.
            """)
            
        if status == "LIVE" and water >= 80 and light >= 100:
            st.balloons()
            st.success("✅ **Maintenance:** No intervention needed. Tree is in peak health!")

else:
    st.info("Please upload an image from your dataset to begin the AI diagnostic.")

# --- FOOTER ---
st.divider()
st.caption("Tree SentinelAI 2026 | Developed for NYC Urban Forestry Analysis")