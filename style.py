#!/usr/bin/env python3
"""
Streamlit Frontend for Daycare Food Tracking System
A comprehensive web interface for managing kitchen inventory, food tracking, and daycare reporting.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Daycare Food Tracking System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #2E86AB;
    }
    .section-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #A23B72;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# API Helper Functions
def make_request(method: str, endpoint: str, data: dict = None, params: dict = None) -> dict:
    """Make API request with authentication"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"

    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, params=params)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)

        if response.status_code == 401:
            st.session_state.token = None
            st.session_state.user_info = None
            st.error("Session expired. Please login again.")
            st.rerun()

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                st.error(f"Details: {error_detail}")
            except:
                pass
        return None

def login(username: str, password: str) -> bool:
    """Login user and store token"""
    data = {"username": username, "password": password}
    response = make_request("POST", "/auth/login", data)

    if response:
        st.session_state.token = response["access_token"]
        # Get user info (we'll need to implement this or extract from token)
        return True
    return False

def logout():
    """Logout user"""
    st.session_state.token = None
    st.session_state.user_info = None
    st.rerun()

# Authentication Section
def show_login():
    """Show login form"""
    st.markdown('<h1 class="main-header">🍽️ Daycare Food Tracking System</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="section-header">Login</div>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if username and password:
                    if login(username, password):
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.error("Please enter both username and password")

        st.markdown("---")
        st.markdown("**Demo Accounts:**")
        st.markdown("- Admin: `admin` / `admin123`")
        st.markdown("- Manager: `manager1` / `manager123`")
        st.markdown("- Cook: `cook1` / `cook123`")

# Dashboard Section
def show_dashboard():
    """Show main dashboard"""
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)

    # Get system info
    system_info = make_request("GET", "/system/info")
    if not system_info:
        st.error("Failed to load system information")
        return

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", system_info["system_statistics"]["total_users"])
    with col2:
        st.metric("Total Products", system_info["system_statistics"]["total_products"])
    with col3:
        st.metric("Total Meals", system_info["system_statistics"]["total_meals"])
    with col4:
        st.metric("Low Inventory", system_info["system_statistics"]["low_inventory_products"], delta=None, delta_color="inverse")

    # Portion estimates
    st.markdown('<div class="section-header">Available Portions</div>', unsafe_allow_html=True)
    portions = make_request("GET", "/meals/portions-estimate")

    if portions:
        df_portions = pd.DataFrame(portions)

        # Create bar chart
        fig = px.bar(
            df_portions,
            x='meal_name',
            y='available_portions',
            title="Available Portions by Meal",
            color='available_portions',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Recent activity
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Recent Servings</div>', unsafe_allow_html=True)
        servings = make_request("GET", "/servings/history", params={"limit": 10})

        if servings:
            for serving in servings[:5]:
                with st.expander(f"{serving['meal_name']} - {serving['user_name']}"):
                    st.write(f"**Served at:** {serving['served_at']}")
                    st.write(f"**Status:** {'✅ Success' if serving['success_status'] else '❌ Failed'}")
                    st.write("**Ingredients used:**")
                    for ingredient in serving['ingredients_used']:
                        st.write(f"- {ingredient['product_name']}: {ingredient['used_grams']}g")

    with col2:
        st.markdown('<div class="section-header">Notifications</div>', unsafe_allow_html=True)
        notifications = make_request("GET", "/notifications", params={"unread_only": True})

        if notifications:
            for notification in notifications[:5]:
                alert_type = "warning" if notification['type'] == 'low_inventory' else "danger"
                st.markdown(f"""
                <div class="alert-box alert-{alert_type}">
                    <strong>{notification['type'].replace('_', ' ').title()}:</strong><br>
                    {notification['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No unread notifications")

# Products Section
def show_products():
    """Show products management"""
    st.markdown('<h1 class="main-header">📦 Product Management</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["View Products", "Add Product", "Record Delivery"])

    with tab1:
        products = make_request("GET", "/products")

        if products:
            df = pd.DataFrame(products)
            df['delivery_date'] = pd.to_datetime(df['delivery_date']).dt.strftime('%Y-%m-%d %H:%M')
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

            # Add status column
            df['status'] = df.apply(lambda row:
                '🔴 Low' if row['current_weight_grams'] < row['threshold_warning_grams']
                else '🟢 Good', axis=1)

            st.dataframe(
                df[['name', 'current_weight_grams', 'threshold_warning_grams', 'status', 'delivery_date']],
                use_container_width=True
            )

            # Low inventory alerts
            low_inventory = df[df['current_weight_grams'] < df['threshold_warning_grams']]
            if not low_inventory.empty:
                st.markdown('<div class="section-header">⚠️ Low Inventory Alert</div>', unsafe_allow_html=True)
                for _, product in low_inventory.iterrows():
                    st.markdown(f"""
                    <div class="alert-box alert-warning">
                        <strong>{product['name']}</strong>: {product['current_weight_grams']}g remaining
                        (threshold: {product['threshold_warning_grams']}g)
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Add New Product</div>', unsafe_allow_html=True)

        with st.form("add_product"):
            name = st.text_input("Product Name")
            current_weight = st.number_input("Current Weight (grams)", min_value=0.0, step=100.0)
            threshold = st.number_input("Warning Threshold (grams)", min_value=0.0, step=100.0, value=1000.0)

            if st.form_submit_button("Add Product"):
                data = {
                    "name": name,
                    "current_weight_grams": current_weight,
                    "threshold_warning_grams": threshold
                }
                result = make_request("POST", "/products", data)
                if result:
                    st.success(f"Product '{name}' added successfully!")
                    time.sleep(1)
                    st.rerun()

    with tab3:
        st.markdown('<div class="section-header">Record Delivery</div>', unsafe_allow_html=True)

        products = make_request("GET", "/products")
        if products:
            product_options = {f"{p['name']} (Current: {p['current_weight_grams']}g)": p['id'] for p in products}

            with st.form("record_delivery"):
                selected_product = st.selectbox("Select Product", options=list(product_options.keys()))
                delivery_amount = st.number_input("Delivery Amount (grams)", min_value=0.0, step=100.0)

                if st.form_submit_button("Record Delivery"):
                    product_id = product_options[selected_product]
                    result = make_request("POST", f"/products/{product_id}/delivery", params={"delivery_amount": delivery_amount})
                    if result:
                        st.success(result["message"])
                        time.sleep(1)
                        st.rerun()

# Meals Section
def show_meals():
    """Show meals management"""
    st.markdown('<h1 class="main-header">🍽️ Meal Management</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["View Meals", "Add Meal", "Serve Meal"])

    with tab1:
        meals = make_request("GET", "/meals")

        if meals:
            for meal in meals:
                with st.expander(f"🍽️ {meal['name']}"):
                    st.write(f"**Description:** {meal.get('description', 'N/A')}")
                    st.write("**Ingredients:**")
                    for ingredient in meal['ingredients']:
                        st.write(f"- Product ID {ingredient['product_id']}: {ingredient['required_grams']}g")

                    # Get portion estimate
                    portion_info = make_request("GET", f"/meals/{meal['id']}/portions-available")
                    if portion_info:
                        st.write(f"**Available Portions:** {portion_info['available_portions']}")

    with tab2:
        st.markdown('<div class="section-header">Add New Meal</div>', unsafe_allow_html=True)

        products = make_request("GET", "/products")
        if products:
            product_options = {p['name']: p['id'] for p in products}

            with st.form("add_meal"):
                name = st.text_input("Meal Name")
                description = st.text_area("Description")

                st.write("**Ingredients:**")
                ingredients = []

                # Dynamic ingredient addition
                num_ingredients = st.number_input("Number of ingredients", min_value=1, max_value=10, value=3)

                for i in range(num_ingredients):
                    col1, col2 = st.columns(2)
                    with col1:
                        product = st.selectbox(f"Ingredient {i+1}", options=list(product_options.keys()), key=f"product_{i}")
                    with col2:
                        amount = st.number_input(f"Amount (grams)", min_value=0.0, step=10.0, key=f"amount_{i}")

                    if product and amount > 0:
                        ingredients.append({
                            "product_id": product_options[product],
                            "required_grams": amount
                        })

                if st.form_submit_button("Add Meal"):
                    if name and ingredients:
                        data = {
                            "name": name,
                            "description": description,
                            "ingredients": ingredients
                        }
                        result = make_request("POST", "/meals", data)
                        if result:
                            st.success(f"Meal '{name}' added successfully!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Please fill in meal name and at least one ingredient")

    with tab3:
        st.markdown('<div class="section-header">Serve Meal</div>', unsafe_allow_html=True)

        meals = make_request("GET", "/meals")
        if meals:
            meal_options = {meal['name']: meal['id'] for meal in meals}

            selected_meal_name = st.selectbox("Select Meal to Serve", options=list(meal_options.keys()))

            if selected_meal_name:
                meal_id = meal_options[selected_meal_name]

                # Show portion availability
                portion_info = make_request("GET", f"/meals/{meal_id}/portions-available")
                if portion_info:
                    if portion_info['available_portions'] > 0:
                        st.success(f"✅ {portion_info['available_portions']} portions available")

                        if st.button("🍽️ Serve Meal", type="primary"):
                            result = make_request("POST", f"/meals/{meal_id}/serve")
                            if result:
                                st.success(result["message"])
                                st.write("**Ingredients used:**")
                                for ingredient in result["ingredients_used"]:
                                    st.write(f"- {ingredient['product_name']}: {ingredient['used_grams']}g")
                                time.sleep(2)
                                st.rerun()
                    else:
                        st.error("❌ No portions available - insufficient ingredients")

# Reports Section
def show_reports():
    """Show reports and analytics"""
    st.markdown('<h1 class="main-header">📊 Reports & Analytics</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Ingredient Usage", "Monthly Summary", "User Activity", "Discrepancy Analysis"])

    with tab1:
        st.markdown('<div class="section-header">Ingredient Usage Report</div>', unsafe_allow_html=True)

        days = st.slider("Report Period (days)", min_value=7, max_value=365, value=30)

        if st.button("Generate Usage Report"):
            report = make_request("GET", "/reports/ingredient-usage", params={"days": days})

            if report and report["ingredient_usage"]:
                st.write(f"**Report Period:** {report['start_date']} to {report['end_date']}")

                # Create dataframe for visualization
                usage_data = []
                for product_name, data in report["ingredient_usage"].items():
                    usage_data.append({
                        "Product": product_name,
                        "Deliveries": data["deliveries"],
                        "Usage": data["usage"],
                        "Adjustments": data["adjustments"],
                        "Net Change": data["deliveries"] - data["usage"] + data["adjustments"]
                    })

                df = pd.DataFrame(usage_data)

                # Display metrics
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(df, x="Product", y="Usage", title="Ingredient Usage")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(df, x="Product", y="Deliveries", title="Deliveries Received")
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Monthly Summary Reports</div>', unsafe_allow_html=True)

        reports = make_request("GET", "/reports/monthly-summary")

        if reports:
            df = pd.DataFrame(reports)
            df['Month/Year'] = df['month'].astype(str) + '/' + df['year'].astype(str)

            # Line chart for discrepancy rate
            fig = px.line(df, x='Month/Year', y='discrepancy_rate',
                         title='Monthly Discrepancy Rate Trend',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart for served vs possible
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Total Served', x=df['Month/Year'], y=df['total_served']))
            fig.add_trace(go.Bar(name='Total Possible', x=df['Month/Year'], y=df['total_possible']))
            fig.update_layout(title='Monthly Meals: Served vs Possible', barmode='group')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df[['Month/Year', 'total_served', 'total_possible', 'discrepancy_rate']], use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">User Activity Report</div>', unsafe_allow_html=True)

        days = st.slider("Activity Period (days)", min_value=7, max_value=365, value=30, key="user_activity_days")

        if st.button("Generate Activity Report"):
            report = make_request("GET", "/reports/user-activity", params={"days": days})

            if report and report["user_activity"]:
                st.write(f"**Report Period:** {report['start_date']} to {report['end_date']}")

                # Create summary dataframe
                activity_data = []
                for user_name, data in report["user_activity"].items():
                    activity_data.append({
                        "User": user_name,
                        "Total Meals Served": data["total_meals_served"],
                        "Unique Meal Types": len(data["meals_by_type"])
                    })

                df = pd.DataFrame(activity_data)

                # Bar chart
                fig = px.bar(df, x="User", y="Total Meals Served", title="Meals Served by User")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header">Discrepancy Analysis</div>', unsafe_allow_html=True)

        if st.button("Generate Discrepancy Analysis"):
            analysis = make_request("GET", "/reports/discrepancy-analysis")

            if analysis:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Average Discrepancy Rate", f"{analysis['average_discrepancy_rate']:.2f}%")
                with col2:
                    trend_emoji = "📈" if analysis['trend'] == 'increasing' else "📉" if analysis['trend'] == 'decreasing' else "➡️"
                    st.metric("Trend", f"{trend_emoji} {analysis['trend'].title()}")
                with col3:
                    st.metric("High Risk Months", len(analysis['high_risk_months']))

                if analysis['monthly_data']:
                    df = pd.DataFrame(analysis['monthly_data'])
                    df['Month/Year'] = df['month'].astype(str) + '/' + df['year'].astype(str)

                    fig = px.bar(df, x='Month/Year', y='discrepancy_rate',
                               title='Discrepancy Rate by Month',
                               color='discrepancy_rate',
                               color_continuous_scale='RdYlGn_r')
                    fig.add_hline(y=15.0, line_dash="dash", line_color="red",
                                annotation_text="High Risk Threshold (15%)")
                    st.plotly_chart(fig, use_container_width=True)

# Main Application
def main():
    """Main application"""
    if not st.session_state.token:
        show_login()
        return

    # Sidebar navigation
    with st.sidebar:
        st.title("🍽️ Navigation")

        pages = ["Dashboard", "Products", "Meals", "Reports"]
        selected_page = st.radio("Go to", pages, key="nav_radio")

        st.markdown("---")

        # Quick actions
        st.markdown("**Quick Actions**")
        if st.button("🔄 Check Inventory", use_container_width=True):
            result = make_request("POST", "/tasks/check-inventory")
            if result:
                st.success("Inventory check triggered!")

        if st.button("📊 Generate Report", use_container_width=True):
            result = make_request("POST", "/tasks/generate-monthly-report")
            if result:
                st.success("Report generation triggered!")

        st.markdown("---")

        # User info and logout
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    # Main content area
    if selected_page == "Dashboard":
        show_dashboard()
    elif selected_page == "Products":
        show_products()
    elif selected_page == "Meals":
        show_meals()
    elif selected_page == "Reports":
        show_reports()

if __name__ == "__main__":
    main()