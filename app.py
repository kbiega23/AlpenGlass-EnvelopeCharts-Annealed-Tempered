import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AlpenGlass Window Size Visualizer",
    page_icon="🪟",
    layout="wide"
)

# Title and description
st.title("🪟 AlpenGlass Sizing Limits")

# Add comprehensive directions in collapsible expander
with st.expander("📖 How to Use This Tool - Click to expand"):
    st.markdown("""
This interactive tool helps you determine if your window dimensions fit within AlpenGlass's manufacturing capabilities for different glass configurations.

**Glass Type Selection:**
- **Tempered Glass**: Shows rectangular envelopes based on maximum long edge and short edge dimensions
- **Annealed Glass**: Shows curved envelopes based on maximum area and maximum edge length (Sizing based on wind load of DP30. Contact your sales rep if higher wind loads needed in your situation)

**Understanding the Visualization:**
- **Core Range** (blue): Efficient, low-cost production range
- **Technical Limit** (orange): Maximum physically achievable size (may require special order and longer lead time)
- **Minimum Size**: At least one edge must be 16" or greater
- **White areas**: Do not meet minimum size requirements

**Configuration Selection:**
- **Select "All"**: View the composite envelope showing the maximum achievable sizes across all configurations in your filter
- **Select Specific Values**: View the exact size limits for a particular glass configuration

**Checking Your Custom Size:**
1. Choose glass type (Tempered or Annealed)
2. Use the dropdowns to filter by glass specifications (or leave as "All")
3. Enter your desired width and height in the custom size input fields
4. A star will appear on the chart showing your size's location
5. Check the status indicator to see if it falls within Core Range, Technical Limit, or outside our capabilities

**Interpreting the Chart:**
- Hover over any point to see exact dimensions and area
- The chart displays both portrait and landscape orientations
- Download the chart as PNG to save the configuration details
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the glass configuration data from Excel file"""
    import os
    
    possible_names = [
        'AlpenGlass max sizing data.xlsx',
        'AlpenGlass_max_sizing_data.xlsx',
        'alpenglass_max_sizing_data.xlsx',
    ]
    
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                tempered_df = pd.read_excel(filename, sheet_name='tempered')
                annealed_df = pd.read_excel(filename, sheet_name='annealed')
                return tempered_df, annealed_df
            except Exception as e:
                st.error(f"Error reading {filename}: {str(e)}")
                return None, None
    
    st.error("Excel file not found.")
    return None, None

def create_tempered_plot(config_data, min_edge=16, show_all=False, all_configs_df=None, custom_point=None, filter_text=""):
    """Create plotly figure for tempered glass"""
    
    if config_data.empty:
        return None
    
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        core_long = all_configs_df['CoreRange_ maxlongedge_inches'].max()
        core_short = all_configs_df['CoreRange_maxshortedge_inches'].max()
        tech_long = all_configs_df['Technical_limit_longedge_inches'].max()
        tech_short = all_configs_df['Technical_limit_shortedge_inches'].max()
    else:
        core_long = config_data['CoreRange_ maxlongedge_inches'].values[0]
        core_short = config_data['CoreRange_maxshortedge_inches'].values[0]
        tech_long = config_data['Technical_limit_longedge_inches'].values[0]
        tech_short = config_data['Technical_limit_shortedge_inches'].values[0]
    
    fig = go.Figure()
    
    x_range = np.arange(0, 151, 1)
    y_range = np.arange(0, 151, 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X, dtype=float)
    hover_text = []
    
    for i in range(len(y_range)):
        row_text = []
        for j in range(len(x_range)):
            x, y = X[i, j], Y[i, j]
            
            meets_min = (x >= min_edge or y >= min_edge)
            in_tech = ((x <= tech_long and y <= tech_short) or 
                      (x <= tech_short and y <= tech_long)) and meets_min
            in_core = ((x <= core_long and y <= core_short) or 
                      (x <= core_short and y <= core_long)) and meets_min
            
            area_sqft = (x * y) / 144
            
            if in_core:
                Z[i, j] = 2
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Core Range</b>")
            elif in_tech:
                Z[i, j] = 1
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>⚠️ Technical Limit</b>")
            else:
                Z[i, j] = 0
                if not meets_min:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Below minimum</b>")
                else:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Outside limits</b>")
        hover_text.append(row_text)
    
    fig.add_trace(go.Heatmap(
        x=x_range, y=y_range, z=Z,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False, hoverinfo='text', text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    tech_x = [min_edge, tech_long, tech_long, tech_short, tech_short, 0, 0, min_edge, min_edge]
    tech_y = [0, 0, tech_short, tech_short, tech_long, tech_long, min_edge, min_edge, 0]
    
    fig.add_trace(go.Scatter(
        x=tech_x, y=tech_y, fill='toself',
        fillcolor='rgba(255, 152, 0, 0.2)',
        line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
        name='Technical Limit', hoverinfo='skip'
    ))
    
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        all_x, all_y = [], []
        for idx, row in all_configs_df.iterrows():
            c_long = row['CoreRange_ maxlongedge_inches']
            c_short = row['CoreRange_maxshortedge_inches']
            rect_x = [min_edge, c_long, c_long, c_short, c_short, 0, 0, min_edge, min_edge]
            rect_y = [0, 0, c_short, c_short, c_long, c_long, min_edge, min_edge, 0]
            all_x.extend(rect_x)
            all_y.extend(rect_y)
            if idx < len(all_configs_df) - 1:
                all_x.append(None)
                all_y.append(None)
        
        fig.add_trace(go.Scatter(
            x=all_x, y=all_y, fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)', line=dict(width=0),
            name='Core Range', hoverinfo='skip'
        ))
    else:
        core_x = [min_edge, core_long, core_long, core_short, core_short, 0, 0, min_edge, min_edge]
        core_y = [0, 0, core_short, core_short, core_long, core_long, min_edge, min_edge, 0]
        fig.add_trace(go.Scatter(
            x=core_x, y=core_y, fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(color='rgba(33, 150, 243, 1)', width=3),
            name='Core Range', hoverinfo='skip'
        ))
    
    if custom_point:
        add_custom_point(fig, custom_point, min_edge, core_long, core_short, tech_long, tech_short, False)
    
    title_text = "AlpenGlass Sizing Limits - Tempered Glass"
    if filter_text:
        title_text += f"<br><sub>{filter_text}</sub>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title="Width (inches)", yaxis_title="Height (inches)",
        xaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', fixedrange=True, constrain='domain'),
        yaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1, fixedrange=True, constrain='domain'),
        plot_bgcolor='white', hovermode='closest', height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                   font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1)
    )
    return fig

def create_annealed_plot(config_data, min_edge=16, show_all=False, all_configs_df=None, custom_point=None, filter_text=""):
    """Create plotly figure for annealed glass with area constraints"""
    
    if config_data.empty:
        return None
    
    # Fixed: Use correct column names from Excel file
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        core_max_edge = all_configs_df['CoreRange_maxedge_inches'].max()
        tech_max_edge = all_configs_df['Technical_limit_maxedge_inches'].max()
        core_max_area = all_configs_df['MaxArea_Core_squarefeet'].max() * 144
        tech_max_area = all_configs_df['MaxArea_Technical_limit_squarefeet'].max() * 144
    else:
        core_max_edge = config_data['CoreRange_maxedge_inches'].values[0]
        tech_max_edge = config_data['Technical_limit_maxedge_inches'].values[0]
        core_max_area = config_data['MaxArea_Core_squarefeet'].values[0] * 144
        tech_max_area = config_data['MaxArea_Technical_limit_squarefeet'].values[0] * 144
    
    # Check if core and technical limits are the same
    limits_are_same = (core_max_edge == tech_max_edge and core_max_area == tech_max_area)
    
    fig = go.Figure()
    
    x_range = np.arange(0, 151, 1)
    y_range = np.arange(0, 151, 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X, dtype=float)
    hover_text = []
    
    for i in range(len(y_range)):
        row_text = []
        for j in range(len(x_range)):
            x, y = X[i, j], Y[i, j]
            area_sqin = x * y
            area_sqft = area_sqin / 144
            meets_min = (x >= min_edge or y >= min_edge)
            max_dim = max(x, y)
            
            in_tech = (area_sqin <= tech_max_area and max_dim <= tech_max_edge and meets_min)
            in_core = (area_sqin <= core_max_area and max_dim <= core_max_edge and meets_min)
            
            if in_core:
                Z[i, j] = 2
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Core Range</b>")
            elif in_tech:
                Z[i, j] = 1
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>⚠️ Technical Limit</b>")
            else:
                Z[i, j] = 0
                if not meets_min:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Below minimum</b>")
                else:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Outside limits</b>")
        hover_text.append(row_text)
    
    fig.add_trace(go.Heatmap(
        x=x_range, y=y_range, z=Z,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False, hoverinfo='text', text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Helper function to build curve coordinates
    def build_annealed_curve(max_area, max_edge):
        curve_x, curve_y = [], []
        
        # Start from min_edge on x-axis, trace the hyperbolic curve
        for x in range(min_edge, min(int(max_edge) + 1, 151)):
            y = min(max_area / x, max_edge, 150)
            if y >= min_edge:
                curve_x.append(x)
                curve_y.append(y)
        
        # If we hit the max edge limit, continue along that line
        if curve_x and curve_y[-1] >= min_edge:
            # Go up the right edge to max_edge
            last_x = curve_x[-1]
            for y in range(int(curve_y[-1]), min(int(max_edge) + 1, 151)):
                curve_x.append(last_x)
                curve_y.append(y)
            
            # Go left along the top edge
            for x in range(int(last_x) - 1, min_edge - 1, -1):
                curve_x.append(x)
                curve_y.append(max_edge)
            
            # Go down the left edge back to where hyperbola meets it
            final_y = min(max_area / min_edge, max_edge)
            for y in range(int(max_edge), int(final_y), -1):
                curve_x.append(min_edge)
                curve_y.append(y)
        
        # Close the shape properly - fixed to not go through (0,0)
        # Go from last point down to x-axis at min_edge, then back to start
        curve_x.append(min_edge)
        curve_y.append(0)
        
        return curve_x, curve_y
    
    # Technical limit curve
    tech_curve_x, tech_curve_y = build_annealed_curve(tech_max_area, tech_max_edge)
    
    fig.add_trace(go.Scatter(
        x=tech_curve_x, y=tech_curve_y, fill='toself',
        fillcolor='rgba(255, 152, 0, 0.2)',
        line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
        name='Technical Limit', hoverinfo='skip'
    ))
    
    # Core range curve - only draw if different from technical limit
    if not limits_are_same:
        core_curve_x, core_curve_y = build_annealed_curve(core_max_area, core_max_edge)
        
        fig.add_trace(go.Scatter(
            x=core_curve_x, y=core_curve_y, fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(color='rgba(33, 150, 243, 1)', width=3),
            name='Core Range', hoverinfo='skip'
        ))
    else:
        # When limits are the same, fill with blue but don't draw a separate line
        core_curve_x, core_curve_y = build_annealed_curve(core_max_area, core_max_edge)
        
        fig.add_trace(go.Scatter(
            x=core_curve_x, y=core_curve_y, fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(width=0),  # No visible line - orange dashed will show instead
            name='Core Range', hoverinfo='skip', showlegend=True
        ))
    
    if custom_point:
        add_custom_point(fig, custom_point, min_edge, core_max_edge, core_max_area, tech_max_edge, tech_max_area, True)
    
    title_text = "AlpenGlass Sizing Limits - Annealed Glass"
    if filter_text:
        title_text += f"<br><sub>{filter_text}</sub>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title="Width (inches)", yaxis_title="Height (inches)",
        xaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', fixedrange=True, constrain='domain'),
        yaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1, fixedrange=True, constrain='domain'),
        plot_bgcolor='white', hovermode='closest', height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                   font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1)
    )
    return fig

def add_custom_point(fig, custom_point, min_edge, core_param1, core_param2, tech_param1, tech_param2, is_annealed):
    """Add custom size point to plot"""
    custom_width, custom_height = custom_point
    area_sqft = (custom_width * custom_height) / 144
    area_sqin = custom_width * custom_height
    meets_min = (custom_width >= min_edge or custom_height >= min_edge)
    
    if is_annealed:
        max_dim = max(custom_width, custom_height)
        in_tech = (area_sqin <= tech_param2 and max_dim <= tech_param1 and meets_min)
        in_core = (area_sqin <= core_param2 and max_dim <= core_param1 and meets_min)
    else:
        in_tech = ((custom_width <= tech_param1 and custom_height <= tech_param2) or 
                  (custom_width <= tech_param2 and custom_height <= tech_param1)) and meets_min
        in_core = ((custom_width <= core_param1 and custom_height <= core_param2) or 
                  (custom_width <= core_param2 and custom_height <= core_param1)) and meets_min
    
    if in_core:
        marker_color, status_text = 'rgb(0, 200, 0)', "✓ Within Core Range"
    elif in_tech:
        marker_color, status_text = 'rgb(255, 165, 0)', "⚠ Within Technical Limit"
    elif not meets_min:
        marker_color, status_text = 'rgb(255, 0, 0)', "✗ Below Minimum Size"
    else:
        marker_color, status_text = 'rgb(255, 0, 0)', "✗ Outside Technical Limits"
    
    fig.add_trace(go.Scatter(
        x=[custom_width], y=[custom_height], mode='markers+text',
        marker=dict(size=15, color=marker_color, symbol='star', line=dict(color='white', width=2)),
        text=[f"{custom_width}\" × {custom_height}\" ({area_sqft:.1f} sf)"],
        textposition="top center",
        textfont=dict(size=12, color=marker_color, family="Arial Black"),
        name='Your Size',
        hovertemplate=f"<b>Your Custom Size</b><br>Width: {custom_width}\"<br>Height: {custom_height}\"<br>Area: {area_sqft:.1f} sq ft<br>{status_text}<extra></extra>"
    ))

def main():
    tempered_df, annealed_df = load_data()
    
    if tempered_df is None or annealed_df is None:
        st.stop()
    
    glass_type = st.radio(
        "**Select Glass Type:**",
        options=["Tempered", "Annealed"],
        horizontal=True
    )
    
    df = tempered_df if glass_type == "Tempered" else annealed_df
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outer_lite_values = ['All'] + sorted(df['Outer Lites'].unique().tolist())
        outer_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Outer Lites'].unique().tolist())]
        outer_lite_display = st.selectbox("Outer Lites Thickness", outer_lite_labels)
        outer_lite = 'All' if outer_lite_display == 'All' else float(outer_lite_display.replace('mm', ''))
    
    with col2:
        inner_lite_values = ['All'] + sorted(df['Inner Lite(s)'].unique().tolist())
        inner_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Inner Lite(s)'].unique().tolist())]
        inner_lite_display = st.selectbox("Center Lite Thickness", inner_lite_labels)
        inner_lite = 'All' if inner_lite_display == 'All' else float(inner_lite_display.replace('mm', ''))
    
    st.markdown("---")
    st.markdown("### 🎯 Check Your Custom Size")
    
    size_col1, size_col2, size_col3 = st.columns([1, 1, 2])
    
    with size_col1:
        custom_width = st.number_input("Width (inches)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    
    with size_col2:
        custom_height = st.number_input("Height (inches)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    
    with size_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if custom_width > 0 and custom_height > 0:
            custom_area = (custom_width * custom_height) / 144
            st.info(f"**Custom Size:** {custom_width}\" × {custom_height}\" ({custom_area:.1f} sq ft)")
        else:
            st.caption("Enter dimensions to plot your custom size on the chart")
    
    st.markdown("---")
    
    filtered_df = df.copy()
    
    if outer_lite != 'All':
        filtered_df = filtered_df[filtered_df['Outer Lites'] == outer_lite]
    
    if inner_lite != 'All':
        filtered_df = filtered_df[filtered_df['Inner Lite(s)'] == inner_lite]
    
    if not filtered_df.empty:
        show_all_configs = (outer_lite == 'All' or inner_lite == 'All')
        
        if show_all_configs:
            st.subheader("Size Envelope")
            config_description = []
            if outer_lite != 'All':
                config_description.append(f"Outer Lites: {outer_lite}mm")
            if inner_lite != 'All':
                config_description.append(f"Center Lite: {inner_lite}mm")
            
            if config_description:
                st.caption(f"Filtered by: {', '.join(config_description)}")
            else:
                st.caption("Showing all available configurations")
            filter_text = ", ".join(config_description) if config_description else "All Configurations"
        else:
            config_name = filtered_df['Name'].values[0]
            st.subheader(f"Configuration: {config_name}")
            filter_text = f"Configuration: {config_name}"
        
        custom_point = (custom_width, custom_height) if custom_width > 0 and custom_height > 0 else None
        
        # Use first row for plotting dimensions
        plot_data = filtered_df.iloc[[0]]
        
        plot_col, specs_col = st.columns([2, 1])
        
        with plot_col:
            if glass_type == "Tempered":
                fig = create_tempered_plot(plot_data, show_all=show_all_configs, 
                                          all_configs_df=filtered_df if show_all_configs else None, 
                                          custom_point=custom_point, filter_text=filter_text)
            else:
                fig = create_annealed_plot(plot_data, show_all=show_all_configs,
                                          all_configs_df=filtered_df if show_all_configs else None,
                                          custom_point=custom_point, filter_text=filter_text)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Add annealed note
            if glass_type == "Annealed":
                st.info("**Note:** Annealed glass sizing based on wind load of DP30. Contact your sales rep if higher wind loads needed in your situation.")
        
        with specs_col:
            st.markdown("### Specifications")
            
            if glass_type == "Tempered":
                if show_all_configs:
                    core_long_max = filtered_df['CoreRange_ maxlongedge_inches'].max()
                    core_short_max = filtered_df['CoreRange_maxshortedge_inches'].max()
                    tech_long_max = filtered_df['Technical_limit_longedge_inches'].max()
                    tech_short_max = filtered_df['Technical_limit_shortedge_inches'].max()
                else:
                    core_long_max = filtered_df['CoreRange_ maxlongedge_inches'].values[0]
                    core_short_max = filtered_df['CoreRange_maxshortedge_inches'].values[0]
                    tech_long_max = filtered_df['Technical_limit_longedge_inches'].values[0]
                    tech_short_max = filtered_df['Technical_limit_shortedge_inches'].values[0]
                
                st.markdown("**Core Range**")
                st.info(f"Max Long Edge: **{core_long_max}\"**\nMax Short Edge: **{core_short_max}\"**")
                
                st.markdown("**Technical Limit**")
                st.warning(f"Max Long Edge: **{tech_long_max}\"**\nMax Short Edge: **{tech_short_max}\"**")
            
            else:  # Annealed - fixed column names
                if show_all_configs:
                    core_max_edge = filtered_df['CoreRange_maxedge_inches'].max()
                    tech_max_edge = filtered_df['Technical_limit_maxedge_inches'].max()
                    core_max_area = filtered_df['MaxArea_Core_squarefeet'].max()
                    tech_max_area = filtered_df['MaxArea_Technical_limit_squarefeet'].max()
                else:
                    core_max_edge = filtered_df['CoreRange_maxedge_inches'].values[0]
                    tech_max_edge = filtered_df['Technical_limit_maxedge_inches'].values[0]
                    core_max_area = filtered_df['MaxArea_Core_squarefeet'].values[0]
                    tech_max_area = filtered_df['MaxArea_Technical_limit_squarefeet'].values[0]
                
                st.markdown("**Core Range**")
                st.info(f"Max Edge: **{core_max_edge}\"**\nMax Area: **{core_max_area} sq ft**")
                
                st.markdown("**Technical Limit**")
                st.warning(f"Max Edge: **{tech_max_edge}\"**\nMax Area: **{tech_max_area} sq ft**")
            
            st.markdown("**Minimum Size**")
            st.error("At least one edge must be **16\"** or greater")
            
            if custom_point:
                st.markdown("---")
                st.markdown("### 🎯 Your Custom Size Status")
                
                custom_width, custom_height = custom_point
                meets_min = (custom_width >= 16 or custom_height >= 16)
                
                if glass_type == "Tempered":
                    in_tech = ((custom_width <= tech_long_max and custom_height <= tech_short_max) or 
                              (custom_width <= tech_short_max and custom_height <= tech_long_max)) and meets_min
                    in_core = ((custom_width <= core_long_max and custom_height <= core_short_max) or 
                              (custom_width <= core_short_max and custom_height <= core_long_max)) and meets_min
                else:
                    area_sqin = custom_width * custom_height
                    max_dim = max(custom_width, custom_height)
                    in_tech = (area_sqin <= tech_max_area * 144 and max_dim <= tech_max_edge and meets_min)
                    in_core = (area_sqin <= core_max_area * 144 and max_dim <= core_max_edge and meets_min)
                
                if in_core:
                    st.success("✓ **Within Core Range** - Standard pricing and lead time")
                elif in_tech:
                    st.warning("⚠ **Within Technical Limit** - May require special order and longer lead time")
                elif not meets_min:
                    st.error("✗ **Below Minimum Size** - At least one edge must be 16\" or greater")
                else:
                    st.error("✗ **Outside Technical Limits** - This size cannot be manufactured")
    else:
        st.error("No configuration found for the selected parameters.")

if __name__ == "__main__":
    main()
