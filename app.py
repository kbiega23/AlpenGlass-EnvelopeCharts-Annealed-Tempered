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

**Understanding the Visualization:**
- **Core Range** (blue): Efficient, low-cost production range
- **Technical Limit** (orange): Maximum physically achievable size (may require special order and longer lead time)
- **Minimum Size**: At least one edge must be 16" or greater
- **White areas**: Do not meet minimum size requirements

**Configuration Selection:**
- **Select "All"**: View the composite envelope showing the maximum achievable sizes across all configurations in your filter. This shows you the outer boundaries of what's possible.
- **Select Specific Values**: View the exact size limits for a particular glass configuration (specific outer lite thickness, center lite thickness, and treatment combination).

**Checking Your Custom Size:**
1. Use the dropdowns below to filter by glass specifications (or leave as "All")
2. Enter your desired width and height in the custom size input fields
3. A star will appear on the chart showing your size's location
4. Check the status indicator to see if it falls within Core Range, Technical Limit, or outside our capabilities

**Interpreting the Chart:**
- Hover over any point to see exact dimensions and area
- Blue and orange labels mark the corner limits for each range
- The chart displays both portrait and landscape orientations
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the glass configuration data from Excel file"""
    import os
    
    # List of possible filenames to try
    possible_names = [
        'AlpenGlass max sizing data.xlsx',
        'AlpenGlass_max_sizing_data.xlsx',
        'alpenglass_max_sizing_data.xlsx',
    ]
    
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename)
                
                # Standardize column names to handle inconsistencies
                column_mapping = {
                    'CoreRange_ maxlongedge': 'CoreRange_ maxlongedge_inches',
                    'CoreRange_maxshortedge': 'CoreRange_maxshortedge_inches',
                    'Technical limit_long edge': 'Technical_limit_long edge_inches',
                    'Technical limit_short edge': 'Technical_limit_short edge_inches'
                }
                df = df.rename(columns=column_mapping)
                
                return df
            except Exception as e:
                st.error(f"Error reading {filename}: {str(e)}")
                return None
    
    st.error("Excel file not found. Please ensure 'AlpenGlass max sizing data.xlsx' is in your GitHub repository.")
    return None

# Create the envelope visualization
def create_envelope_plot(config_data, min_edge=16, show_all=False, all_configs_df=None, custom_point=None, filter_text=""):
    """Create a plotly figure showing the core range and technical limit envelopes"""
    
    if config_data.empty:
        return None
    
    # Extract values for bounds
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        core_long = all_configs_df['CoreRange_ maxlongedge_inches'].max()
        core_short = all_configs_df['CoreRange_maxshortedge_inches'].max()
        tech_long = all_configs_df['Technical_limit_long edge_inches'].max()
        tech_short = all_configs_df['Technical_limit_short edge_inches'].max()
    else:
        core_long = config_data['CoreRange_ maxlongedge_inches'].values[0]
        core_short = config_data['CoreRange_maxshortedge_inches'].values[0]
        tech_long = config_data['Technical_limit_long edge_inches'].values[0]
        tech_short = config_data['Technical_limit_short edge_inches'].values[0]
    
    fig = go.Figure()
    
    # Create a grid snapped to 1" increments for hover - fixed to 0-150"
    x_range = np.arange(0, 151, 1)
    y_range = np.arange(0, 151, 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Determine which region each point is in
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
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>⚠️ Extra charges may apply for extreme sizes</b>")
            else:
                Z[i, j] = 0
                if not meets_min:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Below minimum size</b>")
                else:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Outside technical limits</b>")
        hover_text.append(row_text)
    
    # Add invisible heatmap for hover functionality
    fig.add_trace(go.Heatmap(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False,
        hoverinfo='text',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Technical Limit envelope
    tech_x = [min_edge, tech_long, tech_long, tech_short, tech_short, 0, 0, min_edge, min_edge]
    tech_y = [0, 0, tech_short, tech_short, tech_long, tech_long, min_edge, min_edge, 0]
    
    fig.add_trace(go.Scatter(
        x=tech_x,
        y=tech_y,
        fill='toself',
        fillcolor='rgba(255, 152, 0, 0.2)',
        line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
        name='Technical Limit',
        hoverinfo='skip'
    ))
    
    # Core Range envelope
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        all_x = []
        all_y = []
        
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
            x=all_x,
            y=all_y,
            fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(width=0),
            name='Core Range',
            hoverinfo='skip'
        ))
    else:
        core_x = [min_edge, core_long, core_long, core_short, core_short, 0, 0, min_edge, min_edge]
        core_y = [0, 0, core_short, core_short, core_long, core_long, min_edge, min_edge, 0]
        
        fig.add_trace(go.Scatter(
            x=core_x,
            y=core_y,
            fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(color='rgba(33, 150, 243, 1)', width=3),
            name='Core Range',
            hoverinfo='skip'
        ))
    
    # Add custom point if provided
    if custom_point is not None:
        custom_width, custom_height = custom_point
        
        meets_min = (custom_width >= min_edge or custom_height >= min_edge)
        in_tech = ((custom_width <= tech_long and custom_height <= tech_short) or 
                  (custom_width <= tech_short and custom_height <= tech_long)) and meets_min
        in_core = ((custom_width <= core_long and custom_height <= core_short) or 
                  (custom_width <= core_short and custom_height <= core_long)) and meets_min
        
        if in_core:
            marker_color = 'rgb(0, 200, 0)'
            status_text = "✓ Within Core Range"
        elif in_tech:
            marker_color = 'rgb(255, 165, 0)'
            status_text = "⚠ Within Technical Limit (Premium)"
        elif not meets_min:
            marker_color = 'rgb(255, 0, 0)'
            status_text = "✗ Below Minimum Size"
        else:
            marker_color = 'rgb(255, 0, 0)'
            status_text = "✗ Outside Technical Limits"
        
        area_sqft = (custom_width * custom_height) / 144
        
        fig.add_trace(go.Scatter(
            x=[custom_width],
            y=[custom_height],
            mode='markers+text',
            marker=dict(
                size=15,
                color=marker_color,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            text=[f"{custom_width}\" × {custom_height}\" ({area_sqft:.1f} sf)"],
            textposition="top center",
            textfont=dict(size=12, color=marker_color, family="Arial Black"),
            name='Your Size',
            hovertemplate=f"<b>Your Custom Size</b><br>Width: {custom_width}\"<br>Height: {custom_height}\"<br>Area: {area_sqft:.1f} sq ft<br>{status_text}<extra></extra>"
        ))
    
    # Add corner labels
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        annotations = []
        
        core_corners_set = set()
        tech_corners_set = set()
        
        for idx, row in all_configs_df.iterrows():
            c_long = row['CoreRange_ maxlongedge_inches']
            c_short = row['CoreRange_maxshortedge_inches']
            t_long = row['Technical_limit_long edge_inches']
            t_short = row['Technical_limit_short edge_inches']
            
            core_corners_set.add((c_long, c_short))
            core_corners_set.add((c_short, c_long))
            tech_corners_set.add((t_long, t_short))
            tech_corners_set.add((t_short, t_long))
        
        core_corners = sorted(list(core_corners_set), key=lambda p: (p[0], p[1]), reverse=True)
        tech_corners = sorted(list(tech_corners_set), key=lambda p: (p[0], p[1]), reverse=True)
        
        # Find Pareto frontier for core
        core_frontier = []
        for x, y in core_corners:
            is_dominated = False
            for x2, y2 in core_corners:
                if x2 > x and y2 > y:
                    is_dominated = True
                    break
            if not is_dominated:
                core_frontier.append((x, y))
        
        # Add BLUE labels for CORE frontier points
        for i, (x, y) in enumerate(core_frontier[:4]):
            annotations.append(
                dict(x=x, y=y, 
                     text=f"{int(x)}\" × {int(y)}\"<br>{(x*y)/144:.1f} sq ft",
                     showarrow=True, arrowhead=2, 
                     ax=20 if x >= y else -20, 
                     ay=-20 if x >= y else 20,
                     arrowcolor="rgba(33, 150, 243, 1)",
                     bgcolor="rgba(33, 150, 243, 0.8)", 
                     font=dict(color="white", size=10))
            )
        
        # Find Pareto frontier for tech
        tech_frontier = []
        for x, y in tech_corners:
            is_dominated = False
            for x2, y2 in tech_corners:
                if x2 > x and y2 > y:
                    is_dominated = True
                    break
            if not is_dominated:
                tech_frontier.append((x, y))
        
        # Add ORANGE labels for TECH frontier points
        for i, (x, y) in enumerate(tech_frontier[:4]):
            annotations.append(
                dict(x=x, y=y, 
                     text=f"{int(x)}\" × {int(y)}\"<br>{(x*y)/144:.1f} sq ft",
                     showarrow=True, arrowhead=2, 
                     ax=30 if x >= y else -30, 
                     ay=-30 if x >= y else 30,
                     arrowcolor="rgba(255, 152, 0, 1)",
                     bgcolor="rgba(255, 152, 0, 0.8)", 
                     font=dict(color="white", size=10))
            )
    else:
        annotations = [
            dict(x=core_long, y=core_short, 
                 text=f"{int(core_long)}\" × {int(core_short)}\"<br>{(core_long*core_short)/144:.1f} sq ft",
                 showarrow=True, arrowhead=2, ax=20, ay=-20,
                 arrowcolor="rgba(33, 150, 243, 1)",
                 bgcolor="rgba(33, 150, 243, 0.8)", font=dict(color="white", size=10)),
            dict(x=core_short, y=core_long, 
                 text=f"{int(core_short)}\" × {int(core_long)}\"<br>{(core_short*core_long)/144:.1f} sq ft",
                 showarrow=True, arrowhead=2, ax=-20, ay=20,
                 arrowcolor="rgba(33, 150, 243, 1)",
                 bgcolor="rgba(33, 150, 243, 0.8)", font=dict(color="white", size=10)),
            dict(x=tech_long, y=tech_short, 
                 text=f"{int(tech_long)}\" × {int(tech_short)}\"<br>{(tech_long*tech_short)/144:.1f} sq ft",
                 showarrow=True, arrowhead=2, ax=30, ay=-30,
                 arrowcolor="rgba(255, 152, 0, 1)",
                 bgcolor="rgba(255, 152, 0, 0.8)", font=dict(color="white", size=10)),
            dict(x=tech_short, y=tech_long, 
                 text=f"{int(tech_short)}\" × {int(tech_long)}\"<br>{(tech_short*tech_long)/144:.1f} sq ft",
                 showarrow=True, arrowhead=2, ax=-30, ay=30,
                 arrowcolor="rgba(255, 152, 0, 1)",
                 bgcolor="rgba(255, 152, 0, 0.8)", font=dict(color="white", size=10))
        ]
    
    # Update layout - fixed to 0-150" range
    # Add title with filter information for PNG export
    title_text = "AlpenGlass Sizing Limits"
    if filter_text:
        title_text += f"<br><sub>{filter_text}</sub>"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Width (inches)",
        yaxis_title="Height (inches)",
        xaxis=dict(
            range=[0, 150], 
            showgrid=True, 
            gridcolor='lightgray',
            fixedrange=True,
            constrain='domain'
        ),
        yaxis=dict(
            range=[0, 150], 
            showgrid=True, 
            gridcolor='lightgray', 
            scaleanchor="x", 
            scaleratio=1,
            fixedrange=True,
            constrain='domain'
        ),
        plot_bgcolor='white',
        hovermode='closest',
        height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        annotations=annotations,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        )
    )
    
    return fig

# Main app logic
def main():
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Create three columns for selectors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        outer_lite_values = ['All'] + sorted(df['Outer Lites'].unique().tolist())
        outer_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Outer Lites'].unique().tolist())]
        outer_lite_display = st.selectbox(
            "Outer Lites Thickness",
            outer_lite_labels,
            key="outer_lite_select"
        )
        outer_lite = 'All' if outer_lite_display == 'All' else float(outer_lite_display.replace('mm', ''))
    
    with col2:
        inner_lite_values = ['All'] + sorted(df['Inner Lite'].unique().tolist())
        inner_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Inner Lite'].unique().tolist())]
        inner_lite_display = st.selectbox(
            "Center Lite Thickness",
            inner_lite_labels,
            key="inner_lite_select"
        )
        inner_lite = 'All' if inner_lite_display == 'All' else float(inner_lite_display.replace('mm', ''))
    
    with col3:
        tempered_options = ['All'] + sorted(df['Tempered or Annealed'].unique().tolist())
        tempered = st.selectbox(
            "Glass Treatment",
            tempered_options,
            key="treatment_select"
        )
    
    # Add custom size input section
    st.markdown("---")
    st.markdown("### 🎯 Check Your Custom Size")
    
    size_col1, size_col2, size_col3 = st.columns([1, 1, 2])
    
    with size_col1:
        custom_width = st.number_input(
            "Width (inches)",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0,
            key="custom_width"
        )
    
    with size_col2:
        custom_height = st.number_input(
            "Height (inches)",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0,
            key="custom_height"
        )
    
    with size_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if custom_width > 0 and custom_height > 0:
            custom_area = (custom_width * custom_height) / 144
            st.info(f"**Custom Size:** {custom_width}\" × {custom_height}\" ({custom_area:.1f} sq ft)")
        else:
            st.caption("Enter dimensions to plot your custom size on the chart")
    
    st.markdown("---")
    
    # Filter data based on selection
    filtered_df = df.copy()
    
    if outer_lite != 'All':
        filtered_df = filtered_df[filtered_df['Outer Lites'] == outer_lite]
    
    if inner_lite != 'All':
        filtered_df = filtered_df[filtered_df['Inner Lite'] == inner_lite]
    
    if tempered != 'All':
        filtered_df = filtered_df[filtered_df['Tempered or Annealed'] == tempered]
    
    # Display configuration info
    if not filtered_df.empty:
        show_all_configs = (outer_lite == 'All' or inner_lite == 'All' or tempered == 'All')
        
        if show_all_configs:
            st.subheader("Size Envelope")
            config_description = []
            if outer_lite != 'All':
                config_description.append(f"Outer Lites: {outer_lite}mm")
            if inner_lite != 'All':
                config_description.append(f"Center Lite: {inner_lite}mm")
            if tempered != 'All':
                config_description.append(f"Treatment: {tempered}")
            
            if config_description:
                st.caption(f"Filtered by: {', '.join(config_description)}")
            else:
                st.caption("Showing all available configurations")
        else:
            config_name = filtered_df['Name'].values[0]
            st.subheader(f"Configuration: {config_name}")
        
        # Determine how to calculate max dimensions
        if outer_lite == 'All' or inner_lite == 'All' or tempered == 'All':
            filtered_df['core_area'] = filtered_df['CoreRange_ maxlongedge_inches'] * filtered_df['CoreRange_maxshortedge_inches']
            filtered_df['tech_area'] = filtered_df['Technical_limit_long edge_inches'] * filtered_df['Technical_limit_short edge_inches']
            
            max_core_idx = filtered_df['core_area'].idxmax()
            core_long_max = filtered_df.loc[max_core_idx, 'CoreRange_ maxlongedge_inches']
            core_short_max = filtered_df.loc[max_core_idx, 'CoreRange_maxshortedge_inches']
            
            max_tech_idx = filtered_df['tech_area'].idxmax()
            tech_long_max = filtered_df.loc[max_tech_idx, 'Technical_limit_long edge_inches']
            tech_short_max = filtered_df.loc[max_tech_idx, 'Technical_limit_short edge_inches']
        else:
            core_long_max = filtered_df['CoreRange_ maxlongedge_inches'].values[0]
            core_short_max = filtered_df['CoreRange_maxshortedge_inches'].values[0]
            tech_long_max = filtered_df['Technical_limit_long edge_inches'].values[0]
            tech_short_max = filtered_df['Technical_limit_short edge_inches'].values[0]
        
        plot_data = pd.DataFrame([{
            'CoreRange_ maxlongedge_inches': core_long_max,
            'CoreRange_maxshortedge_inches': core_short_max,
            'Technical_limit_long edge_inches': tech_long_max,
            'Technical_limit_short edge_inches': tech_short_max
        }])
        
        custom_point = None
        if custom_width > 0 and custom_height > 0:
            custom_point = (custom_width, custom_height)
        
        # Create filter text for chart title (to show in PNG export)
        filter_parts = []
        if show_all_configs:
            if outer_lite != 'All':
                filter_parts.append(f"Outer Lites: {outer_lite}mm")
            if inner_lite != 'All':
                filter_parts.append(f"Center Lite: {inner_lite}mm")
            if tempered != 'All':
                filter_parts.append(f"Treatment: {tempered}")
            
            if filter_parts:
                filter_text = "Filtered by: " + ", ".join(filter_parts)
            else:
                filter_text = "All Configurations"
        else:
            filter_text = f"Configuration: {filtered_df['Name'].values[0]}"
        
        # Create two columns for the plot and specifications
        plot_col, specs_col = st.columns([2, 1])
        
        with plot_col:
            fig = create_envelope_plot(plot_data, show_all=show_all_configs, all_configs_df=filtered_df if show_all_configs else None, custom_point=custom_point, filter_text=filter_text)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with specs_col:
            st.markdown("### Specifications")
            
            st.markdown("**Core Range** (Efficient Production)")
            if show_all_configs:
                st.info(f"""
                - Maximum Long Edge: **{core_long_max}\"**
                - Maximum Short Edge: **{core_short_max}\"**
                - Showing composite of {len(filtered_df)} configuration(s)
                - Envelope represents achievable dimensions across all configs
                """)
            else:
                st.info(f"""
                - Maximum Long Edge: **{core_long_max}\"**
                - Maximum Short Edge: **{core_short_max}\"**
                - Max Size: **{core_long_max}\" × {core_short_max}\"**
                - Max Area: **{core_long_max * core_short_max} sq in** ({(core_long_max * core_short_max)/144:.1f} sq ft)
                """)
            
            st.markdown("**Technical Limit** (Special Order)")
            st.warning(f"""
            - Maximum Long Edge: **{tech_long_max}\"**
            - Maximum Short Edge: **{tech_short_max}\"**
            - Max Size: **{tech_long_max}\" × {tech_short_max}\"**
            - Max Area: **{tech_long_max * tech_short_max} sq in** ({(tech_long_max * tech_short_max)/144:.1f} sq ft)
            - May require special order and longer lead time
            """)
            
            st.markdown("**Minimum Size Constraint**")
            st.error("""
            - At least one edge must be **16\"** or greater
            """)
            
            if custom_point is not None:
                st.markdown("---")
                st.markdown("### 🎯 Your Custom Size Status")
                
                custom_width, custom_height = custom_point
                meets_min = (custom_width >= 16 or custom_height >= 16)
                in_tech = ((custom_width <= tech_long_max and custom_height <= tech_short_max) or 
                          (custom_width <= tech_short_max and custom_height <= tech_long_max)) and meets_min
                in_core = ((custom_width <= core_long_max and custom_height <= core_short_max) or 
                          (custom_width <= core_short_max and custom_height <= core_long_max)) and meets_min
                
                if in_core:
                    st.success("✓ **Within Core Range** - Standard pricing and lead time")
                elif in_tech:
                    st.warning("⚠ **Within Technical Limit** - May require special order and longer lead time")
                elif not meets_min:
                    st.error("✗ **Below Minimum Size** - At least one edge must be 16\" or greater")
                else:
                    st.error("✗ **Outside Technical Limits** - This size cannot be manufactured")
            
            # Additional notes
            st.markdown("---")
            st.markdown("### 📝 Notes")
            
            if show_all_configs:
                st.markdown("""
                - **Composite envelope**: Shows the union of all matching configurations
                - **True capabilities**: Outer boundary shows actual maximum achievable in any dimension
                - Example: Can achieve 120"×59" OR 100"×72", creating an L-shaped envelope
                - Hover over chart to see dimensions at any point
                - Select specific values to see one configuration
                """)
            else:
                st.markdown("""
                - Hover over the chart to see dimensions and pricing info
                - Hover snaps to nearest 1" increment
                - White areas do not meet minimum size requirements
                - The chart shows both possible orientations (portrait and landscape)
                - Core Range represents the most cost-effective production envelope
                - Technical Limit sizes will incur a cost premium
                - Select "All" in any dropdown to see overall maximum sizes
                """)
    else:
        st.error("No configuration found for the selected parameters. Please check your data file.")

if __name__ == "__main__":
    main()
