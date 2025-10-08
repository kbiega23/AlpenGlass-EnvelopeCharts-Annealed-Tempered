AlpenGlass Window Size Envelope Visualizer
An interactive Streamlit application for visualizing maximum window sizes across different glass configurations, showing both core production ranges and technical limits.

Features
🎯 Interactive selection of glass configurations (outer lites, center lite, tempered/annealed)
📊 Visual envelope chart showing core range and technical limit overlays
🔄 Automatic handling of both portrait and landscape orientations
📏 Detailed specifications display
🎨 Clean, professional interface
Installation
Prerequisites
Python 3.8 or higher
pip (Python package installer)
Setup Steps
Clone the repository

git clone https://github.com/yourusername/alpenglass-visualizer.git
cd alpenglass-visualizer
Create a virtual environment (recommended)

python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
Install dependencies

pip install -r requirements.txt
Add your data file

Place your AlpenGlass max sizing data.xlsx file in the project root directory
The file should contain columns: Name, Tempered or Annealed, Outer Lites, Inner Lite, CoreRange_ maxlongedge, CoreRange_maxshortedge, Technical limit_long edge, Technical limit_short edge
Running the Application
Locally
streamlit run app.py
The application will open in your default web browser at http://localhost:8501

Deploy to Streamlit Cloud
Push to GitHub

git add .
git commit -m "Initial commit"
git push origin main
Deploy on Streamlit Cloud

Go to share.streamlit.io
Sign in with your GitHub account
Click "New app"
Select your repository, branch (main), and main file (app.py)
Click "Deploy"
Upload your data file

After deployment, you can upload the Excel file through the Streamlit Cloud interface
Or include it in your GitHub repository (if data is not sensitive)
Data Format
Your Excel file should have the following structure:

Name	Tempered or Annealed	Outer Lites	Inner Lite	CoreRange_ maxlongedge	CoreRange_maxshortedge	Technical limit_long edge	Technical limit_short edge
3mm-0.5mm-3mm	Tempered	3	0.5	80	36	80	36
5mm-1.1mm-5mm	Tempered	5	1.1	100	60	120	60
Project Structure
alpenglass-visualizer/
│
├── app.py                              # Main Streamlit application
├── AlpenGlass max sizing data.xlsx     # Your data file
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore file
Usage
Select Configuration

Choose Outer Lites thickness (3mm, 5mm, or 6mm)
Choose Center Lite thickness (0.5mm, 1.1mm, or 1.3mm)
Choose Glass Treatment (Tempered or Annealed)
View Results

The chart shows the operating envelope with both orientations
Blue area = Core Range (efficient, low cost)
Orange area = Technical Limit (possible, but premium cost)
Hover over the chart to see exact dimensions
Review Specifications

Right panel shows detailed specifications for both ranges
Includes maximum dimensions and areas
Customization
Updating Data
Simply replace AlpenGlass max sizing data.xlsx with your updated file (maintaining the same column structure) and restart the app.

Modifying Colors
In app.py, you can modify the colors in the create_envelope_plot function:

Core Range: fillcolor='rgba(33, 150, 243, 0.3)' (blue)
Technical Limit: fillcolor='rgba(255, 152, 0, 0.2)' (orange)
Adding More Glass Types
If you add data for annealed glass or other configurations, the app will automatically detect and include them in the dropdown selectors.

Troubleshooting
Issue: "AlpenGlass max sizing data.xlsx not found"

Ensure the Excel file is in the same directory as app.py
Check the filename matches exactly (case-sensitive)
Issue: App doesn't show any configurations

Verify your Excel file has the correct column names
Check that data types are correct (numbers for dimensions)
Issue: Chart looks incorrect

Verify that long edge values are ≥ short edge values in your data
Check that all dimension values are positive numbers
Support
For issues, questions, or contributions, please open an issue on the GitHub repository.

License
This project is proprietary to AlpenGlass. All rights reserved.
