import re

with open('src/pages/Dashboard.js', 'r') as f:
    content = f.read()

# We want to extract everything from `  // Animation variants` down to the end of `AdvancedAnalyticsPage`
# and place it before `const Dashboard = () => {`
start_marker = "  // Animation variants"
end_marker = "  const ToastNotifications = () => ("

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

extracted = content[start_idx:end_idx]
extracted = extracted.replace("  const containerVariants", "const containerVariants")
extracted = extracted.replace("  const itemVariants", "const itemVariants")
extracted = extracted.replace("  const NavIcon", "const NavIcon")
extracted = extracted.replace("  const OverviewPage", "const OverviewPage")
extracted = extracted.replace("  const PredictorInputForm", "const PredictorInputForm")
extracted = extracted.replace("  const PredictorPage", "const PredictorPage = ({ network }) => {")
extracted = extracted.replace("  const PredictorPage = () => {", "const PredictorPage = ({ network }) => {") # Fix replacement just in case
extracted = extracted.replace("  const RegionalPage", "const RegionalPage")
extracted = extracted.replace("  const HubPage", "const HubPage")
extracted = extracted.replace("  const ConfigPage", "const ConfigPage")
extracted = extracted.replace("  const AdvancedAnalyticsPage", "const AdvancedAnalyticsPage")

# Remove leading indents for these top-level components (just roughly)
lines = extracted.split('\n')
new_lines = []
for line in lines:
    if line.startswith('  '):
        new_lines.append(line[2:])
    else:
        new_lines.append(line)
extracted_clean = '\n'.join(new_lines)

# Fix PredictorPage definition if it wasn't caught
extracted_clean = extracted_clean.replace("const PredictorPage = () => {", "const PredictorPage = ({ network }) => {")

dashboard_def = "const Dashboard = () => {\n"
dash_start = content.find(dashboard_def)

# Reconstruct
new_content = content[:dash_start] + extracted_clean + "\n" + dashboard_def + content[dash_start+len(dashboard_def):start_idx] + content[end_idx:]

# Also update <PredictorPage /> in Dashboard to <PredictorPage network={network} />
new_content = new_content.replace("<Route path=\"/predictor\" element={<PredictorPage />} />", "<Route path=\"/predictor\" element={<PredictorPage network={network} />} />")
# And NavIcon needs useLocation internally now since it's extracted
navicon_def = """const NavIcon = ({ icon, to, tooltip }) => {
  const location = useLocation();"""
new_content = new_content.replace("const NavIcon = ({ icon, to, tooltip }) => {", navicon_def)

with open('src/pages/Dashboard.js', 'w') as f:
    f.write(new_content)

print("Dashboard refactored!")
