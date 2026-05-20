import re

with open('src/App.js', 'r') as f:
    content = f.read()

if "import EnergyChatbot" not in content:
    content = content.replace("import Dashboard", "import Dashboard\nimport EnergyChatbot from './components/EnergyChatbot';")

if "<EnergyChatbot />" not in content:
    # Add it right before the closing router tag, or right before </div> of the app root
    content = content.replace("      </AuthProvider>\n    </Router>\n  );\n}", "        <EnergyChatbot />\n      </AuthProvider>\n    </Router>\n  );\n}")

with open('src/App.js', 'w') as f:
    f.write(content)

print("Chatbot added to App.js")
