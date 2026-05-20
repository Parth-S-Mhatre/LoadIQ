import re

with open('src/pages/LoadPredictor.js', 'r') as f:
    content = f.read()

# Add UserHistoryService import if not present
if "import { UserHistoryService }" not in content:
    content = content.replace("import { AnalyticsService }", "import { AnalyticsService }\nimport { UserHistoryService }")

# Find where it saves to localStorage
old_save_block = """      // Save to analytics history in localStorage
      const historyItem = {
        timestamp: new Date().toISOString(),
        prediction: newPreds[newPreds.length - 1],
        forecastHorizon,
        inputData: historicalLoads
      };

      const existingHistory = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
      existingHistory.unshift(historyItem);
      localStorage.setItem(`analyticsHistory_${user?.uid}`, JSON.stringify(existingHistory.slice(0, 50))); // Keep last 50

      // NoSQL disabled: keep history in localStorage only."""

new_save_block = """      // Save to analytics history in localStorage and Firebase NoSQL
      const historyItem = {
        timestamp: new Date().toISOString(),
        prediction: newPreds[newPreds.length - 1],
        forecastHorizon,
        inputData: historicalLoads
      };

      const existingHistory = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
      existingHistory.unshift(historyItem);
      localStorage.setItem(`analyticsHistory_${user?.uid}`, JSON.stringify(existingHistory.slice(0, 50)));

      // Save to Firebase NoSQL
      if (user?.uid) {
        try {
          await UserHistoryService.savePrediction(user.uid, {
            forecastHorizon,
            predictions: newPreds,
            historicalLoads: historicalLoads,
            average: 0,
            peak: Math.max(...newPreds),
            minimum: Math.min(...newPreds),
            delta: newPreds[newPreds.length - 1] - historicalLoads[historicalLoads.length - 1],
            direction: newPreds[newPreds.length - 1] > historicalLoads[historicalLoads.length - 1] ? 'up' : 'down'
          });
          console.log("Saved prediction to Firebase NoSQL successfully");
        } catch (dbError) {
          console.error("Failed to save to Firebase:", dbError);
        }
      }"""

content = content.replace(old_save_block, new_save_block)

with open('src/pages/LoadPredictor.js', 'w') as f:
    f.write(content)

print("LoadPredictor updated!")

with open('src/pages/UserProfile.js', 'r') as f:
    up_content = f.read()

# Add UserHistoryService import
if "import { UserHistoryService }" not in up_content:
    up_content = up_content.replace("import { useAuth }", "import { useAuth }\nimport { UserHistoryService }")

# Find the useEffect block
old_effect = """    // Load analytics history
    const history = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
    setAnalyticsHistory(history);"""

new_effect = """    // Load analytics history from Firebase NoSQL
    const loadHistory = async () => {
      if (user?.uid) {
        const history = await UserHistoryService.getUserAnalytics(user.uid);
        // Fallback to localStorage if NoSQL is empty
        if (history.length === 0) {
          const local = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
          setAnalyticsHistory(local);
        } else {
          // Map Firebase data to UI structure
          setAnalyticsHistory(history.map(item => ({
            timestamp: item.timestamp?.toDate ? item.timestamp.toDate() : item.timestamp,
            prediction: item.predictions ? item.predictions[item.predictions.length - 1] : 0
          })));
        }
      }
    };
    loadHistory();"""

up_content = up_content.replace(old_effect, new_effect)

with open('src/pages/UserProfile.js', 'w') as f:
    f.write(up_content)

print("UserProfile updated!")

