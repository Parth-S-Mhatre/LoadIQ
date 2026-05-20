import axios from 'axios';

const GROQ_API_KEY = process.env.REACT_APP_GROQ_API_KEY || '';
const GROQ_MODEL = process.env.REACT_APP_GROQ_MODEL || 'llama-3.3-70b-versatile';
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

const ELECTRICITY_PATTERNS = {
  Germany: 'German electricity consumption patterns based on historical data.',
  UK: 'UK electricity usage trends and peak hours.',
  USA: 'USA electricity demand and supply patterns.',
  India: 'India electricity consumption with regional variations.'
};

const formatContextBlock = (context) => {
  if (!context) {
    return 'No recent dashboard or forecast context was provided.';
  }

  const sections = [];

  if (context.workspaceName || context.organization) {
    sections.push(
      `Workspace: ${context.workspaceName || 'Not set'}\nOrganization: ${context.organization || 'Not set'}`
    );
  }

  if (context.latestPrediction) {
    sections.push(
      `Latest predictor session:\nHorizon: ${context.latestPrediction.forecastHorizon || 'n/a'} hours\nNext hour prediction: ${context.latestPrediction.nextHourPrediction || 'n/a'} MW\nLatest step prediction: ${context.latestPrediction.latestPrediction || 'n/a'} MW\nSource: ${context.latestPrediction.predictionSource || 'unknown'}\nFallback used: ${context.latestPrediction.fallbackUsed ? 'yes' : 'no'}`
    );
  }

  if (context.overview) {
    sections.push(
      `Overview telemetry:\nPredicted load: ${context.overview.predictedLoad || 'n/a'} MW\nLatest forecast: ${context.overview.latestForecast || 'n/a'} MW\nPrediction source: ${context.overview.predictionSource || 'unknown'}`
    );
  }

  if (context.advancedAnalytics) {
    sections.push(
      `Advanced analytics:\nBase load: ${context.advancedAnalytics.baseLoad || 'n/a'} MW\nRenewable share: ${context.advancedAnalytics.renewablePercent || 'n/a'}%\nPrediction source: ${context.advancedAnalytics.predictionSource || 'unknown'}\nFallback used: ${context.advancedAnalytics.fallbackUsed ? 'yes' : 'no'}`
    );
  }

  if (Array.isArray(context.recentHistory) && context.recentHistory.length) {
    const recentHistoryText = context.recentHistory
      .slice(0, 3)
      .map((item, index) => `${index + 1}. ${item.timestamp || 'unknown time'} - next hour ${item.nextHourPrediction || 'n/a'} MW, horizon ${item.forecastHorizon || 'n/a'}h`)
      .join('\n');
    sections.push(`Recent saved history:\n${recentHistoryText}`);
  }

  return sections.length ? sections.join('\n\n') : 'No recent dashboard or forecast context was provided.';
};

export const ChatbotService = {
  /**
   * Send a message to the Groq API and get a response
   * @param {string} message - The user's message
   * @param {string} country - The country context (e.g., 'Germany', 'UK', 'USA', 'India')
   * @param {object} context - Latest dashboard/prediction context
   * @returns {Promise<string>} - The chatbot's response
   */
  sendMessage: async (message, country, context = null) => {
    try {
      if (!GROQ_API_KEY) {
        throw new Error('Groq API key is missing.');
      }

      const systemPrompt = ChatbotService.getSystemPrompt();
      const contextBlock = formatContextBlock(context);
      const response = await axios.post(GROQ_API_URL, {
        model: GROQ_MODEL,
        messages: [
          {
            role: 'system',
            content: systemPrompt
          },
          {
            role: 'user',
            content: `Country: ${country}\n\nApp context:\n${contextBlock}\n\nQuestion: ${message}`
          }
        ],
        temperature: 0.4,
        max_tokens: 512
      }, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${GROQ_API_KEY}`
        }
      });

      if (Array.isArray(response.data?.choices) && response.data.choices.length > 0) {
        const content = response.data.choices[0]?.message?.content;
        if (typeof content === 'string' && content.trim()) {
          return content.trim();
        }
      }
      throw new Error('Unexpected response format from Groq API');
    } catch (error) {
      console.error('Error communicating with Groq API:', error);
      const apiMessage = error.response?.data?.error?.message;
      throw new Error(apiMessage || 'Failed to get a response from the chatbot.');
    }
  },

  getSystemPrompt: () => {
    return `You are LoadIQ's advanced Energy Analytics Chatbot. You have been trained on German datasets but your knowledge has been extended with the following electricity patterns:
    
    1. Germany (DE): Heavy reliance on renewables (solar & wind). Pronounced midday solar peaks and high wind generation volatility in northern regions. Grid interconnectivity is strong but suffers from north-to-south transmission bottlenecks.
    2. United Kingdom (UK): Island grid with significant offshore wind. High variability. Evening peak is very pronounced (tea-time spike). Interconnectors to France and Norway play a crucial role in balancing.
    3. United States (USA): Highly fragmented grids (ERCOT, PJM, CAISO, etc.). Summer cooling drives massive afternoon peaks in southern states. California has the famous "Duck Curve" due to extreme solar penetration midday and steep evening ramp-ups.
    4. India (IN): Rapidly growing demand driven by cooling and industrialization. Evening peaks are dominant. Heavy reliance on coal with rapidly accelerating solar capacity. Frequency fluctuations are more common than in Europe.

    When users ask questions, use this context to answer intelligently. You can compare countries, analyze load patterns, and provide insights based on provided prediction results.
    Treat app context as the most recent source of truth when it is available, and refer to those results directly in your answer.
    Keep your answers concise, professional, and directly related to energy analytics. Use markdown formatting to make your answers readable.`;
  },

  /**
   * Get electricity patterns for a specific country
   * @param {string} country - The country name
   * @returns {string} - The electricity pattern description
   */
  getElectricityPattern: (country) => {
    return ELECTRICITY_PATTERNS[country] || 'No data available for the selected country.';
  }
};
