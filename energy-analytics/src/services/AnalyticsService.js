import { API_CONFIG, buildModel1Url, buildModel2Url, fetchWithRetry } from '../config/api';

export const AnalyticsService = {
  extractPredictionItem: (item, defaultSource = 'ml_model') => {
    if (typeof item === 'number') {
      return {
        value: Number(item),
        predictionSource: defaultSource,
        fallbackUsed: defaultSource !== 'ml_model'
      };
    }

    return {
      value: Number(
        item?.predicted_load_mw ??
        item?.predicted_load ??
        item?.predicted_load_original ??
        item?.value ??
        0
      ),
      predictionSource: item?.prediction_source ?? defaultSource,
      fallbackUsed: Boolean(item?.fallback_used ?? (item?.prediction_source && item?.prediction_source !== 'ml_model'))
    };
  },

  extractPredictionValue: (item) => {
    return AnalyticsService.extractPredictionItem(item).value;
  },

  getPredictionItems: (response) => {
    const candidates = [
      response?.predictions,
      response?.forecast,
      response?.forecasts,
      response?.predicted_loads,
      response?.predicted_load_mw,
      response?.predicted_load
    ];

    const predictionList = candidates.find((candidate) => Array.isArray(candidate));
    if (predictionList) {
      return predictionList;
    }

    const singleValue = candidates.find((candidate) => Number.isFinite(Number(candidate)));
    return singleValue === undefined ? [] : [singleValue];
  },

  normalizePredictionResponse: (response) => {
    const defaultSource = response?.prediction_source || 'ml_model';
    const predictions = AnalyticsService.getPredictionItems(response)
      .map((item) => AnalyticsService.extractPredictionItem(item, defaultSource))
      .filter((item) => Number.isFinite(item.value));

    return {
      ...response,
      predictions,
      prediction_source: defaultSource,
      fallback_used: Boolean(response?.fallback_used || predictions.some((item) => item.fallbackUsed))
    };
  },

  buildLocalPredictionValue: (loads) => {
    const history = Array.isArray(loads) ? loads.map(Number).filter(Number.isFinite) : [];

    if (history.length !== 24) {
      throw new Error('Exactly 24 load values are required for fallback analytics.');
    }

    const recent = history.slice(-3);
    const tail = history.slice(-6);
    const recentMean = recent.reduce((sum, value) => sum + value, 0) / Math.max(recent.length, 1);
    const momentum = tail.slice(1).reduce((sum, value, index) => sum + (value - tail[index]), 0) / Math.max(tail.length - 1, 1);
    const variance = tail.reduce((sum, value) => sum + Math.pow(value - recentMean, 2), 0) / Math.max(tail.length, 1);
    const volatility = Math.sqrt(variance);

    const rawPrediction = (
      history[0] * 0.45 +
      recentMean * 0.45 +
      history[history.length - 1] * 0.10 +
      momentum * 0.75
    );

    const lowerBound = Math.max(0, recentMean - Math.max(volatility * 2.5, 1500));
    const upperBound = recentMean + Math.max(volatility * 2.5, 1500);
    return Number(Math.min(Math.max(rawPrediction, lowerBound), upperBound).toFixed(2));
  },

  buildLocalBatchFallback: ({ last24Hours, horizon, loads, scenarios, reason }) => {
    if (Array.isArray(last24Hours) && Number.isFinite(horizon)) {
      const history = [...last24Hours];
      const predictions = Array.from({ length: horizon }, () => {
        const predicted = AnalyticsService.buildLocalPredictionValue(history.slice(-24));
        history.push(predicted);
        return {
          predicted_load: predicted,
          predicted_load_original: predicted,
          prediction_source: 'client_heuristic',
          fallback_used: true
        };
      });

      return {
        predictions,
        horizon,
        mode: 'iterative_forecast_fallback',
        prediction_source: 'client_heuristic',
        fallback_used: true,
        reason
      };
    }

    const sequences = loads || scenarios || [];
    return {
      predictions: sequences.map((sequence) => {
        const last24 = Array.isArray(sequence) ? sequence : sequence?.last_24_hours;
        const predicted = AnalyticsService.buildLocalPredictionValue(last24);
        return {
          predicted_load: predicted,
          predicted_load_original: predicted,
          prediction_source: 'client_heuristic',
          fallback_used: true
        };
      }),
      mode: 'batch_fallback',
      prediction_source: 'client_heuristic',
      fallback_used: true,
      reason
    };
  },

  /**
   * Fetch batch predictions or iterative horizon forecasts.
   * Uses MODEL2_API for batch forecasting
   */
  getBatchPredictions: async ({ last24Hours, horizon, loads, scenarios }) => {
    try {
      const payload = last24Hours
        ? {
            model: 'ensemble',
            last_24_hours: last24Hours,
            horizon
          }
        : {
            model: 'ensemble',
            loads: loads || scenarios || []
          };

      const response = await fetchWithRetry(
        buildModel2Url(API_CONFIG.ENDPOINTS.PREDICT_BATCH),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        },
        3,
        API_CONFIG.TIMEOUTS.BATCH_PREDICTION
      );

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }

      const responsePayload = await response.json();
      return AnalyticsService.normalizePredictionResponse(responsePayload);
    } catch (error) {
      console.warn('Batch prediction fallback used:', error);
      const shouldFallback =
        !error.message.startsWith('Client error:') ||
        error.message.startsWith('Client error: 404') ||
        error.message.startsWith('Client error: 405');

      if (shouldFallback) {
        return AnalyticsService.normalizePredictionResponse(
          AnalyticsService.buildLocalBatchFallback({
            last24Hours,
            horizon,
            loads,
            scenarios,
            reason: error.message
          })
        );
      }

      throw error;
    }
  },

  getSinglePrediction: async (last24Hours) => {
    try {
      const now = new Date();
      const response = await fetchWithRetry(
        buildModel1Url(API_CONFIG.ENDPOINTS.PREDICT),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            hour: now.getHours(),
            day_of_week: (now.getDay() + 6) % 7,
            month: now.getMonth() + 1,
            model: 'ensemble',
            last_24_hours: last24Hours
          })
        },
        3,
        API_CONFIG.TIMEOUTS.PREDICTION
      );

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }

      const payload = await response.json();
      return AnalyticsService.extractPredictionItem(payload).value;
    } catch (error) {
      console.warn('Single prediction fallback used:', error);
      if (
        !error.message.startsWith('Client error:') ||
        error.message.startsWith('Client error: 404') ||
        error.message.startsWith('Client error: 405')
      ) {
        return AnalyticsService.buildLocalPredictionValue(last24Hours);
      }
      throw error;
    }
  },

  buildLoadHistory: ({
    baseLoad,
    renewablePercent = 35,
    solarGen = 4500,
    windOn = 12000,
    windOff = 2000,
    length = 48,
    phaseShift = 0,
    trendStrength = 0
  }) => {
    const totalWind = windOn + windOff;
    const safeBase = Math.max(18000, Number(baseLoad) || 0);

    return Array.from({ length }, (_, index) => {
      const hour = (index + phaseShift) % 24;
      const dailyWave = Math.sin((hour / 24) * Math.PI * 2 - Math.PI / 2);
      const eveningPeak = Math.exp(-Math.pow((hour - 19) / 4.5, 2)) * 0.14;
      const morningLift = Math.exp(-Math.pow((hour - 8) / 5.5, 2)) * 0.08;
      const solarOffset = Math.max(0, Math.sin(((hour - 6) / 24) * Math.PI * 2)) * (solarGen / Math.max(safeBase * 2.2, 1)) * 0.1;
      const windOffset = (Math.cos((hour / 24) * Math.PI * 2) * 0.5 + 0.5) * (totalWind / Math.max(safeBase * 2.6, 1)) * 0.08;
      const renewableOffset = (renewablePercent / 100) * 0.05;
      const trend = (index - length / 2) * trendStrength;

      const load = safeBase * (
        0.82 +
        dailyWave * 0.11 +
        eveningPeak +
        morningLift -
        solarOffset +
        windOffset +
        renewableOffset +
        trend
      );

      return Math.max(12000, Math.round(load));
    });
  },

  buildPriceSeries: ({ chartData, basePrice }) => {
    return chartData.map((point, index) => {
      const loadRatio = point.forecast / Math.max(point.load, 1);
      const timePremium = 1 + Math.sin((index / 24) * Math.PI * 2 - Math.PI / 2) * 0.06;
      return Number((basePrice * loadRatio * timePremium).toFixed(2));
    });
  },

  getOverviewTelemetry: async ({ loadVal, priceVal, solarGen, windOn, windOff }) => {
    const history = AnalyticsService.buildLoadHistory({
      baseLoad: loadVal,
      renewablePercent: Math.min(85, Math.max(10, ((solarGen + windOn + windOff) / Math.max(loadVal, 1)) * 100)),
      solarGen,
      windOn,
      windOff,
      length: 48
    });

    const windows = Array.from({ length: 24 }, (_, index) => history.slice(index, index + 24));
    const response = await AnalyticsService.getBatchPredictions({ loads: windows });
    const forecastValues = Array.isArray(response?.predictions)
      ? response.predictions.map((item) => item.value).filter(Number.isFinite)
      : [];

    const chartData = Array.from({ length: 24 }, (_, index) => ({
      time: `${index}:00`,
      load: Number(history[index + 24] || 0),
      forecast: Number(forecastValues[index] ?? history[index + 24] ?? 0)
    }));

    const priceSeries = AnalyticsService.buildPriceSeries({ chartData, basePrice: priceVal });

    return {
      chartData: chartData.map((point, index) => ({
        ...point,
        price: priceSeries[index]
      })),
      predictedLoad: Math.round(chartData[0]?.forecast || loadVal),
      latestForecast: Math.round(chartData[chartData.length - 1]?.forecast || loadVal),
      predictionSource: response?.prediction_source || 'ml_model',
      fallbackUsed: Boolean(response?.fallback_used),
      fallbackReason: response?.reason || null
    };
  },

  getDashboardAnalytics: async ({ baseLoad, renewablePercent }) => {
    const history = AnalyticsService.buildLoadHistory({
      baseLoad,
      renewablePercent,
      solarGen: baseLoad * (renewablePercent / 100) * 0.18,
      windOn: baseLoad * (renewablePercent / 100) * 0.3,
      windOff: baseLoad * (renewablePercent / 100) * 0.08,
      length: 48
    });

    const hourlyWindows = Array.from({ length: 24 }, (_, index) => history.slice(index, index + 24));
    const hourlyResponse = await AnalyticsService.getBatchPredictions({ loads: hourlyWindows });
    const hourlyPredictions = Array.isArray(hourlyResponse?.predictions)
      ? hourlyResponse.predictions.map((item) => item.value).filter(Number.isFinite)
      : [];

    const hourlyData = Array.from({ length: 24 }, (_, hour) => {
      const actual = Number(history[hour + 24] || 0);
      const forecast = Number(hourlyPredictions[hour] ?? actual);
      return {
        hour: `${hour}:00`,
        hourNum: hour,
        avgLoad: Math.round((actual + forecast) / 2),
        min: Math.round(Math.min(actual, forecast)),
        max: Math.round(Math.max(actual, forecast)),
        variance: Math.round(Math.abs(forecast - actual)),
        actual,
        forecast
      };
    });

    const predictedAverage = hourlyData.reduce((sum, point) => sum + point.forecast, 0) / Math.max(hourlyData.length, 1);
    const regionalWeights = [
      { region: '50Hertz', weight: 0.25 },
      { region: 'Tennet', weight: 0.35 },
      { region: 'Amprion', weight: 0.30 },
      { region: 'TransnetBW', weight: 0.10 }
    ];

    const regionalDistribution = regionalWeights.map((item) => ({
      region: item.region,
      value: Math.round(predictedAverage * item.weight),
      percentage: Math.round(item.weight * 100)
    }));

    const loadComposition = AnalyticsService.generateLoadComposition(
      Math.round(predictedAverage),
      renewablePercent
    );

    const scatterData = hourlyData.map((point) => ({
      actual: point.actual,
      forecast: point.forecast,
      difference: point.forecast - point.actual,
      accuracy: point.actual
        ? 100 - Math.abs((point.forecast - point.actual) / point.actual * 100)
        : 100
    }));

    const baselineWindow = history.slice(history.length - 24);
    const sensitivityScenarios = [
      { factor: 'Temperature', color: '#3b82f6', scale: 0.94 },
      { factor: 'Solar Irradiance', color: '#fbbf24', scale: 0.9 },
      { factor: 'Wind Speed', color: '#60a5fa', scale: 0.96 },
      { factor: 'Time of Day', color: '#818cf8', scale: 1.05 },
      { factor: 'Day of Week', color: '#a855f7', scale: 1.02 }
    ];

    const sensitivityLoads = [
      baselineWindow,
      ...sensitivityScenarios.map((scenario, index) => baselineWindow.map((value, pointIndex) => {
        const wave = 1 + Math.sin(((pointIndex + index) / 24) * Math.PI * 2) * 0.03;
        return Math.max(12000, Math.round(value * scenario.scale * wave));
      }))
    ];

    const sensitivityResponse = await AnalyticsService.getBatchPredictions({ loads: sensitivityLoads });
    const sensitivityPredictions = Array.isArray(sensitivityResponse?.predictions)
      ? sensitivityResponse.predictions.map((item) => item.value).filter(Number.isFinite)
      : [];

    const baselinePrediction = sensitivityPredictions[0] || predictedAverage;
    const sensitivityData = sensitivityScenarios.map((scenario, index) => {
      const predicted = sensitivityPredictions[index + 1] || baselinePrediction;
      return {
        factor: scenario.factor,
        impact: Number((((predicted - baselinePrediction) / Math.max(baselinePrediction, 1)) * 100).toFixed(1)),
        color: scenario.color
      };
    });

    return {
      hourlyData,
      regionalDistribution,
      loadComposition,
      scatterData,
      sensitivityData,
      predictionSource: [hourlyResponse?.prediction_source, sensitivityResponse?.prediction_source]
        .filter(Boolean)
        .find((source) => source !== 'ml_model') || hourlyResponse?.prediction_source || 'ml_model',
      fallbackUsed: Boolean(hourlyResponse?.fallback_used || sensitivityResponse?.fallback_used),
      fallbackReason: hourlyResponse?.reason || sensitivityResponse?.reason || null
    };
  },

  /**
   * Generate hour-wise analysis data
   */
  generateHourlyAnalysis: (loads) => {
    return Array.from({ length: 24 }, (_, hour) => {
      const hourLoads = loads.slice(hour * 4, (hour + 1) * 4);
      const avgLoad = hourLoads.length > 0 
        ? hourLoads.reduce((a, b) => a + b, 0) / hourLoads.length 
        : 0;
      
      return {
        hour: `${hour}:00`,
        hourNum: hour,
        avgLoad: Math.round(avgLoad),
        min: Math.min(...hourLoads),
        max: Math.max(...hourLoads),
        variance: Math.max(...hourLoads) - Math.min(...hourLoads)
      };
    });
  },

  /**
   * Generate regional distribution data
   */
  generateRegionalDistribution: (totalLoad) => {
    return [
      { region: '50Hertz', value: Math.round(totalLoad * 0.25), percentage: 25 },
      { region: 'Tennet', value: Math.round(totalLoad * 0.35), percentage: 35 },
      { region: 'Amprion', value: Math.round(totalLoad * 0.30), percentage: 30 },
      { region: 'TransnetBW', value: Math.round(totalLoad * 0.10), percentage: 10 }
    ];
  },

  /**
   * Generate correlation data between variables
   */
  generateCorrelationScatter: (historicalLoads, forecasts) => {
    return historicalLoads.slice(0, Math.min(50, historicalLoads.length)).map((load, i) => ({
      actual: load,
      forecast: forecasts[i] || 0,
      difference: (forecasts[i] || 0) - load,
      accuracy: 100 - Math.abs(((forecasts[i] || 0) - load) / load * 100)
    }));
  },

  /**
   * Generate load composition data
   */
  generateLoadComposition: (totalLoad, renewablePercent = 35) => {
    const renewable = Math.round(totalLoad * (renewablePercent / 100));
    const conventional = totalLoad - renewable;
    
    return [
      {
        name: 'Renewable Energy',
        value: renewable,
        percentage: renewablePercent,
        color: '#10b981'
      },
      {
        name: 'Conventional',
        value: conventional,
        percentage: 100 - renewablePercent,
        color: '#ef4444'
      }
    ];
  },

  /**
   * Generate sensitivity analysis data
   */
  generateSensitivityAnalysis: (baseLoad) => {
    return [
      { factor: 'Temperature', impact: -8.5, color: '#3b82f6' },
      { factor: 'Solar Irradiance', impact: 12.3, color: '#fbbf24' },
      { factor: 'Wind Speed', impact: 5.7, color: '#60a5fa' },
      { factor: 'Time of Day', impact: -15.2, color: '#818cf8' },
      { factor: 'Day of Week', impact: 3.8, color: '#a855f7' }
    ];
  }
};
