// API Configuration for LoadIQ Backend
const resolveBaseUrl = () => {
    if (process.env.REACT_APP_API_URL) {
        return process.env.REACT_APP_API_URL;
    }

    if (typeof window !== 'undefined') {
        const host = window.location.hostname;
        if (host === 'localhost' || host === '127.0.0.1') {
            return 'http://127.0.0.1:8002';
        }
    }

    return 'https://loadiq.onrender.com';
};

export const API_CONFIG = {
    BASE_URL: resolveBaseUrl(),
    ENDPOINTS: {
        PREDICT: '/predict',
        HEALTH: '/health',
        HEALTH_CHECK: '/api/health_check',
        DEBUG_PREDICT: '/debug_predict',
        PREDICT_BATCH: '/predict_batch'
    },
    TIMEOUTS: {
        DEFAULT: 10000, // 10 seconds
        HEALTH_CHECK: 5000, // 5 seconds
        PREDICTION: 20000, // 20 seconds for single-step inference
        BATCH_PREDICTION: 45000 // 45 seconds for dashboard batch analytics
    },
    RETRY: {
        MAX_ATTEMPTS: 3,
        INITIAL_DELAY: 1000, // 1 second
        MAX_DELAY: 5000, // 5 seconds
        BACKOFF_MULTIPLIER: 2
    },
    HEALTH_CHECK_INTERVAL: 30000 // 30 seconds
};

// Helper function to build full URL
export const buildUrl = (endpoint) => {
    return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Helper function for exponential backoff
export const calculateBackoff = (attempt) => {
    const delay = Math.min(
        API_CONFIG.RETRY.INITIAL_DELAY * Math.pow(API_CONFIG.RETRY.BACKOFF_MULTIPLIER, attempt),
        API_CONFIG.RETRY.MAX_DELAY
    );
    return delay;
};

// Fetch with timeout
export const fetchWithTimeout = async (url, options = {}, timeout = API_CONFIG.TIMEOUTS.DEFAULT) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout');
        }
        throw error;
    }
};

// Fetch with retry logic
export const fetchWithRetry = async (
    url,
    options = {},
    maxAttempts = API_CONFIG.RETRY.MAX_ATTEMPTS,
    timeout = API_CONFIG.TIMEOUTS.DEFAULT
) => {
    let lastError;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const response = await fetchWithTimeout(url, options, timeout);
            if (response.ok) {
                return response;
            }

            // Don't retry on 4xx errors (client errors)
            if (response.status >= 400 && response.status < 500) {
                let detail = '';

                try {
                    const errorPayload = await response.clone().json();
                    detail = errorPayload?.detail || errorPayload?.message || '';
                } catch {
                    try {
                        detail = await response.clone().text();
                    } catch {
                        detail = '';
                    }
                }

                throw new Error(detail
                    ? `Client error: ${response.status} - ${detail}`
                    : `Client error: ${response.status}`);
            }

            lastError = new Error(`Server error: ${response.status}`);
        } catch (error) {
            lastError = error;

            // Don't retry on client errors
            if (error.message.includes('Client error')) {
                throw error;
            }

            // Wait before retrying (except on last attempt)
            if (attempt < maxAttempts - 1) {
                await new Promise(resolve => setTimeout(resolve, calculateBackoff(attempt)));
            }
        }
    }

    throw lastError;
};
