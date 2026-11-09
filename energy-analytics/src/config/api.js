// API Configuration for LoadIQ Backend
const trimTrailingSlash = (url) => url.replace(/\/+$/, '');
const DEFAULT_PRIMARY_BACKEND_URL = 'https://loadiq-backend.onrender.com';
const DEFAULT_BACKUP_MODEL1_URL = 'https://loadiq-model1-production.up.railway.app';
const DEFAULT_BACKUP_MODEL2_URL = 'https://loadiq-model2-production.up.railway.app';
const MAINTENANCE_START_ISO = '2026-05-27T01:00:00Z';
const MAINTENANCE_END_ISO = '2026-05-27T02:00:00Z';

const isLocalHost = () => {
    if (typeof window === 'undefined') {
        return false;
    }

    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
};

const shouldUseLocalModelApis = () => {
    return process.env.REACT_APP_USE_LOCAL_MODEL_APIS === 'true';
};

const resolveBaseUrl = () => {
    if (process.env.REACT_APP_API_URL) {
        return trimTrailingSlash(process.env.REACT_APP_API_URL);
    }

    if (isLocalHost() && shouldUseLocalModelApis()) {
        return 'http://127.0.0.1:8000';
    }

    return DEFAULT_PRIMARY_BACKEND_URL;
};

const resolveModel1Url = () => {
    if (process.env.REACT_APP_MODEL1_API) {
        return trimTrailingSlash(process.env.REACT_APP_MODEL1_API);
    }

    return `${resolveBaseUrl()}/model1`;
};

const resolveModel2Url = () => {
    if (process.env.REACT_APP_MODEL2_API) {
        return trimTrailingSlash(process.env.REACT_APP_MODEL2_API);
    }

    return `${resolveBaseUrl()}/model2`;
};

const resolveBackupModel1Url = () => {
    if (process.env.REACT_APP_BACKUP_MODEL1_API) {
        return trimTrailingSlash(process.env.REACT_APP_BACKUP_MODEL1_API);
    }

    return DEFAULT_BACKUP_MODEL1_URL;
};

const resolveBackupModel2Url = () => {
    if (process.env.REACT_APP_BACKUP_MODEL2_API) {
        return trimTrailingSlash(process.env.REACT_APP_BACKUP_MODEL2_API);
    }

    return DEFAULT_BACKUP_MODEL2_URL;
};

export const MAINTENANCE_WINDOW = {
    START: new Date(MAINTENANCE_START_ISO),
    END: new Date(MAINTENANCE_END_ISO)
};

export const formatCountdown = (totalMs) => {
    const safeMs = Math.max(0, Math.floor(totalMs));
    const totalSeconds = Math.floor(safeMs / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
        return `${hours}h ${String(minutes).padStart(2, '0')}m ${String(seconds).padStart(2, '0')}s`;
    }

    return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
};

export const API_CONFIG = {
    BASE_URL: resolveBaseUrl(),
    MODEL1_API: resolveModel1Url(),
    MODEL2_API: resolveModel2Url(),
    BACKUP_MODEL1_API: resolveBackupModel1Url(),
    BACKUP_MODEL2_API: resolveBackupModel2Url(),
    ENDPOINTS: {
        PREDICT: '/predict',
        HEALTH: '/health',
        HEALTH_CHECK: '/api/health_check',
        DEBUG_PREDICT: '/debug_predict',
        PREDICT_BATCH: '/predict_batch',
        FORECAST: '/forecast'
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
export const buildUrl = (endpoint, baseUrl = API_CONFIG.BASE_URL) => {
    return `${baseUrl}${endpoint}`;
};

// Helper function to build Model1 URL for single predictions
export const buildModel1Url = (endpoint, baseUrl = API_CONFIG.MODEL1_API) => {
    return `${baseUrl}${endpoint}`;
};

// Helper function to build Model2 URL for batch predictions and forecasts
export const buildModel2Url = (endpoint, baseUrl = API_CONFIG.MODEL2_API) => {
    return `${baseUrl}${endpoint}`;
};

export const buildBackupModel1Url = (endpoint, baseUrl = API_CONFIG.BACKUP_MODEL1_API) => {
    return `${baseUrl}${endpoint}`;
};

export const buildBackupModel2Url = (endpoint, baseUrl = API_CONFIG.BACKUP_MODEL2_API) => {
    return `${baseUrl}${endpoint}`;
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

export const fetchWithBackendFallback = async (
    primaryUrl,
    fallbackUrl,
    options = {},
    timeout = API_CONFIG.TIMEOUTS.DEFAULT
) => {
    let primaryResponse;

    try {
        primaryResponse = await fetchWithTimeout(primaryUrl, options, timeout);
    } catch (primaryError) {
        if (!fallbackUrl) {
            throw primaryError;
        }

        const fallbackResponse = await fetchWithTimeout(fallbackUrl, options, timeout);
        if (!fallbackResponse.ok) {
            throw new Error(`Fallback backend responded with ${fallbackResponse.status}`);
        }

        return {
            response: fallbackResponse,
            source: 'fallback'
        };
    }

    if (primaryResponse.ok) {
        return {
            response: primaryResponse,
            source: 'primary'
        };
    }

    const shouldTryFallback = primaryResponse.status >= 500 || primaryResponse.status === 404 || primaryResponse.status === 405;
    if (!shouldTryFallback || !fallbackUrl) {
        throw new Error(`Backend responded with ${primaryResponse.status}`);
    }

    const fallbackResponse = await fetchWithTimeout(fallbackUrl, options, timeout);
    if (!fallbackResponse.ok) {
        throw new Error(`Fallback backend responded with ${fallbackResponse.status}`);
    }

    return {
        response: fallbackResponse,
        source: 'fallback'
    };
};
