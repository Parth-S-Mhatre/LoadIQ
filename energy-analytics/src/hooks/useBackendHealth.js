import { useState, useEffect, useCallback, useRef } from 'react';
import {
    API_CONFIG,
    buildUrl,
    fetchWithTimeout
} from '../config/api';

/**
 * Custom hook for monitoring backend health status
 * Performs periodic health checks and tracks connection status
 */
const defaultStatus = {
    isOnline: false,
    isChecking: true,
    lastCheck: null,
    latency: null,
    error: null,
    modelLoaded: false,
    loadedModels: 0,
    totalModels: 0,
    backendMode: null,
    backendSource: null,
    backendLabel: null,
    activeBackendUrl: null,
    consecutiveFailures: 0
};

const extractRenderModelState = (data) => {
    if (!data || typeof data !== 'object') {
        return {
            modelLoaded: false,
            loadedModels: 0,
            totalModels: 0
        };
    }

    if (typeof data.model_loaded === 'boolean') {
        return {
            modelLoaded: data.model_loaded,
            loadedModels: data.model_loaded ? 1 : 0,
            totalModels: 1
        };
    }

    if (data.models && typeof data.models === 'object') {
        const modelStates = Object.values(data.models).filter((value) => typeof value === 'boolean');
        const loadedModels = modelStates.filter(Boolean).length;
        const totalModels = modelStates.length;

        return {
            modelLoaded: totalModels === 0 ? false : loadedModels === totalModels,
            loadedModels,
            totalModels
        };
    }

    if (Array.isArray(data.models_loaded)) {
        const loadedModels = data.models_loaded.filter(Boolean).length;
        if (loadedModels > 0) {
            return {
                modelLoaded: true,
                loadedModels,
                totalModels: Math.max(loadedModels, 1)
            };
        }

        // Lazy backends often report an empty models_loaded until the first
        // prediction. models_available lets the UI say "Models ready" without
        // forcing a warm-up request.
        if (Array.isArray(data.models_available) && data.models_available.length > 0) {
            const availableModels = data.models_available.length;
            return {
                modelLoaded: true,
                loadedModels: availableModels,
                totalModels: availableModels
            };
        }

        return {
            modelLoaded: false,
            loadedModels: 0,
            totalModels: 1
        };
    }

    return {
        modelLoaded: false,
        loadedModels: 0,
        totalModels: 0
    };
};

const extractRailwayModelState = (data) => {
    if (!data || typeof data !== 'object') {
        return {
            modelLoaded: false,
            loadedModels: 0,
            totalModels: 0
        };
    }

    if (typeof data.model_loaded === 'boolean') {
        return {
            modelLoaded: data.model_loaded,
            loadedModels: data.model_loaded ? 1 : 0,
            totalModels: 1
        };
    }

    return {
        modelLoaded: data.status === 'ok' || data.status === 'running',
        loadedModels: 1,
        totalModels: 1
    };
};

const probeHealthEndpoint = async (baseUrl, endpoint, timeout) => {
    const response = await fetchWithTimeout(
        buildUrl(endpoint, baseUrl),
        { method: 'GET' },
        timeout
    );

    if (!response.ok) {
        throw new Error(`Health check failed at ${baseUrl}${endpoint}: ${response.status}`);
    }

    return response.json();
};

const probeRenderBackend = async (startTime) => {
    // The combined /health route has historically been deployed without CORS
    // headers, while the mounted model health routes always include them.
    // Probe the model route first, then fall back to the combined health routes.
    const healthTargets = [
        { baseUrl: API_CONFIG.MODEL2_API, endpoint: API_CONFIG.ENDPOINTS.HEALTH },
        { baseUrl: API_CONFIG.BASE_URL, endpoint: API_CONFIG.ENDPOINTS.HEALTH },
        { baseUrl: API_CONFIG.BASE_URL, endpoint: API_CONFIG.ENDPOINTS.HEALTH_CHECK },
        { baseUrl: API_CONFIG.MODEL1_API, endpoint: API_CONFIG.ENDPOINTS.HEALTH }
    ];
    let lastError = null;

    for (const { baseUrl, endpoint } of healthTargets) {
        try {
            const data = await probeHealthEndpoint(
                baseUrl,
                endpoint,
                API_CONFIG.TIMEOUTS.HEALTH_CHECK
            );
            const latency = Date.now() - startTime;
            const { modelLoaded, loadedModels, totalModels } = extractRenderModelState(data);

            return {
                isOnline: true,
                isChecking: false,
                lastCheck: new Date(),
                latency,
                error: null,
                modelLoaded,
                loadedModels,
                totalModels,
                backendMode: data.mode || 'combined_lazy_backend',
                backendSource: 'render',
                backendLabel: 'Render backend',
                activeBackendUrl: API_CONFIG.BASE_URL,
                consecutiveFailures: 0
            };
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error('Render backend health check failed');
};

const probeRailwayFallback = async (startTime) => {
    const probes = [
        {
            baseUrl: API_CONFIG.BACKUP_MODEL1_API,
            label: 'Railway model1'
        },
        {
            baseUrl: API_CONFIG.BACKUP_MODEL2_API,
            label: 'Railway model2'
        }
    ];

    const results = await Promise.allSettled(
        probes.map(async (probe) => {
            const data = await probeHealthEndpoint(
                probe.baseUrl,
                API_CONFIG.ENDPOINTS.HEALTH,
                API_CONFIG.TIMEOUTS.HEALTH_CHECK
            );

            return {
                probe,
                data
            };
        })
    );

    const successful = results
        .filter((result) => result.status === 'fulfilled')
        .map((result) => result.value);

    if (!successful.length) {
        const errors = results
            .filter((result) => result.status === 'rejected')
            .map((result) => result.reason?.message)
            .filter(Boolean);
        throw new Error(errors[0] || 'Railway fallback is unavailable');
    }

    const modelStates = successful.map(({ data }) => extractRailwayModelState(data));
    const loadedModels = modelStates.filter((state) => state.modelLoaded).length;
    const totalModels = probes.length;
    const latency = Date.now() - startTime;

    return {
        isOnline: true,
        isChecking: false,
        lastCheck: new Date(),
        latency,
        error: null,
        modelLoaded: loadedModels === totalModels,
        loadedModels,
        totalModels,
        backendMode: 'railway_fallback',
        backendSource: 'railway',
        backendLabel: 'Railway fallback',
        activeBackendUrl: 'Railway fallback',
        consecutiveFailures: 0
    };
};

const useBackendHealth = (enabled = true) => {
    const [status, setStatus] = useState({
        ...defaultStatus,
        isChecking: enabled
    });

    const intervalRef = useRef(null);
    const mountedRef = useRef(true);

    const checkHealth = useCallback(async () => {
        if (!enabled) {
            return;
        }

        const startTime = Date.now();

        try {
            const renderStatus = await probeRenderBackend(startTime);
            if (!mountedRef.current) return;
            setStatus(renderStatus);
        } catch (error) {
            if (!mountedRef.current) return;

            try {
                const fallbackStatus = await probeRailwayFallback(startTime);
                if (!mountedRef.current) return;
                setStatus(fallbackStatus);
            } catch (fallbackError) {
                if (!mountedRef.current) return;

                setStatus(prev => ({
                    ...defaultStatus,
                    isOnline: false,
                    isChecking: false,
                    lastCheck: new Date(),
                    latency: null,
                    error: fallbackError?.message || error.message,
                    consecutiveFailures: prev.consecutiveFailures + 1
                }));
            }
        }
    }, [enabled]);

    // Initial health check
    useEffect(() => {
        if (!enabled) {
            setStatus({
                ...defaultStatus,
                isChecking: false
            });
            return undefined;
        }

        mountedRef.current = true;
        checkHealth();

        return () => {
            mountedRef.current = false;
        };
    }, [checkHealth, enabled]);

    // Periodic health checks
    useEffect(() => {
        if (!enabled) {
            return undefined;
        }

        intervalRef.current = setInterval(() => {
            checkHealth();
        }, API_CONFIG.HEALTH_CHECK_INTERVAL);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [checkHealth, enabled]);

    // Manual refresh function
    const refresh = useCallback(() => {
        if (!enabled) {
            return;
        }

        setStatus(prev => ({ ...prev, isChecking: true }));
        checkHealth();
    }, [checkHealth, enabled]);

    return {
        ...status,
        refresh
    };
};

export default useBackendHealth;
