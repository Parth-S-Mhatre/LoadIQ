import { useState, useEffect, useCallback, useRef } from 'react';
import { API_CONFIG, buildUrl, fetchWithTimeout } from '../config/api';

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
    consecutiveFailures: 0
};

const extractBackendModelState = (data) => {
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

    return {
        modelLoaded: false,
        loadedModels: 0,
        totalModels: 0
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
        const healthEndpoints = [
            API_CONFIG.ENDPOINTS.HEALTH,
            API_CONFIG.ENDPOINTS.HEALTH_CHECK
        ];
        let lastError = null;

        try {
            for (const endpoint of healthEndpoints) {
                try {
                    const response = await fetchWithTimeout(
                        buildUrl(endpoint),
                        { method: 'GET' },
                        API_CONFIG.TIMEOUTS.HEALTH_CHECK
                    );

                    if (!mountedRef.current) return;

                    if (!response.ok) {
                        throw new Error(`Health check failed at ${endpoint}: ${response.status}`);
                    }

                    const data = await response.json();
                    const latency = Date.now() - startTime;
                    const { modelLoaded, loadedModels, totalModels } = extractBackendModelState(data);

                    setStatus({
                        isOnline: true,
                        isChecking: false,
                        lastCheck: new Date(),
                        latency,
                        error: null,
                        modelLoaded,
                        loadedModels,
                        totalModels,
                        backendMode: data.mode || null,
                        consecutiveFailures: 0
                    });
                    return;
                } catch (error) {
                    lastError = error;
                }
            }

            throw lastError || new Error('Health check failed');
        } catch (error) {
            if (!mountedRef.current) return;

            setStatus(prev => ({
                isOnline: false,
                isChecking: false,
                lastCheck: new Date(),
                latency: null,
                error: error.message,
                modelLoaded: false,
                loadedModels: 0,
                totalModels: 0,
                backendMode: null,
                consecutiveFailures: prev.consecutiveFailures + 1
            }));
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
