import { useEffect, useState } from 'react';

const SLOW_CONNECTION_TYPES = new Set(['slow-2g', '2g', '3g']);

const readConnectionState = () => {
    if (typeof navigator === 'undefined') {
        return {
            downlink: null,
            effectiveType: null,
            isLowBandwidth: false,
            saveData: false
        };
    }

    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;

    if (!connection) {
        return {
            downlink: null,
            effectiveType: null,
            isLowBandwidth: false,
            saveData: false
        };
    }

    const effectiveType = connection.effectiveType || null;
    const downlink = typeof connection.downlink === 'number' ? connection.downlink : null;
    const saveData = Boolean(connection.saveData);
    const isLowBandwidth = Boolean(
        saveData ||
        !navigator.onLine ||
        SLOW_CONNECTION_TYPES.has(effectiveType) ||
        (downlink !== null && downlink <= 1.5)
    );

    return {
        downlink,
        effectiveType,
        isLowBandwidth,
        saveData
    };
};

const useNetworkQuality = () => {
    const [networkState, setNetworkState] = useState(() => readConnectionState());

    useEffect(() => {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        const updateNetworkState = () => setNetworkState(readConnectionState());

        updateNetworkState();

        window.addEventListener('online', updateNetworkState);
        window.addEventListener('offline', updateNetworkState);

        if (connection?.addEventListener) {
            connection.addEventListener('change', updateNetworkState);
        }

        return () => {
            window.removeEventListener('online', updateNetworkState);
            window.removeEventListener('offline', updateNetworkState);

            if (connection?.removeEventListener) {
                connection.removeEventListener('change', updateNetworkState);
            }
        };
    }, []);

    return networkState;
};

export default useNetworkQuality;
