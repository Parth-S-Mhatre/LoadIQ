import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Stars, Preload } from '@react-three/drei';
import * as THREE from 'three';

const Bar = ({ position, height, color, label, isPredicted }) => {
    const mesh = useRef();
    const [hovered, setHover] = useState(false);

    useFrame(() => {
        if (mesh.current) {
            mesh.current.scale.y = THREE.MathUtils.lerp(mesh.current.scale.y, Math.max(0.01, height), 0.1);
            mesh.current.position.y = mesh.current.scale.y / 2;
        }
    });

    return (
        <group position={position}>
            <mesh
                ref={mesh}
                position={[0, 0.5, 0]}
                scale={[1, 0.01, 1]} // Start small
                onPointerOver={() => setHover(true)}
                onPointerOut={() => setHover(false)}
            >
                <boxGeometry args={[0.8, 1, 0.8]} />
                <meshStandardMaterial
                    color={hovered ? '#ffffff' : color}
                    emissive={color}
                    emissiveIntensity={hovered ? 0.8 : 0.2}
                    metalness={0.6}
                    roughness={0.2}
                    transparent
                    opacity={isPredicted ? 0.8 : 1}
                />
            </mesh>
            {/* Label only on hover or specific indices to avoid clutter */}
            {(hovered || position[0] % 6 === 0) && (
                <Text
                    position={[0, -0.5, 0.5]}
                    fontSize={0.5}
                    color="white"
                    anchorX="center"
                    anchorY="top"
                >
                    {label}
                </Text>
            )}
        </group>
    );
};

const ThreeLoadChartComponent = ({ safeData, safePredicted, maxVal, getNormalizedHeight }) => {
    // Calculate total bars and optimal camera position
    const totalBars = safeData.length + safePredicted.length;
    const cameraDistance = Math.max(20, totalBars * 1.2); // Dynamic distance based on data
    
    return (
        <>
            {/* Lighting - Enhanced for visibility without Environment */}
            <ambientLight intensity={0.6} />
            <directionalLight position={[15, 25, 15]} intensity={1.4} color="#ffffff" castShadow />
            <pointLight position={[-10, 10, 15]} intensity={0.9} color="#6366f1" />
            <spotLight position={[5, 30, 10]} angle={0.5} penumbra={1} intensity={1.2} color="#818cf8" />

            {/* Bars - Center the graph */}
            <group position={[-totalBars / 2 + 0.5, 0, 0]}>

                {/* Safe Data Rendering */}
                {safeData.map((val, i) => (
                    <Bar
                        key={`hist-${i}`}
                        position={[i, 0, 0]}
                        height={getNormalizedHeight(val)}
                        color="#6366f1"
                        label={`H${i}`}
                    />
                ))}

                {/* Safe Predicted Rendering */}
                {safePredicted.map((val, i) => (
                    <Bar
                        key={`pred-${i}`}
                        position={[safeData.length + i, 0, 0]}
                        height={getNormalizedHeight(val)}
                        color="#10b981"
                        label={`P${i + 1}`}
                        isPredicted={true}
                    />
                ))}

                {/* Floor Plane reflection */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-0.5, -0.5, 0]}>
                    <planeGeometry args={[totalBars + 5, 20]} />
                    <meshStandardMaterial
                        color="#1e293b"
                        roughness={0.2}
                        metalness={0.5}
                        transparent
                        opacity={0.3}
                    />
                </mesh>
            </group>

            <OrbitControls
                enableZoom={true}
                enablePan={true}
                minPolarAngle={Math.PI / 8}
                maxPolarAngle={Math.PI / 2.5}
                autoRotate={safePredicted.length > 0}
                autoRotateSpeed={0.4}
                makeDefault
            />

            {/* Stars Background */}
            <Stars radius={150} depth={80} count={2000} factor={4} saturation={0} fade speed={0.3} />
            <Preload all />
        </>
    );
};

const ThreeLoadChart = ({ data = [], predicted = [] }) => {
    // Safety check for data
    const safeData = Array.isArray(data) ? data : [];
    const safePredicted = Array.isArray(predicted) ? predicted : [];
    const canvasRef = useRef();

    // Calculate max value for normalization
    const maxVal = useMemo(() => {
        const allValues = [...safeData, ...safePredicted];
        const max = allValues.length > 0 ? Math.max(...allValues) : 1000;
        return max === 0 ? 1000 : max; // Prevent division by zero
    }, [safeData, safePredicted]);

    const scaleFactor = 8; // Taller bars for better visibility

    const getNormalizedHeight = (val) => {
        if (typeof val !== 'number' || isNaN(val)) return 0.1;
        return (val / maxVal) * scaleFactor;
    };

    // Calculate optimal camera position based on data
    const totalBars = safeData.length + safePredicted.length;
    const cameraZ = Math.max(25, totalBars * 1.5);
    const cameraY = Math.max(15, totalBars * 0.6);

    // Handle WebGL context loss with proper cleanup
    useEffect(() => {
        const canvas = canvasRef.current?.querySelector('canvas');
        if (!canvas) return;

        const handleContextLost = (e) => {
            e.preventDefault();
            console.warn('ThreeLoadChart: WebGL context lost. Context lost event triggered.');
        };

        const handleContextRestored = () => {
            console.log('ThreeLoadChart: WebGL context restored.');
        };

        canvas.addEventListener('webglcontextlost', handleContextLost, false);
        canvas.addEventListener('webglcontextrestored', handleContextRestored, false);

        return () => {
            canvas.removeEventListener('webglcontextlost', handleContextLost);
            canvas.removeEventListener('webglcontextrestored', handleContextRestored);
        };
    }, []);

    return (
        <div className="h-[400px] w-full bg-[#0f172a] rounded-2xl overflow-hidden border border-slate-700 relative" ref={canvasRef}>
            {/* Legend Overlay */}
            <div className="absolute top-4 left-4 z-10 flex gap-4 pointer-events-none">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-indigo-500 rounded-full shadow-[0_0_10px_rgba(99,102,241,0.5)]"></div>
                    <span className="text-xs font-bold text-slate-300">Historical</span>
                </div>
                {safePredicted.length > 0 && (
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                        <span className="text-xs font-bold text-slate-300">Predicted</span>
                    </div>
                )}
            </div>

            <Canvas
                camera={{ position: [0, cameraY, cameraZ], fov: 45, near: 0.1, far: 1000 }}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: 'high-performance',
                    preserveDrawingBuffer: true,
                    failIfMajorPerformanceCaveat: false
                }}
                onCreated={({ gl }) => {
                    gl.setClearColor('#0f172a', 1);
                }}
                onError={(error) => console.warn('Canvas error:', error)}
            >
                <ThreeLoadChartComponent
                    safeData={safeData}
                    safePredicted={safePredicted}
                    maxVal={maxVal}
                    getNormalizedHeight={getNormalizedHeight}
                />
            </Canvas>
        </div>
    );
};

export default ThreeLoadChart;
