import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Preload } from '@react-three/drei';
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
                    color={hovered ? '#FFFFFF' : color}
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

const ThreeLoadChartComponent = ({ safeData, safePredicted, getNormalizedHeight }) => {
    // Calculate total bars and optimal camera position
    const totalBars = safeData.length + safePredicted.length;
    
    return (
        <>
            {/* Lighting - Enhanced for visibility without Environment */}
            <ambientLight intensity={0.75} />
            <directionalLight position={[15, 25, 15]} intensity={1.15} color="#F2F0E9" castShadow />
            <pointLight position={[-10, 10, 15]} intensity={0.7} color="#D8F24D" />
            <spotLight position={[5, 30, 10]} angle={0.5} penumbra={1} intensity={1} color="#3A6B8A" />

            {/* Bars - Center the graph */}
            <group position={[-totalBars / 2 + 0.5, 0, 0]}>

                {/* Safe Data Rendering */}
                {safeData.map((val, i) => (
                    <Bar
                        key={`hist-${i}`}
                        position={[i, 0, 0]}
                        height={getNormalizedHeight(val)}
                        color="#D2CDBE"
                        label={`H${i}`}
                    />
                ))}

                {/* Safe Predicted Rendering */}
                {safePredicted.map((val, i) => (
                    <Bar
                        key={`pred-${i}`}
                        position={[safeData.length + i, 0, 0]}
                        height={getNormalizedHeight(val)}
                        color="#D8F24D"
                        label={`P${i + 1}`}
                        isPredicted={true}
                    />
                ))}

                {/* Floor Plane reflection */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-0.5, -0.5, 0]}>
                    <planeGeometry args={[totalBars + 5, 20]} />
                    <meshStandardMaterial
                        color="#24302A"
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

            {/* Ground grid */}
            <gridHelper args={[60, 60, '#24302A', '#18231D']} position={[0, -0.5, 0]} />
            <Preload all />
        </>
    );
};

const ThreeLoadChart = ({ data = [], predicted = [] }) => {
    // Safety check for data
    const safeData = useMemo(() => (Array.isArray(data) ? data : []), [data]);
    const safePredicted = useMemo(() => (Array.isArray(predicted) ? predicted : []), [predicted]);
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
        <div className="relative h-full min-h-[300px] w-full overflow-hidden rounded-xl bg-[#0E1712]" ref={canvasRef}>
            {/* Legend Overlay */}
            <div className="pointer-events-none absolute left-4 top-4 z-10 flex gap-4">
                <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-[#D2CDBE]" />
                    <span className="label-sm text-paper/55">Historical</span>
                </div>
                {safePredicted.length > 0 && (
                    <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-signal" />
                        <span className="label-sm text-paper/55">Predicted</span>
                    </div>
                )}
            </div>

            <Canvas
                camera={{ position: [0, cameraY, cameraZ], fov: 45, near: 0.1, far: 1000 }}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: 'high-performance',
                    failIfMajorPerformanceCaveat: false
                }}
                onCreated={({ gl }) => {
                    gl.setClearColor('#0E1712', 1);
                }}
                onError={(error) => console.warn('Canvas error:', error)}
            >
                <ThreeLoadChartComponent
                    safeData={safeData}
                    safePredicted={safePredicted}
                    getNormalizedHeight={getNormalizedHeight}
                />
            </Canvas>
        </div>
    );
};

export default ThreeLoadChart;
