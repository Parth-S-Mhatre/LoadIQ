import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Sparkles, Float, Preload } from '@react-three/drei';

const AnimatedParticles = () => {
    const ref = useRef();
    useFrame((state) => {
        // Slow rotation
        if (ref.current) {
            ref.current.rotation.y = state.clock.elapsedTime * 0.05;
            ref.current.rotation.x = state.clock.elapsedTime * 0.02;
        }
    });

    return (
        <group ref={ref}>
            <Sparkles
                count={500}
                scale={20}
                size={2}
                speed={0.4}
                opacity={0.5}
                color="#6366f1" // Indigo
            />
            <Sparkles
                count={300}
                scale={15}
                size={3}
                speed={0.2}
                opacity={0.3}
                color="#10b981" // Emerald
            />
        </group>
    );
};

const CanvasErrorBoundary = ({ children }) => {
    useEffect(() => {
        const handleContextLoss = (e) => {
            e.preventDefault();
            console.warn('ThreeBackground: WebGL context lost.');
        };
        
        const handleContextRestored = () => {
            console.log('ThreeBackground: WebGL context restored.');
        };

        // Add listeners to window for canvas context events
        window.addEventListener('webglcontextlost', handleContextLoss, true);
        window.addEventListener('webglcontextrestored', handleContextRestored, true);

        return () => {
            window.removeEventListener('webglcontextlost', handleContextLoss, true);
            window.removeEventListener('webglcontextrestored', handleContextRestored, true);
        };
    }, []);

    return children;
};

const ThreeBackground = () => {
    const canvasRef = useRef();

    return (
        <div className="fixed inset-0 z-0 pointer-events-none bg-gradient-to-b from-[#0B1121] to-[#0F172A]">
            <CanvasErrorBoundary>
                <Canvas
                    ref={canvasRef}
                    camera={{ position: [0, 0, 10], fov: 60 }}
                    gl={{ 
                        antialias: false, 
                        alpha: true, 
                        powerPreference: 'low-power',
                        failIfMajorPerformanceCaveat: false
                    }}
                    dpr={[1, 2]}
                    onError={(error) => console.warn('Canvas error:', error)}
                >
                    <fog attach="fog" args={['#0B1121', 5, 20]} />
                    <ambientLight intensity={0.5} />
                    <Float speed={1} rotationIntensity={0.2} floatIntensity={0.2}>
                        <AnimatedParticles />
                    </Float>
                    <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={0.5} />
                    <Preload all />
                </Canvas>
            </CanvasErrorBoundary>
        </div>
    );
};

export default ThreeBackground;
