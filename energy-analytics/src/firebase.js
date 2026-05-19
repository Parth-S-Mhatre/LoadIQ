// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, setPersistence, browserLocalPersistence } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDyQnd2O5vaIiCACVZM3Nbpyc_TIkgnyMY",
  authDomain: "loadiq-smart-ai.firebaseapp.com",
  projectId: "loadiq-smart-ai",
  storageBucket: "loadiq-smart-ai.firebasestorage.app",
  messagingSenderId: "456244141760",
  appId: "1:456244141760:web:a4575d3f06534ee3fcd725",
  measurementId: "G-HXP3YPLK6S"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export Firebase Authentication instance
export const auth = getAuth(app);
auth.useDeviceLanguage();

// Keep users signed in across reloads/browser restarts
setPersistence(auth, browserLocalPersistence).catch((error) => {
  console.warn("Auth persistence setup failed:", error);
});

// Shared Google provider config
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({
  prompt: "select_account"
});

// Export Firestore instance for database operations
export const db = getFirestore(app);
