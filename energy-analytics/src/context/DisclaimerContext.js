import React, { createContext, useContext, useState, useEffect } from "react";

const DisclaimerContext = createContext();

export function useDisclaimer() {
  return useContext(DisclaimerContext);
}

export function DisclaimerProvider({ children }) {
  const [disclaimerDismissed, setDisclaimerDismissed] = useState(false);

  useEffect(() => {
    // Check if disclaimer was already dismissed in this session
    const sessionDismissed = sessionStorage.getItem("disclaimerDismissed");
    if (sessionDismissed === "true") {
      setDisclaimerDismissed(true);
    }
  }, []);

  const dismissDisclaimer = () => {
    sessionStorage.setItem("disclaimerDismissed", "true");
    setDisclaimerDismissed(true);
  };

  const value = {
    disclaimerDismissed,
    dismissDisclaimer
  };

  return (
    <DisclaimerContext.Provider value={value}>
      {children}
    </DisclaimerContext.Provider>
  );
}
