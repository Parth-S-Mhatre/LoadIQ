export const getFriendlyAuthError = (error) => {
  const code = error?.code || "";

  switch (code) {
    case "auth/popup-closed-by-user":
      return "Google sign-in was closed before completion.";
    case "auth/cancelled-popup-request":
      return "Another sign-in popup is already open.";
    case "auth/popup-blocked":
      return "Popup was blocked by the browser. Redirecting to Google sign-in...";
    case "auth/unauthorized-domain":
      return "This domain is not authorized in Firebase Auth settings.";
    case "auth/account-exists-with-different-credential":
      return "An account already exists with this email using a different sign-in method.";
    case "auth/network-request-failed":
      return "Network error. Check internet connection and try again.";
    default:
      return "Google sign-in failed. Please try again.";
  }
};
