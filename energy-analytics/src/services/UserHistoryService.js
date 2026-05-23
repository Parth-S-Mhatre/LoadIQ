import { db } from '../firebase';
import { collection, addDoc, query, where, getDocs, doc, getDoc, setDoc } from 'firebase/firestore';

const ANALYTICS_COLLECTION = 'user_analytics';
const PROFILE_COLLECTION = 'user_profiles';
const LEGACY_USERS_COLLECTION = 'users';

const removeUndefinedValues = (value) => {
  if (Array.isArray(value)) {
    return value.map(removeUndefinedValues);
  }

  if (value && typeof value === 'object' && typeof value.toDate !== 'function') {
    return Object.entries(value).reduce((cleaned, [key, item]) => {
      if (item !== undefined) {
        cleaned[key] = removeUndefinedValues(item);
      }
      return cleaned;
    }, {});
  }

  return value;
};

const isOfflineFirestoreError = (error) => {
  return error?.code === 'unavailable' || String(error?.message || '').toLowerCase().includes('client is offline');
};

const toTimestampValue = (value) => {
  if (!value) {
    return 0;
  }

  if (typeof value?.toDate === 'function') {
    return value.toDate().getTime();
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 0 : date.getTime();
};

export const UserHistoryService = {
  /**
   * Save prediction results to Firebase
   * @param {string} userId - User's UID
   * @param {object} analyticsData - Object containing prediction data
   */
  savePrediction: async (userId, analyticsData) => {
    try {
      const timestamp = new Date();
      const predictionSeries = Array.isArray(analyticsData.predictions)
        ? analyticsData.predictions.map((value, index) => ({
            step: index + 1,
            value: Number(value)
          })).filter((item) => Number.isFinite(item.value))
        : [];

      const docRef = await addDoc(collection(db, ANALYTICS_COLLECTION), {
        userId,
        timestamp,
        timestampIso: timestamp.toISOString(),
        type: 'prediction',
        forecastHorizon: analyticsData.forecastHorizon || 6,
        model: analyticsData.model || 'ensemble',
        predictionSource: analyticsData.predictionSource || 'ml_model',
        fallbackUsed: Boolean(analyticsData.fallbackUsed),
        fallbackReason: analyticsData.fallbackReason || null,
        predictions: predictionSeries.map((item) => item.value),
        predictionSeries,
        historicalLoads: analyticsData.historicalLoads || [],
        latestPrediction: analyticsData.latestPrediction ?? predictionSeries[predictionSeries.length - 1]?.value ?? null,
        nextHourPrediction: analyticsData.nextHourPrediction ?? predictionSeries[0]?.value ?? null,
        statistics: {
          average: analyticsData.average,
          peak: analyticsData.peak,
          minimum: analyticsData.minimum,
          lastActual: analyticsData.lastActual,
          firstPrediction: analyticsData.firstPrediction,
          delta: analyticsData.delta,
          direction: analyticsData.direction
        },
        status: 'completed'
      });
      return docRef.id;
    } catch (error) {
      console.error('Error saving prediction to Firebase:', error);
      throw error;
    }
  },

  /**
   * Get user's analytics history
   * @param {string} userId - User's UID
   * @param {number} limit - Number of records to fetch (default: 50)
   */
  getUserAnalytics: async (userId, limit = 50) => {
    try {
      const q = query(
        collection(db, ANALYTICS_COLLECTION),
        where('userId', '==', userId)
      );

      const snapshot = await getDocs(q);
      return snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }))
        .sort((a, b) => toTimestampValue(b.timestamp || b.timestampIso) - toTimestampValue(a.timestamp || a.timestampIso))
        .slice(0, limit);
    } catch (error) {
      console.error('Error fetching user analytics:', error);
      return [];
    }
  },

  /**
   * Save user profile/settings
   * @param {string} userId - User's UID
   * @param {object} profileData - User profile information
   */
  saveUserProfile: async (userId, profileData) => {
    try {
      const profileRef = doc(db, PROFILE_COLLECTION, userId);
      const timestamp = new Date();
      const profilePayload = removeUndefinedValues({
        uid: userId,
        userId,
        ...profileData,
        createdAt: profileData?.createdAt || profileData?.createdAtIso || timestamp,
        updatedAt: timestamp,
        updatedAtIso: timestamp.toISOString()
      });

      await setDoc(profileRef, profilePayload, { merge: true });

      try {
        await setDoc(doc(db, LEGACY_USERS_COLLECTION, userId), profilePayload, { merge: true });
      } catch (mirrorError) {
        console.warn('Profile saved to user_profiles but users mirror failed:', mirrorError);
      }

      return profileRef.id;
    } catch (error) {
      if (!isOfflineFirestoreError(error)) {
        console.error('Error saving user profile:', error);
      }
      throw error;
    }
  },

  getUserProfile: async (userId) => {
    try {
      const profileSnapshot = await getDoc(doc(db, PROFILE_COLLECTION, userId));
      if (profileSnapshot.exists()) {
        return profileSnapshot.data();
      }

      const legacyProfileSnapshot = await getDoc(doc(db, LEGACY_USERS_COLLECTION, userId));
      return legacyProfileSnapshot.exists() ? legacyProfileSnapshot.data() : null;
    } catch (error) {
      if (!isOfflineFirestoreError(error)) {
        console.error('Error fetching user profile:', error);
      }
      return null;
    }
  },

  /**
   * Get user's analytics with filters
   * @param {string} userId - User's UID
   * @param {number} daysBack - Number of days to fetch (default: 30)
   */
  getUserAnalyticsForPeriod: async (userId, daysBack = 30) => {
    try {
      const date = new Date();
      date.setDate(date.getDate() - daysBack);

      const q = query(
        collection(db, ANALYTICS_COLLECTION),
        where('userId', '==', userId),
        where('timestamp', '>=', date)
      );

      const snapshot = await getDocs(q);
      return snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }))
        .sort((a, b) => toTimestampValue(b.timestamp || b.timestampIso) - toTimestampValue(a.timestamp || a.timestampIso));
    } catch (error) {
      console.error('Error fetching analytics for period:', error);
      return [];
    }
  },

  /**
   * Save troubleshooting/feedback report
   * @param {string} userId - User's UID
   * @param {object} reportData - Report details
   */
  saveReport: async (userId, reportData) => {
    try {
      const docRef = await addDoc(collection(db, 'user_reports'), {
        userId,
        ...reportData,
        timestamp: new Date(),
        status: 'open'
      });
      return docRef.id;
    } catch (error) {
      console.error('Error saving report:', error);
      throw error;
    }
  }
};
