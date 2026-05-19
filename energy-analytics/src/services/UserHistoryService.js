import { db } from '../firebase';
import { collection, addDoc, query, where, orderBy, getDocs, updateDoc, doc } from 'firebase/firestore';

const ANALYTICS_COLLECTION = 'user_analytics';

export const UserHistoryService = {
  /**
   * Save prediction results to Firebase
   * @param {string} userId - User's UID
   * @param {object} analyticsData - Object containing prediction data
   */
  savePrediction: async (userId, analyticsData) => {
    try {
      const docRef = await addDoc(collection(db, ANALYTICS_COLLECTION), {
        userId,
        timestamp: new Date(),
        type: 'prediction',
        forecastHorizon: analyticsData.forecastHorizon || 6,
        predictions: analyticsData.predictions || [],
        historicalLoads: analyticsData.historicalLoads || [],
        statistics: {
          average: analyticsData.average,
          peak: analyticsData.peak,
          minimum: analyticsData.minimum,
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
        where('userId', '==', userId),
        orderBy('timestamp', 'desc')
      );

      const snapshot = await getDocs(q);
      return snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })).slice(0, limit);
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
      const docRef = await addDoc(collection(db, 'user_profiles'), {
        userId,
        ...profileData,
        createdAt: new Date(),
        updatedAt: new Date()
      });
      return docRef.id;
    } catch (error) {
      console.error('Error saving user profile:', error);
      throw error;
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
        where('timestamp', '>=', date),
        orderBy('timestamp', 'desc')
      );

      const snapshot = await getDocs(q);
      return snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
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
