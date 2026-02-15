/**
 * Authentication store using Zustand.
 *
 * Manages authentication state including access token, user info,
 * and authentication-related operations.
 */

import { create } from "zustand";

export interface User {
  id: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

interface AuthState {
  // State
  accessToken: string | null;
  refreshToken: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setAccessToken: (token: string | null) => void;
  setRefreshToken: (token: string | null) => void;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshAccessToken: () => Promise<boolean>;
  initialize: () => Promise<void>;
  clearAuth: () => void;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000/api/v1";

export const useAuthStore = create<AuthState>((set, get) => ({
  // Initial state
  accessToken: null,
  refreshToken: null,
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,

  // Actions
  setAccessToken: (token) => {
    set({ accessToken: token, isAuthenticated: !!token });
    if (token) {
      localStorage.setItem("accessToken", token);
    } else {
      localStorage.removeItem("accessToken");
    }
  },

  setRefreshToken: (token) => {
    set({ refreshToken: token });
    if (token) {
      // In production, store in HttpOnly cookie
      localStorage.setItem("refreshToken", token);
    } else {
      localStorage.removeItem("refreshToken");
    }
  },

  setUser: (user) => set({ user }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  login: async (email, password) => {
    set({ isLoading: true, error: null });

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Login failed");
      }

      const data = await response.json();

      set({
        accessToken: data.access_token,
        refreshToken: data.refresh_token,
        isAuthenticated: true,
        isLoading: false,
      });

      // Store tokens
      localStorage.setItem("accessToken", data.access_token);
      localStorage.setItem("refreshToken", data.refresh_token);

      // Fetch current user
      await get().initialize();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Login failed";
      set({ error: errorMessage, isLoading: false });
      throw error;
    }
  },

  logout: async () => {
    const { accessToken } = get();

    try {
      // Call logout endpoint
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
    } catch {
      // Ignore logout errors
    } finally {
      get().clearAuth();
    }
  },

  refreshAccessToken: async () => {
    const { refreshToken } = get();

    if (!refreshToken) {
      return false;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        get().clearAuth();
        return false;
      }

      const data = await response.json();

      set({
        accessToken: data.access_token,
        isAuthenticated: true,
      });

      localStorage.setItem("accessToken", data.access_token);
      return true;
    } catch {
      get().clearAuth();
      return false;
    }
  },

  initialize: async () => {
    const accessToken = localStorage.getItem("accessToken");
    const refreshToken = localStorage.getItem("refreshToken");

    if (!accessToken) {
      set({ isLoading: false });
      return;
    }

    set({ accessToken, refreshToken, isAuthenticated: true });

    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });

      if (!response.ok) {
        // Token might be expired, try refresh
        const refreshed = await get().refreshAccessToken();
        if (!refreshed) {
          get().clearAuth();
          return;
        }

        // Retry after refresh
        const newToken = get().accessToken;
        const retryResponse = await fetch(`${API_BASE_URL}/auth/me`, {
          headers: {
            Authorization: `Bearer ${newToken}`,
          },
        });

        if (!retryResponse.ok) {
          get().clearAuth();
          return;
        }

        const userData = await retryResponse.json();
        set({ user: userData, isLoading: false });
        return;
      }

      const userData = await response.json();
      set({ user: userData, isLoading: false });
    } catch {
      get().clearAuth();
    }
  },

  clearAuth: () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    set({
      accessToken: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });
  },
}));
