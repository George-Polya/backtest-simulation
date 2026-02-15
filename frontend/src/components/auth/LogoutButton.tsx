/**
 * LogoutButton component.
 *
 * Provides logout functionality for authenticated users.
 */

"use client";

import { useAuthStore } from "@/stores";

interface LogoutButtonProps {
  className?: string;
}

export function LogoutButton({ className = "" }: LogoutButtonProps) {
  const { logout } = useAuthStore();

  const handleLogout = async () => {
    await logout();
    // Redirect to login page
    window.location.href = "/login";
  };

  return (
    <button
      onClick={handleLogout}
      className={`text-sm text-gray-600 hover:text-gray-900 ${className}`}
    >
      Logout
    </button>
  );
}
