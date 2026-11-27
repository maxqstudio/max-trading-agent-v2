import React from 'react';

// Max Quant mark: bull and bear locked in a yin-yang circle. The red bear holds the left,
// the green bull rises on the right with its horns thrusting up.
export const BullishLogo: React.FC<{ size?: number }> = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Max Quant" xmlns="http://www.w3.org/2000/svg">
    <circle cx="32" cy="32" r="31" fill="#0A101B"/>
    <circle cx="32" cy="32" r="27.5" fill="#10B981"/>
    <path d="M51.45 12.55A27.5 27.5 0 0 0 12.55 51.45A13.75 13.75 0 0 0 32 32A13.75 13.75 0 0 1 51.45 12.55Z" fill="#EF4444"/>
    <path d="M51.45 12.55A13.75 13.75 0 0 0 32 32A13.75 13.75 0 0 1 12.55 51.45" fill="none" stroke="#0A101B" strokeWidth="2.2"/>
    <g fill="#0A101B">
      <circle cx="11.6" cy="16.6" r="3.8"/>
      <circle cx="24.4" cy="14" r="3.8"/>
      <path d="M9.4 25C8.6 18.6 13 14.6 18.6 14.6C24.6 14.6 28.8 18.6 28.2 24.6C27.8 29.6 24 33.8 18.8 34C13.6 34.2 10 30.4 9.4 25Z"/>
    </g>
    <circle cx="11.6" cy="16.6" r="1.6" fill="#EF4444"/>
    <circle cx="24.4" cy="14" r="1.6" fill="#EF4444"/>
    <path d="M13.2 21.4L16.8 22.8M24.4 20.6L21 22.4" stroke="#EF4444" strokeWidth="1.5" strokeLinecap="round"/>
    <ellipse cx="18.9" cy="28" rx="5.4" ry="4.2" fill="#EF4444"/>
    <ellipse cx="18.9" cy="25.6" rx="1.8" ry="1.1" fill="#0A101B"/>
    <path d="M15.6 28C17.4 27.2 20.6 27.2 22.2 28C21.8 30.4 20.6 31.4 18.9 31.4C17.2 31.4 16 30.4 15.6 28Z" fill="#0A101B"/>
    <path d="M16.6 27.9L17.3 29.6L18.1 27.7ZM21.2 27.9L20.5 29.6L19.7 27.7Z" fill="#FFFFFF"/>
    <g fill="#0A101B">
      <path d="M36.6 40.2C30.6 39.4 28.4 33.4 30.8 26.8C31.2 32.2 33.6 35.6 38 36.8Z"/>
      <path d="M49.4 40.2C55.4 39.4 57.6 33.4 55.2 26.8C54.8 32.2 52.4 35.6 48 36.8Z"/>
      <path d="M36 37.6C40.4 36 45.6 36 50 37.6L48.6 49.6C48.2 53 45.8 55 43 55C40.2 55 37.8 53 37.4 49.6Z"/>
      <path d="M36.4 40L31.6 41.8L37 44Z"/>
      <path d="M49.6 40L54.4 41.8L49 44Z"/>
    </g>
    <path d="M38.6 42.4L41.6 43.6M47.4 42.4L44.4 43.6" stroke="#10B981" strokeWidth="1.5" strokeLinecap="round"/>
    <circle cx="43" cy="52.6" r="2.4" fill="none" stroke="#FFFFFF" strokeWidth="1.2"/>
  </svg>
);
