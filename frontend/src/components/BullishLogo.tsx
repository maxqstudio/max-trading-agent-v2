import React from 'react';

// Bullish mark: a simple bull head whose horns double as an up-trend line.
export const BullishLogo: React.FC<{ size?: number }> = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="Bullish" xmlns="http://www.w3.org/2000/svg">
    <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeOpacity="0.5" />
    {/* horns: rising curves */}
    <path d="M5 9.5c1.6 3.4 4.2 5 7.2 5.2" fill="none" stroke="#10B981" strokeWidth="2.2" strokeLinecap="round" />
    <path d="M27 9.5c-1.6 3.4-4.2 5-7.2 5.2" fill="none" stroke="#10B981" strokeWidth="2.2" strokeLinecap="round" />
    {/* head */}
    <path d="M11 13.8h10l-1.3 8.6c-.3 1.9-1.9 3.3-3.7 3.3s-3.4-1.4-3.7-3.3z" fill="#10B981" />
    {/* nostrils */}
    <circle cx="14.4" cy="22.3" r="0.9" fill="#0A101B" />
    <circle cx="17.6" cy="22.3" r="0.9" fill="#0A101B" />
    {/* eyes */}
    <circle cx="13.6" cy="17" r="0.9" fill="#0A101B" />
    <circle cx="18.4" cy="17" r="0.9" fill="#0A101B" />
  </svg>
);
