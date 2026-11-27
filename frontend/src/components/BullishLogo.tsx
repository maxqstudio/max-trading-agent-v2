import React from 'react';

// Max Quant mark: a bull with its head down, charging, horns thrusting upward (the "bullish" pose).
const BODY =
  'M4 14C4 12 6 11 8 11C11 10 13 8 16 7.5C18.5 7.2 20.5 8.5 22 10.5L25.5 13C27 13.6 28.3 15 28.6 16.8L29 19.2C29.1 20.3 28.2 21 27.2 20.7L24.5 19.8C23.5 20.8 22.5 21.6 21.5 22L22 26L20 26L19.2 22.6L18 26.2L16.2 26.2L16.6 22.8C13.5 23.3 10.5 23.2 8.6 22.6L7 26L5 26L5.8 21.6C4.6 20.2 4 17.5 4 14Z';
const HORNS = 'M26.6 14.2C28.6 13.4 29.8 11.4 29.6 8.2M25.2 13.2C26.8 12 27.6 10.2 27.3 7.6';
const TAIL = 'M4.2 13C2.6 13.6 2.1 15.4 2.8 17.4';

export const BullishLogo: React.FC<{ size?: number }> = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="Max Quant" xmlns="http://www.w3.org/2000/svg">
    <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeOpacity="0.5" />
    <path d={BODY} fill="#10B981" />
    <path d={HORNS} fill="none" stroke="#10B981" strokeWidth="1.6" strokeLinecap="round" />
    <path d={TAIL} fill="none" stroke="#10B981" strokeWidth="1.1" strokeLinecap="round" />
  </svg>
);
