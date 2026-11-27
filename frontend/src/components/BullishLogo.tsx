import React from 'react';

// App logo files live in frontend/public: logo.png (bull icon, header), favicon.png (browser tab)
// and logo-full.png (bull + "Max Trading Agent" wordmark). Replace those files to change the logo.
export const LOGO_URL = '/logo.png';
export const LOGO_FULL_URL = '/logo-full.png';

export const BullishLogo: React.FC<{ size?: number }> = ({ size = 32 }) => (
  <img src={LOGO_URL} width={size} height={size} alt="Max Quant" className="shrink-0 object-contain" />
);
