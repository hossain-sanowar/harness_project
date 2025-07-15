// contexts/DesignContext.js
import { createContext, useState } from 'react';

export const DesignContext = createContext();

export function DesignProvider({ children }) {
  const [harnessConfig, setHarnessConfig] = useState({});
  const [results, setResults] = useState(null);
  
  return (
    <DesignContext.Provider value={{ harnessConfig, setHarnessConfig, results, setResults }}>
      {children}
    </DesignContext.Provider>
  );
}