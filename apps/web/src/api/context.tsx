/* eslint-disable react-refresh/only-export-components -- provider and its typed hook share one private context */
import { createContext, useContext } from "react";
import type { PropsWithChildren } from "react";

import { apiClient } from "./client";
import type { ApiClient } from "./client";

const ApiContext = createContext<ApiClient>(apiClient);

interface ApiProviderProps extends PropsWithChildren {
  client: ApiClient;
}

export function ApiProvider({ client, children }: ApiProviderProps) {
  return <ApiContext.Provider value={client}>{children}</ApiContext.Provider>;
}

export function useApi(): ApiClient {
  return useContext(ApiContext);
}
