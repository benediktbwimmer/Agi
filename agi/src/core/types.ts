export type UID = string;

export type Prediction = {
  id: UID;
  metric: string;
  expectation: string;
  eval_procedure: string;
};

export type Source = {
  kind: "file" | "url" | "dataset" | "paper" | "tool";
  ref: string;
  note?: string;
};

export type Claim = {
  id: UID;
  text: string;
  predictions: Array<Prediction>;
  variables: Record<string, any>;
  provenance?: Source[];
};

export type ToolCall = {
  id: UID;
  tool: string;
  args: Record<string, any>;
  safety_level: "T0" | "T1" | "T2";
};

export type ToolResult = {
  call_id: UID;
  ok: boolean;
  cost_tokens?: number;
  wall_time_ms?: number;
  stdout?: string;
  data?: any;
  figures?: string[];
  provenance: Source[];
};

export type Plan = {
  id: UID;
  claim_ids: UID[];
  steps: ToolCall[];
  expected_cost: { tokens?: number; time_s?: number };
  risks: string[];
  ablations: string[];
};

export type Belief = {
  claim_id: UID;
  credence: number;
  evidence: Source[];
  last_updated: string;
};

export type Report = {
  goal: string;
  summary: string;
  key_findings: string[];
  belief_deltas: Belief[];
  artifacts: string[];
};

export interface Tool {
  name: string;
  safety: "T0" | "T1" | "T2";
  run(args: Record<string, any>, ctx: RunContext): Promise<ToolResult>;
}

export type MemoryEpisode = {
  tool?: string;
  call_id?: UID;
  stdout?: string;
  summary?: string;
  goal?: string;
  time?: string;
  claim_ids?: UID[];
  [key: string]: any;
};

export type EpisodicRecallOptions = {
  tool?: string;
  limit?: number;
  text_query?: string;
};

export type RunContext = {
  working_dir: string;
  timeout_s: number;
  env_whitelist: string[];
  network: "off" | "read" | "write";
  record_provenance: boolean;
  working_memory?: MemoryEpisode[];
  episodic_memory?: unknown;
  recall_from_episodic?: (
    options?: EpisodicRecallOptions
  ) => Promise<MemoryEpisode[]> | MemoryEpisode[];
};
