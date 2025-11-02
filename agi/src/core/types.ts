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
  safety_level: "T0" | "T1" | "T2" | "T3";
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

export type BranchCondition =
  | string
  | {
      when?: "always" | "never" | "success" | "failure" | "stdout_contains" | string;
      step?: UID;
      value?: string;
    };

export type PlanStep = {
  id: UID;
  tool?: string;
  args?: Record<string, any>;
  safety_level?: "T0" | "T1" | "T2" | "T3";
  description?: string;
  goal?: string;
  sub_steps?: PlanStep[];
  branches?: PlanBranch[];
};

export type PlanBranch = {
  condition?: BranchCondition;
  steps: PlanStep[];
};

export type Plan = {
  id: UID;
  claim_ids: UID[];
  steps: PlanStep[];
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

export type RunContext = {
  working_dir: string;
  timeout_s: number;
  env_whitelist: string[];
  network: "off" | "read" | "write";
  record_provenance: boolean;
};
