export enum ContributionFrequency {
  Monthly = 'monthly',
  Quarterly = 'quarterly',
  SemiAnnual = 'semiannual',
  Annual = 'annual'
}

export interface ContributionPlan {
  frequency: ContributionFrequency;
  amount: number;
}

export interface FeeSettings {
  trading_fee_percent: number;
  slippage_percent: number;
}

export interface LLMSettings {
  provider: string;
  model?: string | null;
  web_search_enabled: boolean;
  seed?: number | null;
  temperature?: number | null;
}

export interface BacktestParams {
  start_date: string;
  end_date: string;
  initial_capital: number;
  contribution: ContributionPlan;
  fees: FeeSettings;
  dividend_reinvestment: boolean;
  benchmarks: string[];
  explicit_tickers?: string[] | null;
  llm_settings: LLMSettings;
  reference_date?: string | null;
}

export interface BacktestRequest {
  strategy: string;
  params: BacktestParams;
}

export interface ModelInfo {
  provider: string;
  model_id: string;
  max_tokens: number;
  supports_system_prompt: boolean;
  cost_per_1k_input: number;
  cost_per_1k_output: number;
}

export interface GeneratedCode {
  code: string;
  strategy_summary: string;
  model_info: ModelInfo;
  tickers: string[];
}

export interface GenerateBacktestResponse {
  generated_code: GeneratedCode;
  tickers_found: string[];
  generation_time_seconds: number;
}

export interface GenerationMetadata {
  model_info: ModelInfo;
  strategy_summary: string;
  tickers_found: string[];
  generation_time_seconds: number;
}

export enum JobStatus {
  Pending = 'pending',
  Running = 'running',
  Completed = 'completed',
  Failed = 'failed',
  Cancelled = 'cancelled',
  Timeout = 'timeout'
}

export interface ExecuteBacktestRequest {
  code?: string;
  code_reference?: string;
  params: BacktestParams;
  timeout?: number;
  async_mode?: boolean;
}

export interface ExecutionResult {
  success: boolean;
  job_id: string;
  status: JobStatus;
  data?: Record<string, unknown> | null;
  error?: string | null;
  logs: string;
  duration_seconds?: number | null;
}

export interface ExecuteBacktestResponse {
  job_id: string;
  status: JobStatus;
  message: string;
  result?: ExecutionResult | null;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
}

export interface ChartDataPoint {
  date: string;
  value: number;
}

export interface PerformanceMetrics {
  total_return: number;
  cagr: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  volatility: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
}

export interface EquityCurveData {
  strategy: ChartDataPoint[];
  benchmark: ChartDataPoint[] | null;
  log_scale: boolean;
}

export interface DrawdownData {
  data: ChartDataPoint[];
}

export interface MonthlyHeatmapData {
  years: number[];
  months: string[];
  returns: Array<Array<number | null>>;
}

export interface BacktestResultResponse {
  job_id: string;
  status: JobStatus;
  metrics: PerformanceMetrics;
  equity_curve: EquityCurveData;
  drawdown: DrawdownData;
  monthly_heatmap: MonthlyHeatmapData;
  trades: Array<Record<string, unknown>>;
  logs: string;
}

export interface EquityChartResponse {
  job_id: string;
  strategy: ChartDataPoint[];
  benchmark: ChartDataPoint[] | null;
  log_scale: boolean;
}

export interface DrawdownChartResponse {
  job_id: string;
  data: ChartDataPoint[];
}

export interface MonthlyReturnsResponse {
  job_id: string;
  years: number[];
  months: string[];
  returns: Array<Array<number | null>>;
}
