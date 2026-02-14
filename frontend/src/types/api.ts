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
  web_search_enabled: boolean;
}

export interface BacktestParams {
  start_date: string;
  end_date: string;
  initial_capital: number;
  benchmarks: string[];
  contribution: ContributionPlan;
  fees: FeeSettings;
  dividend_reinvestment: boolean;
  llm_settings: LLMSettings;
}

export interface BacktestRequest {
  strategy: string;
  params: BacktestParams;
}

export interface ModelInfo {
  provider?: string;
  model_name: string;
  temperature?: number;
  web_search_enabled?: boolean;
}

export interface GeneratedCode {
  code: string;
  strategy_summary?: string;
  model_info?: ModelInfo;
  generation_duration_ms?: number;
  detected_tickers?: string[];
}

export enum JobStatus {
  Pending = 'pending',
  Running = 'running',
  Completed = 'completed',
  Failed = 'failed'
}

export interface ExecuteBacktestRequest {
  code: string;
  params: BacktestParams;
  async_mode: boolean;
}

export interface ExecuteBacktestResponse {
  job_id: string;
  status: JobStatus;
  message?: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress?: number;
  error?: string;
  logs?: string;
}

export interface BacktestMetrics {
  total_return: number;
  cagr: number;
  max_drawdown: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  calmar_ratio?: number;
  benchmark_total_return?: number;
  benchmark_cagr?: number;
  benchmark_max_drawdown?: number;
}

export interface EquityPoint {
  date: string;
  strategy: number;
  benchmark?: number;
}

export interface DrawdownPoint {
  date: string;
  strategy: number;
  benchmark?: number;
}

export interface MonthlyReturnPoint {
  month: string;
  value: number;
}

export interface TradeRecord {
  entry_date: string;
  exit_date: string;
  symbol: string;
  action: string;
  quantity: number;
  entry_price: number;
  exit_price: number;
  profit: number;
  profit_pct: number;
}

export interface BacktestResultResponse {
  job_id: string;
  status: JobStatus;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  drawdown_curve: DrawdownPoint[];
  monthly_returns: MonthlyReturnPoint[];
  trades: TradeRecord[];
  logs?: string;
}

export interface EquityChartResponse {
  job_id: string;
  data: EquityPoint[];
}

export interface DrawdownChartResponse {
  job_id: string;
  data: DrawdownPoint[];
}

export interface MonthlyReturnsResponse {
  job_id: string;
  data: MonthlyReturnPoint[];
}
