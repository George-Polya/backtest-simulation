'use client';

import { useBacktestActions } from '@/hooks';
import { useBacktestStore } from '@/stores';
import { Alert, Button, Card } from '@/components/ui';
import { ConfigForm } from './ConfigForm';

export function BacktestWorkspace() {
  const {
    generatedCode,
    jobId,
    jobStatus,
    requestConfig,
    results,
    uiToggles,
    setUiToggle,
    setSelectedTab
  } = useBacktestStore();
  const { resetState, saveConfig } = useBacktestActions();

  const handleToggleCodePanel = () => {
    const next = !uiToggles.isCodeEditorOpen;
    setUiToggle('isCodeEditorOpen', next);

    if (!next && uiToggles.selectedTab === 'code' && uiToggles.isResultsOpen) {
      setSelectedTab('results');
    }
    if (next && !uiToggles.isResultsOpen) {
      setSelectedTab('code');
    }
  };

  const handleToggleResultsPanel = () => {
    const next = !uiToggles.isResultsOpen;
    setUiToggle('isResultsOpen', next);

    if (!next && uiToggles.selectedTab === 'results' && uiToggles.isCodeEditorOpen) {
      setSelectedTab('code');
    }
    if (next && !uiToggles.isCodeEditorOpen) {
      setSelectedTab('results');
    }
  };

  return (
    <div className="grid gap-4 md:grid-cols-2" id="workspace">
      <Card className="h-fit" title="Configuration & Actions">
        <div className="mb-4 flex flex-wrap gap-2">
          <Button
            type="button"
            variant="secondary"
            onClick={() => setUiToggle('isConfigExpanded', !uiToggles.isConfigExpanded)}
          >
            {uiToggles.isConfigExpanded ? 'Collapse Config' : 'Expand Config'}
          </Button>
          <Button type="button" variant="secondary" onClick={resetState}>
            Reset
          </Button>
        </div>

        {uiToggles.isConfigExpanded ? (
          <div className="space-y-3">
            <ConfigForm onSubmitConfig={saveConfig} />
            {requestConfig ? (
              <Alert title="Ready for code generation" variant="info">
                Configuration is saved with {requestConfig.params.benchmarks.length} benchmark(s).
              </Alert>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-slate-600">Configuration panel is collapsed.</p>
        )}
      </Card>

      <Card title="Code & Results">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <Button type="button" variant="secondary" onClick={handleToggleCodePanel}>
            {uiToggles.isCodeEditorOpen ? 'Hide Code' : 'Show Code'}
          </Button>
          <Button type="button" variant="secondary" onClick={handleToggleResultsPanel}>
            {uiToggles.isResultsOpen ? 'Hide Results' : 'Show Results'}
          </Button>
        </div>

        <div className="mb-4 flex items-center gap-2">
          <Button
            disabled={!uiToggles.isCodeEditorOpen}
            onClick={() => setSelectedTab('code')}
            type="button"
            variant={uiToggles.selectedTab === 'code' ? 'primary' : 'secondary'}
          >
            Code
          </Button>
          <Button
            disabled={!uiToggles.isResultsOpen}
            onClick={() => setSelectedTab('results')}
            type="button"
            variant={uiToggles.selectedTab === 'results' ? 'primary' : 'secondary'}
          >
            Results
          </Button>
        </div>

        {!uiToggles.isCodeEditorOpen && !uiToggles.isResultsOpen ? (
          <Alert title="Panels Hidden" variant="warning">
            Enable at least one panel to view output.
          </Alert>
        ) : null}

        {uiToggles.selectedTab === 'code' && uiToggles.isCodeEditorOpen ? (
          <section className="space-y-3" id="code-panel">
            <p className="text-sm text-slate-600">{generatedCode ? 'Generated code is available.' : 'Code has not been generated yet.'}</p>
            <pre className="max-h-[360px] overflow-auto rounded-lg border border-[var(--border)] bg-slate-950 p-4 text-xs text-slate-100">
              <code>{generatedCode || '# Generated strategy code will appear here.'}</code>
            </pre>
          </section>
        ) : null}

        {uiToggles.selectedTab === 'results' && uiToggles.isResultsOpen ? (
          <section className="space-y-3" id="results-panel">
            <dl className="grid gap-2 text-sm text-slate-700">
              <div>
                <dt className="font-medium">Job ID</dt>
                <dd>{jobId ?? '-'}</dd>
              </div>
              <div>
                <dt className="font-medium">Job Status</dt>
                <dd>{jobStatus ?? '-'}</dd>
              </div>
              <div>
                <dt className="font-medium">Result Payload</dt>
                <dd>{results ? 'Available' : 'Not available'}</dd>
              </div>
            </dl>
            <Alert title="Results Placeholder" variant="info">
              Charts and detailed metrics will be implemented in later tasks.
            </Alert>
          </section>
        ) : null}
      </Card>
    </div>
  );
}
