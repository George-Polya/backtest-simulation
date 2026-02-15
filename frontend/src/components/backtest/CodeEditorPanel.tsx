'use client';

import { useMemo, useState } from 'react';
import Editor from '@monaco-editor/react';
import { Alert, Button, LoadingSpinner } from '@/components/ui';

interface CodeEditorPanelProps {
  code: string;
  isLoading?: boolean;
  onChange?: (code: string) => void;
}

const EMPTY_CODE_PLACEHOLDER = '# Generated strategy code will appear here.';

export function CodeEditorPanel({ code, isLoading = false, onChange }: CodeEditorPanelProps) {
  const [isEditMode, setIsEditMode] = useState(false);

  const editorValue = useMemo(
    () => (code.trim().length > 0 ? code : EMPTY_CODE_PLACEHOLDER),
    [code]
  );

  const isReadOnly = !isEditMode || code.trim().length === 0;

  return (
    <section className="space-y-3" id="code-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-slate-600">
          {code ? 'Generated code is available.' : 'Code has not been generated yet.'}
        </p>
        <Button
          disabled={isLoading || code.trim().length === 0}
          onClick={() => setIsEditMode((previous) => !previous)}
          type="button"
          variant="secondary"
        >
          {isEditMode ? 'Lock Editor' : 'Edit Code'}
        </Button>
      </div>

      {isLoading ? (
        <div className="rounded-lg border border-[var(--border)] bg-slate-950 px-4 py-6">
          <LoadingSpinner centered size="lg" />
          <p className="mt-3 text-center text-sm text-slate-200">Generating code...</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-[var(--border)]">
          <Editor
            height="360px"
            language="python"
            theme="vs-dark"
            value={editorValue}
            onChange={(nextValue) => onChange?.(nextValue ?? '')}
            options={{
              readOnly: isReadOnly,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: 'on',
              fontSize: 13,
              automaticLayout: true
            }}
          />
        </div>
      )}

      {code ? null : (
        <Alert title="No generated code" variant="info">
          Save configuration and click Generate Code to request strategy code.
        </Alert>
      )}
    </section>
  );
}

