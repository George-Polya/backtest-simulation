import { spawn } from 'node:child_process';
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const standaloneDir = path.join(rootDir, '.next', 'standalone');
const serverPath = path.join(standaloneDir, 'server.js');
const staticSourceDir = path.join(rootDir, '.next', 'static');
const staticTargetDir = path.join(standaloneDir, '.next', 'static');
const publicSourceDir = path.join(rootDir, 'public');
const publicTargetDir = path.join(standaloneDir, 'public');

function fail(message) {
  console.error(message);
  process.exit(1);
}

if (!existsSync(serverPath)) {
  fail('Missing .next/standalone/server.js. Run `npm run build` before starting Playwright.');
}

if (!existsSync(staticSourceDir)) {
  fail('Missing .next/static assets. Run `npm run build` before starting Playwright.');
}

mkdirSync(path.dirname(staticTargetDir), { recursive: true });
cpSync(staticSourceDir, staticTargetDir, { force: true, recursive: true });

if (existsSync(publicSourceDir)) {
  cpSync(publicSourceDir, publicTargetDir, { force: true, recursive: true });
}

const child = spawn(process.execPath, [serverPath], {
  cwd: rootDir,
  env: process.env,
  stdio: 'inherit'
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    child.kill(signal);
  });
}

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 0);
});
