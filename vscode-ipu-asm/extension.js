'use strict';

// Surface assembler errors as squiggles.
//
// The TextMate grammar colours tokens but cannot report errors — it assigns
// scopes to spans and has no notion of what the grammar expects next. Validity
// is not a lexical property: whether a comma is legal depends on the parser's
// state, not on the characters. So the only thing that can answer "is this
// valid" is the assembler, which this runs.
//
// It shells out to `ipu-as check --json` rather than speaking LSP, which keeps
// the extension dependency-free: no vscode-languageclient, no bundled
// node_modules, and the packaged vsix stays a few kilobytes.

const cp = require('child_process');
const path = require('path');
const vscode = require('vscode');

const LANGUAGE_ID = 'ipu-asm';
const CONFIG_SECTION = 'ipuAsm';

/** Debounce so a burst of keystrokes runs the checker once. */
const DEBOUNCE_MS = 400;

let diagnostics;
let pending;
const running = new Map();

function checkCommand(folder) {
  const configured = vscode.workspace
    .getConfiguration(CONFIG_SECTION, folder && folder.uri)
    .get('checkCommand');
  if (Array.isArray(configured) && configured.length) return configured;

  // Default to Bazel: it is how everything else in this repo is run, and it
  // resolves the assembler's dependencies without the user setting anything up.
  return [
    'bazel',
    'run',
    '--ui_event_filters=-info,-stdout,-stderr',
    '--noshow_progress',
    '//src/tools/ipu-as-py:ipu-as',
    '--',
    'check',
    '--json',
    '--input',
  ];
}

/** Run the checker for one document; resolves to a diagnostics array. */
function runCheck(document) {
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  const cwd = folder ? folder.uri.fsPath : path.dirname(document.uri.fsPath);
  const [command, ...args] = checkCommand(folder);

  return new Promise((resolve) => {
    let child;
    try {
      child = cp.spawn(command, [...args, document.uri.fsPath], { cwd });
    } catch (err) {
      resolve({ error: `could not start ${command}: ${err.message}` });
      return;
    }

    // Supersede an in-flight run for the same document rather than racing it.
    const previous = running.get(document.uri.toString());
    if (previous) previous.kill();
    running.set(document.uri.toString(), child);

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));

    child.on('error', (err) => resolve({ error: `${command}: ${err.message}` }));
    child.on('close', () => {
      running.delete(document.uri.toString());
      // Exit code 1 just means "found problems", so it is not an error here.
      try {
        resolve({ found: JSON.parse(stdout) });
      } catch {
        resolve({
          error: `could not parse checker output: ${(stderr || stdout).trim().slice(0, 300)}`,
        });
      }
    });
  });
}

function toVscodeDiagnostic(raw) {
  const range = new vscode.Range(
    Math.max(raw.line, 0),
    Math.max(raw.column, 0),
    Math.max(raw.end_line, 0),
    Math.max(raw.end_column, 0)
  );
  const message = raw.approximate
    ? `${raw.message}\n(position approximate)`
    : raw.message;

  const diagnostic = new vscode.Diagnostic(
    range,
    message,
    vscode.DiagnosticSeverity.Error
  );
  diagnostic.source = `ipu-as (${raw.stage})`;
  return diagnostic;
}

async function refresh(document) {
  if (!document || document.languageId !== LANGUAGE_ID) return;
  if (document.uri.scheme !== 'file') return; // unsaved/untitled has no path

  const result = await runCheck(document);

  if (result.error) {
    // A missing toolchain should not paint the file red — that would be a
    // diagnostic about the environment masquerading as one about the code.
    console.warn(`[ipu-asm] ${result.error}`);
    diagnostics.delete(document.uri);
    return;
  }

  diagnostics.set(document.uri, result.found.map(toVscodeDiagnostic));
}

function scheduleRefresh(document) {
  clearTimeout(pending);
  pending = setTimeout(() => refresh(document), DEBOUNCE_MS);
}

function activate(context) {
  diagnostics = vscode.languages.createDiagnosticCollection(LANGUAGE_ID);
  context.subscriptions.push(diagnostics);

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(refresh),
    vscode.workspace.onDidSaveTextDocument(refresh),
    vscode.workspace.onDidChangeTextDocument((e) => scheduleRefresh(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => diagnostics.delete(d.uri)),
    vscode.commands.registerCommand('ipuAsm.check', () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) refresh(editor.document);
    })
  );

  vscode.workspace.textDocuments.forEach(refresh);
}

function deactivate() {
  clearTimeout(pending);
  for (const child of running.values()) child.kill();
}

module.exports = { activate, deactivate };
