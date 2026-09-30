import test from 'node:test';
import assert from 'node:assert/strict';
import {apiGet, loadCatalog} from '../src/data.js';

test('catalog startup does not depend on the unused summary endpoint', async t => {
  const requested = [];
  const data = {source:'postgresql',schema_version:1,categories:[],models:[],devices:[],benchmarks:[],sources:[],runtime_requirements:[],artifacts:[]};
  t.mock.method(globalThis, 'fetch', async path => {
    requested.push(path);
    if (path === '/api/catalog') return Response.json(data);
    if (path === '/api/chips') return Response.json({source:'postgresql',items:[],contexts:[8192]});
    throw new Error('The unrelated summary service is down');
  });
  assert.deepEqual(await loadCatalog(), data);
  assert.deepEqual(requested.sort(), ['/api/catalog','/api/chips']);
});

test('network failures have a useful user-facing retry message', async t => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(apiGet('/api/catalog'), /Check your connection and try again/);
});

test('leaving a page preserves request cancellation rather than showing an error', async t => {
  t.mock.method(globalThis, 'fetch', async (_, {signal}) => new Promise((resolve,reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Canceled','AbortError')), {once:true});
  }));
  const controller = new AbortController();
  const pending = apiGet('/api/catalog', {signal:controller.signal});
  controller.abort();
  await assert.rejects(pending, {name:'AbortError'});
});

test('invalid JSON and unsupported response shapes fail recoverably', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>Unavailable</html>'));
  await assert.rejects(apiGet('/api/catalog'), /invalid response/);
});
