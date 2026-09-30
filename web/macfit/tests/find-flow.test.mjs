import test from 'node:test';
import assert from 'node:assert/strict';
import {mountFind} from '../src/find-view.js';
import {mountLocalRun} from '../src/run-view.js';
import {focusAfterNavigation} from '../src/navigation-focus.js';
import {readPreference,writePreference,removePreference} from '../src/browser-storage.js';

const preferenceKeys=['macfit.recommendation.profile','macfit.recommendation.needs','macfit.recommendation.needs-confirmed'];
const deviceId='apple-m4-16';
const hardware={items:[{name:'Apple M4',memory_kind:'unified',configurations:[{id:deviceId,ram_gib:16}]}]};
const artifact={id:'example-gguf',model_id:'example',repo_id:'example/model',revision:'a'.repeat(40),identity_status:'verified',format:'GGUF',files:[{filename:'model.gguf'}],quantization:'Q4_K_M',runtime_family:'llama.cpp',actual_size_bytes:128*1024**2};
const catalog={models:[{id:'example',name:'Example model',hf_id:'example/model',repository_revision:'b'.repeat(40),context_length:16384}],artifacts:[artifact],benchmarks:[],quality_evaluations:[]};
const row={model_id:'example',artifact_id:artifact.id,device_id:deviceId,context_tokens:8192,quantization:'Q4_K_M',runtime:'llama.cpp',verdict:'planning_candidate',runtime_support_status:'family_backend_documented',artifact_bytes:128*1024**2,kv_cache_bytes:64*1024**2,policy_reserve_bytes:128*1024**2,estimated_required_bytes:320*1024**2,headroom_bytes:16*1024**3-320*1024**2};
const proof={confirmed:true,deviceId,confirmedAt:'2026-09-30T12:00:00.000Z'};
const tick=()=>new Promise(resolve=>setImmediate(resolve));

function setup(t){
 const originals=new Map(['localStorage','location','history','window','document','FormData'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 const entries=new Map();
 globalThis.localStorage={getItem:key=>entries.get(key)??null,setItem:(key,value)=>entries.set(key,value),removeItem:key=>entries.delete(key)};
 globalThis.location={pathname:'/for-your-device',search:''};
 globalThis.history={replaceState(){}};
 globalThis.window={scrollTo(){}};
 globalThis.document={activeElement:null,addEventListener(){},removeEventListener(){}};
 globalThis.FormData=class extends Map{constructor(form){super(form.entries);}};
 for(const key of preferenceKeys)removePreference(key);
 t.after(()=>{for(const [key,descriptor] of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
 return entries;
}

// A small DOM boundary fixture: real view code owns markup, events and focus;
// the fixture records which current element receives focus after a render.
function rootFixture(){
 let html='',heading=null;const listeners=new Map();
 const root={isConnected:true,focusLog:[],get innerHTML(){return html;},set innerHTML(value){html=value;const match=html.match(/<h1\b([^>]*)>([\s\S]*?)<\/h1>/);heading=match?{textContent:match[2].replace(/<[^>]*>/g,''),setAttribute(){},focus(options){document.activeElement=this;root.focusLog.push({text:this.textContent,options});}}:null;},
  addEventListener(type,handler){listeners.set(type,handler);},removeEventListener(type,handler){if(listeners.get(type)===handler)listeners.delete(type);},contains(element){return element!=null&&element===heading;},
  querySelector(selector){if(selector==='#find-device')return {value:deviceId};if(selector==='#find-confirm')return {checked:true};if(selector==='[data-find-heading]'||selector==='[data-run-heading]'||selector==='h1')return heading;return null;},
  submit(id,entries=[]){listeners.get('submit')?.({preventDefault(){},target:{id,entries}});},click(action){listeners.get('click')?.({target:{closest:()=>({dataset:{find:action}})}});}};
 return root;
}

test('blocked storage preserves a confirmed device through needs, recommendations and run guide',async t=>{
 setup(t);globalThis.localStorage={getItem(){throw new DOMException('denied','SecurityError');},setItem(){throw new DOMException('full','QuotaExceededError');},removeItem(){throw new DOMException('denied','SecurityError');}};
 let resolve;const request=new Promise(done=>{resolve=done;});const root=rootFixture();let confirmed=null;
 const dispose=mountFind(root,{catalog,hardware,fetchCompatibility:()=>request,onConfirm:config=>{confirmed=config.id;}});
 assert.equal(root.focusLog.length,0,'initial mount must not steal focus');
 root.submit('find-device-form');
 assert.equal(confirmed,deviceId);assert.match(root.innerHTML,/find-needs-form/);assert.match(root.innerHTML,/settings work for this session/);
 assert.equal(root.focusLog.at(-1).text,'What would you like to do?');
 assert.equal(readPreference(preferenceKeys[0],null).deviceId,deviceId);
 root.submit('find-needs-form',[['task','coding'],['language','en'],['preference','balanced'],['context','8192']]);
 assert.match(root.innerHTML,/Finding your fit/);assert.equal(readPreference(preferenceKeys[2],false),true);
 const loadingFocusCount=root.focusLog.length;resolve({items:[row]});await tick();
 assert.match(root.innerHTML,/Run on my computer/);assert.equal(root.focusLog.length,loadingFocusCount+1);
 assert.equal(root.focusLog.at(-1).text,'Your computer. Your shortlist.');
 assert.equal((root.innerHTML.match(/aria-current="step"/g)||[]).length,1);
 dispose();
 location.pathname='/run/'+artifact.id;location.search='?device='+deviceId+'&context=8192';
 const runRoot=rootFixture();const disposeRun=mountLocalRun(runRoot,{catalog,hardware,fetchCompatibility:async()=>({items:[row]})});await tick();
 assert.match(runRoot.innerHTML,/Download launcher/);assert.doesNotMatch(runRoot.innerHTML,/Confirm your computer first/);disposeRun();
});

test('restoring saved recommendations does not steal focus; user edits and retries do',async t=>{
 setup(t);writePreference(preferenceKeys[0],proof);writePreference(preferenceKeys[2],true);
 const root=rootFixture();let calls=0;
 const dispose=mountFind(root,{catalog,hardware,fetchCompatibility:async()=>{if(++calls===2)throw Error('Temporary service failure');return {items:[row]};}});
 await tick();assert.equal(root.focusLog.length,0);
 root.click('needs');assert.equal(root.focusLog.at(-1).text,'What would you like to do?');
 root.submit('find-needs-form',[['context','8192']]);await tick();assert.match(root.innerHTML,/Try again/);assert.match(root.innerHTML,/Temporary service failure/);
 const before=root.focusLog.length;root.click('retry');await tick();
 assert.equal(root.focusLog.length,before+2);assert.match(root.innerHTML,/Run on my computer/);
 root.click('device');assert.equal(root.focusLog.at(-1).text,'A good fit starts with your computer.');
 dispose();
});

test('an aborted result cannot replace the device step after changing computer',async t=>{
 setup(t);writePreference(preferenceKeys[0],proof);writePreference(preferenceKeys[2],true);
 let resolve,signal;const request=new Promise(done=>{resolve=done;});const root=rootFixture();
 const dispose=mountFind(root,{catalog,hardware,fetchCompatibility:(_id,_context,options)=>{signal=options.signal;return request;}});
 root.click('device');assert.equal(signal.aborted,true);const focusCount=root.focusLog.length;
 resolve({items:[row]});await tick();assert.match(root.innerHTML,/find-device-form/);assert.equal(root.focusLog.length,focusCount);dispose();
});

test('navigation to saved recommendations keeps heading focus when asynchronous results replace it',async t=>{
 setup(t);writePreference(preferenceKeys[0],proof);writePreference(preferenceKeys[2],true);
 for(const moveAway of [false,true]){
  let resolve;const request=new Promise(done=>{resolve=done;});const root=rootFixture();
  const dispose=mountFind(root,{catalog,hardware,fetchCompatibility:()=>request});
  const stopFocus=focusAfterNavigation({querySelector:selector=>selector==='main'?root:null});
  assert.equal(document.activeElement,root.querySelector('h1'));
  const outside={outside:true};if(moveAway)document.activeElement=outside;
  resolve({items:[row]});await tick();
  assert.equal(document.activeElement,moveAway?outside:root.querySelector('h1'));
  assert.equal(root.focusLog.length,moveAway?1:2);
  assert.match(root.innerHTML,/Run on my computer/);stopFocus();dispose();
 }
});

test('malformed run links render recovery without throwing or requesting compatibility',t=>{
 setup(t);writePreference(preferenceKeys[0],proof);
 for(const path of ['/run/%','/run/%E0%A4%A','/run/%2Fbad','/run/']){
  location.pathname=path;location.search='?device='+deviceId;const root=rootFixture();let requests=0;
  assert.doesNotThrow(()=>mountLocalRun(root,{catalog,hardware,fetchCompatibility:()=>{requests++;}}));
  assert.match(root.innerHTML,/This version is unavailable/);assert.match(root.innerHTML,/Find another model/);assert.equal(requests,0);
 }
});

test('run loading transfers existing heading focus but does not steal focus from outside the view',async t=>{
 setup(t);writePreference(preferenceKeys[0],proof);location.pathname='/run/'+artifact.id;location.search='?device='+deviceId;
 for(const focusInside of [true,false]){
  let resolve;const request=new Promise(done=>{resolve=done;});const root=rootFixture();
  const dispose=mountLocalRun(root,{catalog,hardware,fetchCompatibility:()=>request});
  if(focusInside)root.querySelector('[data-run-heading]').focus({preventScroll:true});else document.activeElement={outside:true};
  const before=root.focusLog.length;resolve({items:[row]});await tick();
  assert.equal(root.focusLog.length,before+(focusInside?1:0));assert.match(root.innerHTML,/Download launcher/);dispose();
 }
});

test('preference fallback respects failed deletes and returns to durable storage after recovery',t=>{
 const entries=setup(t);const key='find-flow-test-preference';writePreference(key,{value:'old'});
 const working=localStorage;globalThis.localStorage={getItem:working.getItem,setItem(){throw Error('blocked');},removeItem(){throw Error('blocked');}};
 assert.equal(writePreference(key,{value:'new'}),false);assert.deepEqual(readPreference(key,null),{value:'new'});
 assert.equal(removePreference(key),false);assert.equal(readPreference(key,'empty'),'empty');assert.ok(entries.has(key));
 globalThis.localStorage=working;assert.equal(writePreference(key,{value:'recovered'}),true);
 entries.set(key,JSON.stringify({value:'other tab'}));assert.deepEqual(readPreference(key,null),{value:'other tab'});
 assert.equal(removePreference(key),true);assert.equal(readPreference(key,'empty'),'empty');
 entries.set(key,'broken JSON');assert.equal(readPreference(key,'empty'),'empty');
});
