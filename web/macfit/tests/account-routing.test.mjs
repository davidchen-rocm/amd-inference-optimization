import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createAccountService} from '../src/account.js';

// Exercise the actual entrypoint listener without starting catalog HTTP requests.
const source=await readFile(new URL('../src/app.js',import.meta.url),'utf8');
const start=source.indexOf("document.addEventListener('macfit:account-changed',");
assert.ok(start>=0,'account navigation listener must exist');
const registration=source.slice(start,source.indexOf('void initializeAccount();',start));
const register=new Function('document','catalog','location','navigate',registration);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function fixture(path){
 const events=new EventTarget(),navigations=[],observers=[],auth={currentUser:null};let offline=false;
 const modules={app:{getApps:()=>[],initializeApp:()=>({})},auth:{getAuth:()=>auth,onAuthStateChanged(_auth,next,error){observers.push({next,error});return()=>{};}}};
 register(events,{models:[{id:'example'}]},{pathname:path},href=>navigations.push(href));
 const service=createAccountService({events,getStorage:()=>({getItem:()=>null,setItem(){},removeItem(){}}),timeoutMs:100,
  loadModule:async name=>{if(offline)throw new TypeError('Failed to fetch dynamically imported module: https://www.gstatic.com/firebase-app.js');return modules[name];}});
 return {service,navigations,observers,set offline(value){offline=value;},async restore(user){const operation=service.initializeAccount();await tick();auth.currentUser=user;observers.at(-1).next(user);await operation;}};
}

test('ordinary initial account restore preserves a project deep link',async()=>{
 const f=fixture('/my-models/existing-project');await f.restore({uid:'alice'});assert.deepEqual(f.navigations,[]);
});

test('restoring an account after SDK failure replaces the previously opened local workspace',async()=>{
 const f=fixture('/teach');f.offline=true;await f.service.initializeAccount();assert.equal(f.service.getAccount().phase,'error');
 f.offline=false;await f.restore({uid:'alice'});assert.deepEqual(f.navigations,['/my-models']);
});

test('restoring the same account after a session error refreshes its failed workspace',async()=>{
 const f=fixture('/my-models');await f.restore({uid:'alice'});f.observers[0].error({code:'auth/network-request-failed'});
 await f.restore({uid:'alice'});assert.deepEqual(f.navigations,['/my-models']);
});

test('recovering a guest session refreshes an existing workspace, but leaves device finding alone',async()=>{
 for(const path of ['/my-models','/for-your-device']){
  const f=fixture(path);f.offline=true;await f.service.initializeAccount();f.offline=false;await f.restore(null);
  assert.deepEqual(f.navigations,path==='/my-models'?['/my-models']:[]);
 }
});
