import test from 'node:test';
import assert from 'node:assert/strict';
import {mountPersonal} from '../src/personal-view.js';
import {initializeAccount} from '../src/account.js';
import {saveModel,storageKey} from '../src/personal-domain.js';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
const props={catalog:{models:[{id:'model',name:'Example model'}],artifacts:[],benchmarks:[],quality_evaluations:[]},hardware:{items:[]},fetchCompatibility:async()=>({items:[]})};
class Root extends EventTarget{innerHTML='';querySelector(){return null;}}
function setup(t,path){
 const originals=new Map(['localStorage','location','window'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 const values=new Map();
 globalThis.localStorage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 globalThis.location={pathname:path,search:''};globalThis.window={scrollTo(){}};
 t.after(()=>{for(const [key,descriptor] of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
 return values;
}
function saveLocal(){return saveModel({name:'Local assistant',work:'coding',instructions:'Explain this clearly.',examples:[]},{fitting:true,model:{id:'model'},artifact:{id:'artifact',revision:'revision'}},{id:'device'},8192,{id:'local-model'});}

test('pure optimization renders immediately while account initialization is unresolved',t=>{
 setup(t,'/optimize/missing');const root=new Root(),dispose=mountPersonal(root,props);
 assert.match(root.innerHTML,/Choose a model first/);assert.doesNotMatch(root.innerHTML,/Opening your models/);dispose();
});

test('a guest library and saved detail remain accessible when sign-in is unavailable',async t=>{
 setup(t,'/my-models');saveLocal();
 // Node cannot load browser HTTPS modules: exercise the real SDK-failure path.
 await initializeAccount();
 const root=new Root();let dispose=mountPersonal(root,props);await tick();
 assert.match(root.innerHTML,/Local assistant/);assert.match(root.innerHTML,/Saved in this browser/);assert.doesNotMatch(root.innerHTML,/id="account-gate"/);dispose();
 location.pathname='/my-models/local-model';dispose=mountPersonal(root,props);await tick();
 assert.match(root.innerHTML,/Local assistant/);assert.match(root.innerHTML,/Explain this clearly/);assert.match(root.innerHTML,/Edit and update/);dispose();
});

test('a damaged local row shows backup recovery beside healthy records',async t=>{
 const values=setup(t,'/my-models');saveLocal();const good=JSON.parse(values.get(storageKey))[0];const raw=JSON.stringify([good,null]);values.set(storageKey,raw);
 const root=new Root(),dispose=mountPersonal(root,props);await tick();
 assert.match(root.innerHTML,/Local assistant/);assert.match(root.innerHTML,/Download saved-data backup/);assert.match(root.innerHTML,/nothing has been removed/);assert.equal(values.get(storageKey),raw);dispose();
});

test('a malformed encoded personal route has an actionable recovery page',async t=>{
 setup(t,'/my-models/%');const root=new Root(),dispose=mountPersonal(root,props);await tick();
 assert.match(root.innerHTML,/This link is not valid/);assert.match(root.innerHTML,/For your device/);dispose();
});
