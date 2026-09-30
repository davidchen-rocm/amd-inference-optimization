import test from 'node:test';
import assert from 'node:assert/strict';
import {mountTeaching} from '../src/lesson-view.js';
import {saveDraft} from '../src/teach-domain.js';
import {upgradeDraft} from '../src/dataset-flow.js';

const project='10000000-0000-4000-8000-000000000001';
const job='10000000-0000-4000-8000-000000000002';
const models=[{id:'qwen3-0-6b',name:'Qwen3 0.6B',repo_id:'Qwen/Qwen3-0.6B',revision:'a'.repeat(40)},{id:'qwen3-4b',name:'Qwen3 4B',repo_id:'Qwen/Qwen3-4B',revision:'b'.repeat(40)}];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const rows=Array.from({length:3},(_,i)=>({id:'seed-'+i,question:'What is the policy for item '+i+'?',answer:'The reviewed policy for item '+i+' is available on weekdays.',approved:true,origin:'user-authored'}));

class Root{
 innerHTML='';listeners=new Map();
 addEventListener(type,listener){const list=this.listeners.get(type)||[];list.push(listener);this.listeners.set(type,list);}
 removeEventListener(type,listener){this.listeners.set(type,(this.listeners.get(type)||[]).filter(item=>item!==listener));}
 querySelector(selector){if(selector==='[role="alert"]')return {scrollIntoView(){}};return null;}
 querySelectorAll(){return [];}
 button(action){const tag=this.innerHTML.match(new RegExp('<button[^>]*data-lesson="'+action+'"[^>]*>'))?.[0];assert.ok(tag,'button exists: '+action);return {dataset:{lesson:action},disabled:/\sdisabled(?:\s|>)/.test(tag)};}
 async click(action,{force=false}={}){const button=this.button(action);if(force)button.disabled=false;const event={target:{closest:selector=>selector==='[data-lesson]'?button:null}};for(const listener of this.listeners.get('click')||[])await listener(event);}
}

async function setup(t,{user=null,available=true,sourceMode='guided',generationModelId}={}){
 const originals=new Map(['localStorage','location','window','document','history'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 const values=new Map(),storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 Object.assign(globalThis,{localStorage:storage,location:{pathname:'/teach',search:'?draft='+project},window:{scrollTo(){}},document:new EventTarget(),history:{replaceState(){}}});
 document.activeElement=null;
 t.after(()=>{for(const [key,descriptor] of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
 const draft=upgradeDraft({id:project,flowVersion:2,stage:3,task:'answers',goal:'Answer policy questions using our reviewed support material.',sourceMode,seeds:structuredClone(rows),examples:sourceMode==='import'?structuredClone(rows):[],previewConfirmed:true,modelId:'qwen3-0-6b',generationModelId});
 saveDraft(storage,draft,project);
 const submissions=[],saved=[];
 const client={capabilities:async()=>({available,generation:true,training:true,models,reason:available?'':'The GPU worker is unavailable.'}),create:async payload=>{submissions.push(payload);return {id:job,request_id:payload.request_id,project_id:payload.project_id,kind:payload.kind,status:'queued',stage:'queued'};}};
 const root=new Root(),dispose=mountTeaching(root,{catalog:{models:[]},hardware:{items:[]},loadWorkspace:async()=>{},getAccount:()=>({user}),client,saveProject:async draft=>{saved.push(structuredClone(draft));return {...draft,updatedAt:'2026-09-30T00:00:00Z'};}});
 t.after(dispose);await tick();await tick();return {root,submissions,saved};
}

test('guest dataset preparation exposes direct sign-in and never submits generation',async t=>{
 const {root,submissions}=await setup(t);
 assert.match(root.innerHTML,/Generate the full dataset on the GPU/);
 assert.match(root.innerHTML,/data-action="google-signin"/);
 assert.match(root.innerHTML,/Generation model<select/);
 assert.match(root.innerHTML,/value="qwen3-4b" selected/);
 assert.equal(root.button('build-dataset').disabled,true);
 await root.click('build-dataset');assert.equal(submissions.length,0);
 await root.click('build-dataset',{force:true});assert.equal(submissions.length,0);
 assert.match(root.innerHTML,/Sign in with Google to submit or open GPU jobs/);
});

test('signed-in generation submits the displayed 4B default without changing the training base',async t=>{
 const {root,submissions,saved}=await setup(t,{user:{uid:'alice'}});
 assert.equal(root.button('build-dataset').disabled,false);
 await root.click('build-dataset');
 assert.equal(submissions.length,1);assert.equal(submissions[0].kind,'generation');
 assert.equal(submissions[0].input.model_id,'qwen3-4b');
 assert.equal(submissions[0].input.purpose,'dataset');assert.equal(submissions[0].input.target_count,12);
 assert.deepEqual(submissions[0].input.seeds.map(row=>row.answer),rows.map(row=>row.answer));
 assert.equal(saved.at(-1).modelId,'qwen3-0-6b');assert.equal(saved.at(-1).generationModelId,'qwen3-4b');
});

test('an explicit earlier generation-model choice is retained and submitted',async t=>{
 const {root,submissions}=await setup(t,{user:{uid:'alice'},generationModelId:'qwen3-0-6b'});
 assert.match(root.innerHTML,/value="qwen3-0-6b" selected/);
 await root.click('build-dataset');assert.equal(submissions[0].input.model_id,'qwen3-0-6b');
});

test('an unavailable GPU disables generation for signed-in users before any POST',async t=>{
 const {root,submissions}=await setup(t,{user:{uid:'alice'},available:false});
 assert.match(root.innerHTML,/The GPU worker is unavailable/);
 assert.equal(root.button('build-dataset').disabled,true);
 await root.click('build-dataset',{force:true});assert.equal(submissions.length,0);
});

test('guests can check imported examples without sign-in or GPU generation',async t=>{
 const {root,submissions}=await setup(t,{sourceMode:'import'});
 assert.match(root.innerHTML,/Check my full dataset/);assert.equal(root.button('build-dataset').disabled,false);
 await root.click('build-dataset');assert.match(root.innerHTML,/Your complete dataset/);
 assert.equal(submissions.length,0);
});
