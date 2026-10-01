import test from 'node:test';
import assert from 'node:assert/strict';
import {mountTeaching} from '../src/lesson-view.js';
import {saveDraft} from '../src/teach-domain.js';
import {upgradeDraft} from '../src/dataset-flow.js';

const project='10000000-0000-4000-8000-000000000001';
const current='10000000-0000-4000-8000-000000000002';
const older='10000000-0000-4000-8000-000000000003';
const request='10000000-0000-4000-8000-000000000004';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const ref={requestId:request,projectId:project,jobId:current,kind:'training',modelId:'qwen3-0-6b',ownerUid:'alice',status:'succeeded',stage:'complete',preset:'quick'};
const completed=(id=current)=>({id,request_id:request,project_id:project,kind:'training',status:'succeeded',stage:'complete',created_at:'2026-09-30T18:00:00Z',updated_at:'2026-09-30T18:10:00Z',base_model:{repo_id:'Qwen/Qwen3-0.6B'},result:{evaluation:{samples:[{question:'A held-out policy question?',expected:'The reviewed policy',before:'Original response',after:'Real adapter response'}]}},artifacts:[{id:'adapter-tar-gz',name:'adapter.tar.gz'}]});

class Root{
 _html='';listeners=new Map();opened=new Set();
 get innerHTML(){return this._html;}
 set innerHTML(value){this._html=value;this.opened.clear();}
 openHistory(id){assert.ok(this.innerHTML.includes('data-history="'+id+'"'));this.opened.add(id);}
 addEventListener(type,listener){const list=this.listeners.get(type)||[];list.push(listener);this.listeners.set(type,list);}
 removeEventListener(type,listener){this.listeners.set(type,(this.listeners.get(type)||[]).filter(item=>item!==listener));}
 querySelector(selector){const history=selector.match(/^\[data-history="([^"]+)"\]$/)?.[1];if(history&&this.innerHTML.includes('data-history="'+history+'"'))return {setAttribute:()=>this.opened.add(history)};if(selector==='[role="alert"]')return {scrollIntoView(){}};return null;}
 querySelectorAll(selector){return selector==='[data-history][open]'?[...this.opened].map(id=>({dataset:{history:id}})):[];}
 async click(action,match={}){
  const tags=[...this.innerHTML.matchAll(/<button\b[^>]*>/g)].map(row=>row[0]);
  const tag=tags.find(tag=>tag.includes('data-lesson="'+action+'"')&&Object.entries(match).every(([key,value])=>tag.includes('data-'+key+'="'+value+'"')));
  assert.ok(tag,'button exists: '+action);const dataset=Object.fromEntries([...tag.matchAll(/data-([\w-]+)="([^"]*)"/g)].map(row=>[row[1],row[2]]));
  const button={dataset,disabled:/\sdisabled(?:\s|>)/.test(tag)};
  for(const listener of this.listeners.get('click')||[])await listener({target:{closest:selector=>selector==='[data-lesson]'?button:null}});
 }
}

async function setup(t,{user={uid:'alice'},jobRef=ref,getJob=completed(),listJobs=[completed(older)],listError=null}={}){
 const originals=new Map(['localStorage','location','window','document','history'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 const objectURL=URL.createObjectURL,revokeURL=URL.revokeObjectURL,values=new Map(),storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 const downloaded=[],blobs=[];let identity=user;
 Object.assign(globalThis,{localStorage:storage,location:{pathname:'/teach',search:'?draft='+project},window:{scrollTo(){}},document:new EventTarget(),history:{replaceState(){}}});
 document.activeElement=null;document.body={append(){}};document.createElement=()=>({click(){downloaded.push(this.download);},remove(){}});
 URL.createObjectURL=blob=>{blobs.push(blob);return 'blob:test';};URL.revokeObjectURL=()=>{};
 t.after(()=>{URL.createObjectURL=objectURL;URL.revokeObjectURL=revokeURL;for(const [key,descriptor] of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
 saveDraft(storage,upgradeDraft({id:project,gpuProjectId:project,flowVersion:2,stage:3,task:'answers',goal:'Answer our reviewed support-policy questions.',sourceMode:'guided',modelId:'qwen3-0-6b',gpuJob:jobRef}),project);
 const gets=[],lists=[],artifacts=[],saved=[],posts=[];
 const client={capabilities:async()=>({available:false,generation:true,training:true,models:[],reason:'New GPU jobs are paused. Your saved jobs and downloads remain available.'}),get:async id=>{gets.push(id);return getJob;},list:async query=>{lists.push(query);if(listError)throw Error(listError);return {items:typeof listJobs==='function'?await listJobs():listJobs};},artifact:async(jobId,artifactId)=>{artifacts.push({jobId,artifactId});return new Blob(['verified archived bytes']);},create:async body=>{posts.push(body);throw Error('A history action must never submit.');}};
 const root=new Root(),dispose=mountTeaching(root,{catalog:{models:[]},hardware:{items:[]},loadWorkspace:async()=>{},getAccount:()=>({user:identity}),client,saveProject:async draft=>{saved.push(structuredClone(draft));return draft;}});
 t.after(dispose);await tick();await tick();return {root,gets,lists,artifacts,saved,posts,downloaded,blobs,setUser:user=>{identity=user;}};
}

test('archived current results can be reopened after reviewing the dataset and downloaded while admission is closed',async t=>{
 const {root,gets,artifacts,downloaded,blobs,posts}=await setup(t);
 assert.deepEqual(gets,[current]);assert.match(root.innerHTML,/View saved job results/);
 await root.click('view-job-results');assert.match(root.innerHTML,/Real adapter response/);
 await root.click('download-artifact');assert.deepEqual(artifacts,[{jobId:current,artifactId:'adapter-tar-gz'}]);
 assert.deepEqual(downloaded,['adapter.tar.gz']);assert.equal(await blobs[0].text(),'verified archived bytes');assert.equal(posts.length,0);
 await root.click('improve');assert.match(root.innerHTML,/View saved job results/);
});

test('archived project history shows and downloads earlier training results without replacing the current job',async t=>{
 const {root,lists,artifacts,saved,posts}=await setup(t);
 await root.click('load-history');assert.deepEqual(lists,[{projectId:project}]);
 assert.match(root.innerHTML,/Actual held-out outputs/);assert.match(root.innerHTML,new RegExp(older));
 await root.click('download-artifact',{job:older});assert.deepEqual(artifacts,[{jobId:older,artifactId:'adapter-tar-gz'}]);
 await root.click('save');assert.equal(saved.at(-1).gpuJob.jobId,current);assert.equal(saved.at(-1).stage,3);assert.equal(posts.length,0);
});

test('history excludes jobs from another project and never offers artifacts from a failed restored run',async t=>{
 const {root}=await setup(t,{listJobs:[{...completed(older),project_id:'20000000-0000-4000-8000-000000000001'},{...completed(),status:'failed',error:{message:'Archived before completion.'}}]});
 await root.click('load-history');assert.doesNotMatch(root.innerHTML,new RegExp(older));
 assert.match(root.innerHTML,/Archived before completion/);assert.doesNotMatch(root.innerHTML,/data-artifact=/);
});

test('signed-out users cannot read a restored job or open authenticated history',async t=>{
 const {root,gets,lists}=await setup(t,{user:null});assert.deepEqual(gets,[]);assert.deepEqual(lists,[]);
 assert.match(root.innerHTML,/Sign in with Google to submit or open GPU jobs/);assert.doesNotMatch(root.innerHTML,/data-lesson="load-history"/);
});

test('history errors are recoverable and never submit a replacement job',async t=>{
 const {root,posts}=await setup(t,{listError:'Archive connection is temporarily unavailable.'});await root.click('load-history');
 assert.match(root.innerHTML,/Archive connection is temporarily unavailable/);assert.match(root.innerHTML,/data-lesson="load-history"/);assert.equal(posts.length,0);
});

test('history response is discarded when its account changes before completion',async t=>{
 let finish;const pending=new Promise(resolve=>{finish=resolve;});const {root,setUser}=await setup(t,{listJobs:()=>pending});
 const loading=root.click('load-history');await tick();setUser({uid:'bob'});finish([completed(older)]);await loading;
 assert.doesNotMatch(root.innerHTML,new RegExp(older));assert.doesNotMatch(root.innerHTML,/Real adapter response/);
});


test('expanded saved runs and questions survive a status repaint and download',async t=>{
 const {root}=await setup(t);await root.click('load-history');root.openHistory(older);root.openHistory(older+'-0');
 await root.click('save');assert.deepEqual([...root.opened],[older,older+'-0']);
 await root.click('download-artifact',{job:older});assert.deepEqual([...root.opened],[older,older+'-0']);
});
