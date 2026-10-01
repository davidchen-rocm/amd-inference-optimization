import test from 'node:test';
import assert from 'node:assert/strict';
import {createWorkspaceService} from '../src/workspace.js';
import {storageKey} from '../src/personal-domain.js';
import {TEACH_VERSION} from '../src/teach-domain.js';

const draft={name:'My assistant',work:'coding',instructions:'Give concise answers.',examples:[]};
const selection={fitting:true,model:{id:'model'},artifact:{id:'artifact',revision:'revision'}},config={id:'device'};
function fixture(initial={phase:'ready',user:null,knownUid:null}){
 let state=initial,initializations=0,loads=0;const values=new Map(),cloudRecords=new Map();
 const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 const account={getAccount:()=>({...state}),initializeAccount:async()=>{initializations++;},getCloudDatabase:async()=>{throw Error('unexpected database request');}};
 const projects={reset(){},async load(){loads++;},list:kind=>cloudRecords.get(kind)||[],async save(kind,record){const result={...record,_ownerUid:state.user.uid,_cloudRevision:'saved-revision'};cloudRecords.set(kind,[result,...this.list(kind).filter(item=>item.id!==result.id)]);return result;}};
 const service=createWorkspaceService({account,projects,getStorage:()=>storage,events:new EventTarget()});
 return {service,storage,values,projects,set state(next){state=next;},get state(){return state;},get initializations(){return initializations;},get loads(){return loads;}};
}

test('a guest can create, reopen and edit a local setup without cloud access',async()=>{
 const f=fixture();await f.service.loadWorkspace();
 const first=await f.service.saveLocalSetup(draft,selection,config,8192);
 assert.equal(first._localOnly,true);assert.equal(f.service.localSetups()[0].id,first.id);
 const second=await f.service.saveLocalSetup({...f.service.localSetups()[0],name:'Edited'},selection,config,8192);
 assert.equal(second.id,first.id);assert.equal(second.version,2);assert.equal(f.loads,0);assert.equal(f.initializations,0);
 assert.equal(JSON.parse(f.values.get(storageKey))[0]._localOnly,undefined,'view-only origin markers stay out of persisted data');
});

test('an unavailable account SDK does not block new guest projects',async()=>{
 const f=fixture({phase:'error',user:null,knownUid:null,error:'Sign-in could not load'});
 await f.service.loadWorkspace();await f.service.saveLocalSetup(draft,selection,config,8192);
 await f.service.saveTeachingProject({schema:TEACH_VERSION,id:'teach-1',name:'Lesson',goal:'Explain clearly',examples:[]},'teach-1');
 assert.equal(f.service.localSetups().length,1);assert.equal(f.service.teachingProjects().length,1);assert.equal(f.loads,0);
});

test('a known account outage cannot silently save into the guest library',async()=>{
 for(const user of [null,{uid:'alice'}]){
  const f=fixture({phase:'error',user,knownUid:'alice',error:'offline'});
  await assert.rejects(f.service.loadWorkspace(),/account could not be reached/);
  await assert.rejects(f.service.saveLocalSetup(draft,selection,config,8192),/account could not be reached/);
  await assert.rejects(f.service.saveTeachingProject({id:'lesson'},'lesson'),/account could not be reached/);
  assert.equal(f.values.size,0);assert.throws(()=>f.service.localSetups(),/account could not be reached/);
 }
});

test('cloud read failures are surfaced without writing a local fallback',async()=>{
 const f=fixture({phase:'ready',user:{uid:'alice'},knownUid:'alice'});
 f.projects.load=async()=>{throw Error('Cloud unavailable');};
 await assert.rejects(f.service.saveLocalSetup(draft,selection,config,8192),/Cloud unavailable/);assert.equal(f.values.size,0);
});

test('account changes cannot redirect a pending cloud save',async()=>{
 const f=fixture({phase:'ready',user:{uid:'alice'},knownUid:'alice'});
 f.projects.load=async()=>{f.state={phase:'ready',user:{uid:'bob'},knownUid:'bob'};};
 f.projects.save=()=>assert.fail('must not write a former account draft into a new account');
 await assert.rejects(f.service.saveLocalSetup(draft,selection,config,8192),/account changed/);assert.equal(f.values.size,0);
});

test('local and account records require explicit import or their original owner',async()=>{
 const f=fixture();const local=await f.service.saveLocalSetup(draft,selection,config,8192);const original=f.values.get(storageKey);
 f.state={phase:'ready',user:{uid:'alice'},knownUid:'alice'};
 await assert.rejects(f.service.saveLocalSetup(local,selection,config,8192),/Import local projects/);
 const withoutMarker={...local};delete withoutMarker._localOnly;
 await assert.rejects(f.service.saveLocalSetup(withoutMarker,selection,config,8192),/Import local projects/);
 assert.equal(f.values.get(storageKey),original);
 assert.equal(await f.service.importLocalProjects(),1);assert.equal(f.values.get(storageKey),original);
 assert.equal(f.service.localSetups()[0]._ownerUid,'alice');
 f.state={phase:'ready',user:null,knownUid:null};
 await assert.rejects(f.service.saveLocalSetup({...draft,_ownerUid:'alice',_cloudRevision:'revision'},selection,config,8192),/owner’s account/);
 assert.equal(f.values.get(storageKey),original);
});


test('ordinary signed-in saves and updates stay in the account with revision metadata',async()=>{
 const f=fixture({phase:'ready',user:{uid:'alice'},knownUid:'alice'});
 const first=await f.service.saveLocalSetup(draft,selection,config,8192);
 assert.equal(first._ownerUid,'alice');assert.equal(first._cloudRevision,'saved-revision');assert.equal(first.version,1);
 const second=await f.service.saveLocalSetup({...first,instructions:'An updated instruction.'},selection,config,8192);
 assert.equal(second.id,first.id);assert.equal(second.version,2);assert.equal(second._ownerUid,'alice');assert.equal(second._cloudRevision,'saved-revision');
 assert.equal(f.service.localSetups().length,1);assert.equal(f.values.size,0);assert.equal(f.loads,2);
});
