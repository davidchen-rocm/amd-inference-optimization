import test from 'node:test';
import assert from 'node:assert/strict';
import {createAccountService,accountError} from '../src/account.js';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
function fixture({hint=null,timeoutMs=20}={}){
 const values=new Map(hint?[['macfit.account-owner',hint]]:[]),observers=[],events=new EventTarget(),eventLog=[];
 events.addEventListener('macfit:account-changed',e=>eventLog.push(e.detail));
 const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 const auth={currentUser:null};
 const modules={app:{getApps:()=>[],initializeApp:()=>({})},auth:{getAuth:()=>auth,onAuthStateChanged(_auth,next,error){const observer={next,error,stopped:false};observers.push(observer);return()=>{observer.stopped=true;};}}};
 const service=createAccountService({loadModule:async name=>modules[name],getStorage:()=>storage,events,timeoutMs});
 return {service,observers,values,eventLog,auth,modules};
}

test('simultaneous initialization shares one observer and settles guest identity',async()=>{
 const f=fixture(),first=f.service.initializeAccount(),second=f.service.initializeAccount();assert.equal(first,second);
 await tick();assert.equal(f.observers.length,1);f.observers[0].next(null);await first;
 assert.equal(f.service.getAccount().phase,'ready');assert.equal(f.service.getAccount().knownUid,null);
 await f.service.initializeAccount();assert.equal(f.observers.length,1);
});

test('retry unsubscribes timed-out observers and rejects their late account callbacks',async()=>{
 const f=fixture();await f.service.initializeAccount();
 assert.equal(f.service.getAccount().phase,'error');assert.equal(f.observers[0].stopped,true);
 const retry=f.service.initializeAccount();await tick();assert.equal(f.observers.length,2);
 f.observers[0].next({uid:'old-account'});assert.equal(f.service.getAccount().user,null);assert.equal(f.eventLog.length,0);
 f.auth.currentUser={uid:'alice'};f.observers[1].next(f.auth.currentUser);await retry;
 assert.equal(f.service.getAccount().user.uid,'alice');assert.equal(f.values.get('macfit.account-owner'),'alice');
 assert.equal(f.eventLog[0].initial,true);assert.equal(f.eventLog[0].recovered,true);
 f.observers[0].next(null);assert.equal(f.service.getAccount().user.uid,'alice');assert.equal(f.eventLog.length,1);
});

test('failed SDK load retains the remembered account identity',async()=>{
 const values=new Map([['macfit.account-owner','alice']]);
 const service=createAccountService({loadModule:async()=>{throw Error('Network unavailable');},getStorage:()=>({getItem:key=>values.get(key)}),events:new EventTarget(),timeoutMs:10});
 await service.initializeAccount();assert.equal(service.getAccount().phase,'error');assert.equal(service.getAccount().knownUid,'alice');assert.equal(service.getAccount().user,null);
});

test('an observer failure keeps the known user and ignores further stale state',async()=>{
 const f=fixture();const start=f.service.initializeAccount();await tick();f.observers[0].next({uid:'alice'});await start;
 f.observers[0].error({code:'auth/network-request-failed'});
 assert.equal(f.service.getAccount().phase,'error');assert.equal(f.service.getAccount().knownUid,'alice');assert.equal(f.observers[0].stopped,true);
 f.observers[0].next(null);assert.equal(f.service.getAccount().user.uid,'alice');
});

test('account configuration errors use visitor-facing instructions',()=>{
 assert.doesNotMatch(accountError({code:'auth/unauthorized-domain'}),/Firebase|Authorized domains|Settings/);
 assert.doesNotMatch(accountError({code:'permission-denied'}),/Firestore|rules/);
 assert.match(accountError({code:'auth/popup-blocked'}),/Allow pop-ups/);
 for(const message of ['Failed to fetch dynamically imported module: https://www.gstatic.com/firebasejs/12.19.0/firebase-app.js','error loading dynamically imported module: https://www.gstatic.com/module.js','Importing a module script failed.']){
  const display=accountError(new TypeError(message));assert.match(display,/Sign-in could not be loaded/);assert.doesNotMatch(display,/gstatic|firebase|module\.js/);
 }
});

test('GPU access tokens require a current account and are never written to browser storage',async()=>{
 const f=fixture();const start=f.service.initializeAccount();await tick();const requested=[];
 f.auth.currentUser={uid:'alice',getIdToken:async force=>{requested.push(force);return 'private-access-token';}};f.observers[0].next(f.auth.currentUser);await start;
 assert.equal(await f.service.getAccessToken(true),'private-access-token');assert.deepEqual(requested,[true]);
 assert.equal([...f.values.values()].includes('private-access-token'),false);
 f.auth.currentUser=null;f.observers[0].next(null);await assert.rejects(f.service.getAccessToken(),/Sign in with Google/);
});

test('a token resolving after account change is discarded',async()=>{
 const f=fixture();const start=f.service.initializeAccount();await tick();let resolve;
 f.auth.currentUser={uid:'alice',getIdToken:()=>new Promise(done=>{resolve=done;})};f.observers[0].next(f.auth.currentUser);await start;
 const pending=f.service.getAccessToken();await tick();f.auth.currentUser={uid:'bob'};f.observers[0].next(f.auth.currentUser);resolve('old-account-token');
 await assert.rejects(pending,/account changed/);
});
