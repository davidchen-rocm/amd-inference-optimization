import test from 'node:test';
import assert from 'node:assert/strict';
import {saveModel,loadModels,readModelStore,storageKey,examplesFromText} from '../src/personal-domain.js';

const draft={name:'My assistant',work:'coding',instructions:'Give concise answers.',examples:[{question:'Hello?',answer:'Hello.'}]};
const selection={fitting:true,model:{id:'model'},artifact:{id:'artifact',revision:'revision'}};
const config={id:'device'};
function storage(initial=null){let value=initial,writes=0;return {getItem:()=>value,setItem:(_key,next)=>{value=next;writes++;},get raw(){return value;},get writes(){return writes;}};}
function saved(){const store=storage();return saveModel(draft,selection,config,8192,{storage:store,id:'local-model',now:'2026-09-30T12:00:00Z'});}

test('local setups survive a save and an edit without changing identity',()=>{
 const store=storage();const first=saveModel(draft,selection,config,8192,{storage:store,id:'first'});
 assert.equal(loadModels(store)[0].id,'first');assert.equal(first.version,1);
 const second=saveModel({...first,instructions:'Updated instructions.'},selection,config,8192,{storage:store,id:'unused'});
 assert.equal(second.id,'first');assert.equal(second.version,2);assert.equal(loadModels(store).length,1);
});

test('one malformed setup cannot hide healthy setups or be deleted by the next save',()=>{
 const good=saved();const raw=JSON.stringify([null,good,{...good,id:'bad-example',examples:[null]},{...good,id:'bad-version',version:'<img src=x>'},good]);
 const store=storage(raw),result=readModelStore(store);
 assert.equal(result.items.length,1);assert.equal(result.items[0].id,good.id);assert.equal(result.raw,raw);assert.equal(result.blocked,true);
 assert.match(result.warning,/4 saved setups could not be opened/);
 assert.throws(()=>saveModel({...good,name:'Updated'},selection,config,8192,{storage:store}),/download a backup/);
 assert.equal(store.raw,raw);assert.equal(store.writes,0,'corruption must not be silently repaired by dropping records');
});

test('invalid JSON remains available verbatim for recovery',()=>{
 const raw='[{"name":"unfinished';const store=storage(raw);const result=readModelStore(store);
 assert.deepEqual(result.items,[]);assert.equal(result.raw,raw);assert.match(result.warning,/Download a backup/);assert.equal(store.raw,raw);
 assert.throws(()=>saveModel(draft,selection,config,8192,{storage:store}),/original data has been kept/);assert.equal(store.writes,0);
});

test('denied storage is a recoverable condition rather than an empty writable library',()=>{
 const store={getItem(){throw new DOMException('denied','SecurityError');},setItem(){assert.fail('must never write after a failed read');}};
 assert.equal(readModelStore(store).blocked,true);assert.match(readModelStore(store).warning,/storage is unavailable/);
 assert.throws(()=>saveModel(draft,selection,config,8192,{storage:store}),/saved setups need attention/);
});

test('malformed example message containers show a validation error',()=>{
 assert.throws(()=>examplesFromText('[{"messages":"not an array"}]'),/Example 1 needs a question and answer/);
 assert.throws(()=>examplesFromText('[{"messages":[null]}]'),/Example 1 needs a question and answer/);
});
