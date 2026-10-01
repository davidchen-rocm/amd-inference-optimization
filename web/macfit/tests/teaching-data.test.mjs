import test from 'node:test';
import assert from 'node:assert/strict';
import {TEACH_VERSION,emptyDraft,readDrafts,saveDraft} from '../src/teach-domain.js';
import {upgradeDraft} from '../src/dataset-flow.js';
import {teachingLibraryMarkup} from '../src/lesson-view.js';

function storage(initial=null){
  let raw=initial,writes=0;
  return {getItem:()=>raw,setItem:(_key,value)=>{raw=value;writes++;},get raw(){return raw;},get writes(){return writes;}};
}
const example={question:'When is support available?',answer:'Support is available every weekday.',approved:false};
const saved=()=>({...emptyDraft(),id:'lesson-1',name:'Support project',task:'answers',goal:'Answer support questions clearly.',examples:[{...example}]});

test('legacy teaching drafts can still be opened, migrated and saved without losing their answers',()=>{
  const legacy={schema:TEACH_VERSION,id:'legacy',name:'Legacy project',examples:[example],test:{question:'How do I contact support?',answer:'Use the contact form.'}};
  const original=JSON.stringify([legacy]),store=storage(original);
  const upgraded=upgradeDraft(readDrafts(store)[0]);
  assert.equal(upgraded.seeds[0].answer,example.answer);
  assert.equal(upgraded.evaluationSamples[0].answer,legacy.test.answer);
  assert.equal(store.raw,original,'reading and migration must not write');
  assert.equal(store.writes,0);
  const result=saveDraft(store,{...upgraded,name:'Updated project'},'unused');
  assert.equal(result.id,'legacy');assert.equal(readDrafts(store).length,1);
  assert.equal(readDrafts(store)[0].examples[0].answer,example.answer);
});

test('malformed nested teaching data blocks destructive saves and preserves every stored byte',()=>{
  const malformed=[
    null,{...saved(),schema:'unknown-version'},{...saved(),id:''},
    {...saved(),examples:[null]},{...saved(),examples:{}},
    {...saved(),examples:[{...example,answer:17}]},
    {...saved(),seeds:[null]},{...saved(),evaluationSamples:[null]},
    {...saved(),test:null},{...saved(),goal:{}},{...saved(),source:[]},
    {...saved(),stage:'3'},{...saved(),datasetConfirmed:'true'}
  ];
  for(const row of malformed){
    const damaged=row?.id==='lesson-1'?{...row,id:'damaged-project'}:row;
    const raw=JSON.stringify([saved(),damaged]),store=storage(raw);
    assert.throws(()=>readDrafts(store),/original data has been kept/);
    assert.throws(()=>saveDraft(store,{...saved(),name:'Changed'},'lesson-1'),/saving is blocked/);
    assert.equal(store.raw,raw);assert.equal(store.writes,0);
  }
  for(const raw of ['', '[{"unfinished"', '{}', JSON.stringify([saved(),saved()])]){
    const store=storage(raw);
    assert.throws(()=>saveDraft(store,saved(),'lesson-1'),/original data has been kept/);
    assert.equal(store.raw,raw);assert.equal(store.writes,0);
  }
});

test('an invalid new teaching project cannot damage a healthy saved library',()=>{
  const raw=JSON.stringify([saved()]),store=storage(raw);
  assert.throws(()=>saveDraft(store,{...saved(),examples:[null]},'lesson-1'),/could not be saved/);
  assert.throws(()=>saveDraft(store,{...saved(),examples:new Array(1)},'lesson-1'),/could not be saved/);
  assert.equal(store.raw,raw);assert.equal(store.writes,0);
});

test('a malformed teaching example renders the existing library error instead of crashing My models',t=>{
  const original=Object.getOwnPropertyDescriptor(globalThis,'localStorage');
  const raw=JSON.stringify([{...saved(),examples:[null]}]),store=storage(raw);
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:store});
  t.after(()=>{if(original)Object.defineProperty(globalThis,'localStorage',original);else delete globalThis.localStorage;});
  let markup;
  assert.doesNotThrow(()=>{markup=teachingLibraryMarkup({models:[]});});
  assert.match(markup,/Your teaching projects could not be read/);
  assert.equal(store.raw,raw);assert.equal(store.writes,0);
});
