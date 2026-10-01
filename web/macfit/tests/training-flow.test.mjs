import test from 'node:test';
import assert from 'node:assert/strict';
import {upgradeDraft,makePreview,generationInput,trainingInput,applyGenerationResult,prepareImportedDataset} from '../src/dataset-flow.js';
import {emptyDraft,saveDraft,readDrafts,validJobReference} from '../src/teach-domain.js';

const request='10000000-0000-4000-8000-000000000001',project='10000000-0000-4000-8000-000000000002',jobId='10000000-0000-4000-8000-000000000003';
const rows=count=>Array.from({length:count},(_,i)=>({id:'row-'+i,question:'Please classify ticket '+i,answer:'Category '+i,approved:true,origin:'user-authored'}));
function draft(){const d=upgradeDraft({...emptyDraft(),flowVersion:2,task:'classify',goal:'Classify each request using the reviewed ticket categories.',sourceMode:'guided',language:'en'});d.seeds=rows(3);d.examples=rows(12);d.evaluationSamples=Array.from({length:3},(_,i)=>({id:'test-'+i,question:'A new evaluation ticket '+i,answer:'Expected category '+i,approved:true}));d.datasetBuilt=true;d.datasetConfirmed=true;d.previewConfirmed=true;return d;}
const reference=(kind='generation',purpose='dataset')=>({requestId:request,projectId:project,jobId,kind,purpose,preset:'quick',modelId:'qwen3-0-6b',ownerUid:'alice',status:'succeeded',applied:false});
const generationJob=examples=>({id:jobId,kind:'generation',status:'succeeded',base_model:{repo_id:'Qwen/Qwen3-0.6B',revision:'a'.repeat(40)},result:{examples,provenance:{method:'lm-inference'}}});

test('unsigned manual preview has empty editable rows, never fabricated AI answers',()=>{
 const d=upgradeDraft({...emptyDraft(),flowVersion:2,task:'custom',goal:'Answer my questions in a consistent format.'});makePreview(d);
 assert.equal(d.seeds.length,3);assert.ok(d.seeds.every(e=>e.question===''&&e.answer===''&&!e.approved&&e.origin==='user-authored'));
});

test('generation requests include the actual corrected seeds but never held-out answers',()=>{
 const d=draft();const preview=generationInput(d,'qwen3-0-6b','preview');assert.equal(preview.target_count,3);assert.deepEqual(preview.seeds,[]);
 const batch=generationInput(d,'qwen3-0-6b','dataset');assert.equal(batch.target_count,12);assert.equal(batch.seeds[0].answer,d.seeds[0].answer);
 assert.equal(JSON.stringify(batch).includes(d.evaluationSamples[0].answer),false);
});

test('real generated dataset preserves corrected seed approvals and requires review of all additions',()=>{
 const d=draft();d.gpuJob=reference();const returned=rows(12).map(e=>({...e,approved:true,origin:'gpu-generated'}));
 assert.equal(applyGenerationResult(d,generationJob(returned)),true);assert.equal(d.examples.length,12);
 assert.ok(d.examples.slice(0,3).every(e=>e.approved));assert.ok(d.examples.slice(3).every(e=>!e.approved&&e.origin==='gpu-generated'));
 assert.equal(d.datasetConfirmed,false);assert.equal(d.generation.mode,'gpu');assert.equal(d.gpuJob.applied,true);
 d.examples[4].answer='My reviewed correction';assert.equal(applyGenerationResult(d,generationJob(returned)),false);assert.equal(d.examples[4].answer,'My reviewed correction');
});

test('bad or incomplete generated output never replaces the existing dataset',()=>{
 for(const returned of [rows(2),[...rows(11),rows(1)[0]],rows(12).map(e=>e.id==='row-0'?{...e,answer:'Changed corrected seed'}:e)]){
  const d=draft();d.gpuJob=reference();const before=JSON.stringify(d.examples);
  assert.throws(()=>applyGenerationResult(d,generationJob(returned)),/validation|corrected examples|duplicate/);assert.equal(JSON.stringify(d.examples),before);assert.equal(d.gpuJob.applied,false);
 }
});

test('generated preview is unapproved and cannot mark old reviewed data current',()=>{
 const d=draft();d.gpuJob=reference('generation','preview');applyGenerationResult(d,generationJob(rows(3)));
 assert.equal(d.stage,2);assert.equal(d.datasetStale,true);assert.equal(d.datasetConfirmed,false);assert.ok(d.seeds.every(e=>!e.approved));
});

test('training sends reviewed chat messages and disjoint evaluation rows as separate fields',()=>{
 const d=draft();const input=trainingInput(d,'qwen3-0-6b','standard');
 assert.equal(input.preset,'standard');assert.equal(input.training.length,12);assert.equal(input.evaluation.length,3);
 assert.deepEqual(input.training[0].messages.map(m=>m.role),['system','user','assistant']);assert.equal(input.training[0].messages[2].content,d.examples[0].answer);
 assert.equal(JSON.stringify(input.training).includes(d.evaluationSamples[0].answer),false);assert.equal(input.evaluation[0].expected,d.evaluationSamples[0].answer);
});

test('unreviewed or overlapping data cannot be submitted for training',()=>{
 const d=draft();d.examples[3].approved=false;assert.throws(()=>trainingInput(d,'qwen3-0-6b'),/Review the remaining/);
 d.examples[3].approved=true;d.evaluationSamples[0].question=d.examples[0].question;assert.throws(()=>trainingInput(d,'qwen3-0-6b'),/repeats a training/);
 d.evaluationSamples[0].question='A separate question';d.datasetConfirmed=false;assert.throws(()=>trainingInput(d,'qwen3-0-6b'),/Confirm/);
});

test('import preparation preserves every imported answer without invoking generation',()=>{
 const d=draft();d.sourceMode='import';const before=d.examples.map(e=>e.answer);prepareImportedDataset(d);
 assert.deepEqual(d.examples.map(e=>e.answer),before);assert.equal(d.generation.mode,'imported');assert.equal(d.datasetConfirmed,false);
});

test('refresh restores an active GPU reference without resetting it to a draft or inventing completion',()=>{
 const d=draft();d.gpuJob={...reference('training'),status:'running'};d.stage=4;
 const restored=upgradeDraft(JSON.parse(JSON.stringify(d)));assert.equal(restored.gpuJob.jobId,jobId);assert.equal(restored.gpuJob.status,'running');assert.equal(restored.stage,5);
 const terminal=upgradeDraft({...d,stage:3,gpuJob:{...d.gpuJob,status:'succeeded'}});assert.equal(terminal.stage,3,'reviewing data after completion survives a refresh');
});

test('old demo completion is never promoted into a successful GPU job',()=>{
 const d=upgradeDraft({...draft(),status:'demo-complete',stage:5});assert.equal(d.status,'draft');assert.equal(d.stage,4);assert.equal(d.gpuJob,undefined);
});

test('invalid or cross-shaped job references block destructive project saves',()=>{
 const valid={...draft(),id:project,name:'Project',gpuJob:reference('training')};assert.equal(validJobReference(valid.gpuJob),true);
 let raw=JSON.stringify([valid]);const storage={getItem:()=>raw,setItem:(_key,value)=>{raw=value;}};
 const before=raw;assert.throws(()=>saveDraft(storage,{...valid,gpuJob:{...valid.gpuJob,jobId:'../../elsewhere'}},project),/invalid data/);assert.equal(raw,before);
  raw=JSON.stringify([{...valid,gpuJob:{...valid.gpuJob,ownerUid:null}}]);assert.throws(()=>readDrafts(storage),/unreadable data/);
  assert.throws(()=>upgradeDraft({...valid,gpuJob:{...valid.gpuJob,jobId:'invalid'}}),/invalid GPU job reference/);
});
