import {emptyDraft,taskTypes,goalError,datasetIssues,trainingBrief,datasetJSONL,validJobReference} from './teach-domain.js';

export const FLOW_VERSION=2;
export const sourceModes={guided:{title:'Help me create examples',description:'Start with your goal. No dataset needed.'},reference:{title:'I have reference material',description:'Use your notes, policies, or writing samples.'},import:{title:'I have a dataset',description:'Bring your own question-and-answer pairs.'}};
export const exampleKinds={everyday:'Everyday request',variation:'Another situation',boundary:'Missing information'};
export const projectName=d=>d.name||({answers:'My support assistant',writing:'My writing assistant',format:'My meeting notes assistant',classify:'My message classifier',custom:'My personal assistant'}[d.task])||'My teaching project';
const clean=v=>String(v??'').trim();
const key=v=>clean(v).toLowerCase().replace(/\s+/g,' ');
const testPlaceholders=new Set(['Write a new test question.','Write a second new question.','Write a difficult or incomplete request.','Describe a correct answer.','Describe how to handle it.'].map(key));
const withIds=(rows,prefix)=>rows.map((e,i)=>({...e,id:e.id||prefix+'-'+i,kind:e.kind||['everyday','variation','boundary'][i%3]}));

export function upgradeDraft(raw={}){
  if(raw.gpuJob!=null&&!validJobReference(raw.gpuJob))throw Error('This project has an invalid GPU job reference. The saved project has not been changed.');
  const d={...emptyDraft(),sourceMode:'guided',referenceName:'',seeds:[],evaluationSamples:[],datasetBuilt:false,datasetConfirmed:false,datasetStale:false,previewConfirmed:false,generationTarget:12,...structuredClone(raw)};
  d.examples=withIds(Array.isArray(d.examples)?d.examples:[],'data');
  if(raw.flowVersion!==FLOW_VERSION){
    d.seeds=withIds(d.examples.slice(0,3).map(e=>({...e})),'seed');
    d.evaluationSamples=raw.test?.question?[{...raw.test,id:'test-legacy',approved:false,origin:'migrated'}]:[];
    d.sourceMode=d.examples.some(e=>e.origin==='imported')?'import':d.source?'reference':'guided';
    d.datasetBuilt=d.examples.length>3;d.datasetConfirmed=false;
    d.previewConfirmed=d.seeds.length>=3&&d.seeds.every(e=>e.approved);
    d.stage=raw.stage>=3?3:raw.stage===2?2:0;d.status='draft';
  }
  d.flowVersion=FLOW_VERSION;
  d.seeds=withIds(Array.isArray(d.seeds)?d.seeds:[],'seed');
  d.evaluationSamples=(Array.isArray(d.evaluationSamples)?d.evaluationSamples:[]).map((e,i)=>({...e,id:e.id||'test-'+i}));
  d.stage=Number.isInteger(d.stage)?Math.max(0,Math.min(5,d.stage)):0;
  if(!taskTypes[d.task]||!clean(d.goal))d.stage=0;
  if(d.status==='demo-running'||d.status==='demo-complete'){d.status='draft';d.stage=4;}
  if(d.gpuJob?.kind==='training'&&!['succeeded','failed','cancelled'].includes(d.gpuJob.status))d.stage=5;
  return d;
}
export function invalidateDataset(d,{seedChange=false}={}){
  d.datasetConfirmed=false;d.status='draft';
  if(seedChange){d.previewConfirmed=false;if(d.datasetBuilt)d.datasetStale=true;for(const e of d.evaluationSamples||[])e.approved=false;}
}
export function changeBrief(d,field,value){
  if(d[field]===value)return;
  d[field]=value;
  if(field==='task'&&d.sourceMode!=='import')d.seeds=[];
  for(const e of d.seeds)e.approved=false;
  for(const e of d.examples)e.approved=false;
  for(const e of d.evaluationSamples)e.approved=false;
  invalidateDataset(d,{seedChange:true});
}
export function makePreview(d){
  if(d.sourceMode==='import'&&d.examples.length){d.seeds=d.examples.slice(0,3).map(e=>({...e}));}
  else if(!d.seeds.length)d.seeds=withIds(Array.from({length:3},()=>({question:'',answer:'',approved:false,origin:'user-authored'})),'seed');
  d.previewConfirmed=false;return d.seeds;
}
export function sourceError(d){
  if(!Object.hasOwn(sourceModes,d.sourceMode))return 'Choose how you want to prepare your data.';
  if(d.sourceMode==='reference'&&clean(d.source).length<20)return 'Add a few sentences of reference material, or choose Help me create examples.';
  if(d.sourceMode==='import'&&d.examples.length<3)return 'Import at least three examples for this preview. You can also start with guided examples.';
  return '';
}
export function seedError(d){
  if(d.seeds.length<3)return 'Review three examples to establish the direction.';
  const issues=datasetIssues(d.seeds);if(issues.length)return issues[0];
  if(d.seeds.some(e=>!e.approved))return 'Approve each preview example. Your edits need a fresh approval.';
  return '';
}
export function approveSeed(d,id){
  const e=d.seeds.find(e=>e.id===id);if(!e)throw Error('This example is no longer available.');
  const problem=datasetIssues([e])[0];if(problem)throw Error(problem);
  e.approved=true;
  if(d.sourceMode==='import'){const row=d.examples.find(r=>r.id===e.id);if(row)Object.assign(row,structuredClone(e));}
  d.previewConfirmed=!seedError(d);d.datasetConfirmed=false;
}
export function editSeed(d,id,field,value){
  const e=d.seeds.find(e=>e.id===id);if(!e)return;
  e[field]=value;e.approved=false;e.needsReplacement=false;
  if(d.sourceMode==='import'){const row=d.examples.find(r=>r.id===e.id);if(row)Object.assign(row,structuredClone(e));}
  invalidateDataset(d,{seedChange:true});
}
export function prepareImportedDataset(d){
  const error=seedError(d);if(error)throw Error(error);
  if(d.sourceMode!=='import')throw Error('Import a dataset or use GPU generation to prepare more examples.');
  d.examples=d.examples.map(e=>{const seed=d.seeds.find(s=>s.id===e.id);return seed?{...e,...structuredClone(seed)}:e;});
  d.previewConfirmed=true;d.datasetBuilt=true;d.datasetConfirmed=false;d.datasetStale=false;
  d.generation={mode:'imported',count:d.examples.length};
  return d.examples;
}
export function generationInput(d,modelId,purpose){
  const error=goalError(d)||sourceError(d)||(purpose==='dataset'?seedError(d):'');if(error)throw Error(error);
  if(!modelId||!['preview','dataset'].includes(purpose))throw Error('Choose an available generation model.');
  return {model_id:modelId,task:d.task,goal:d.goal,language:d.language,source:d.source,purpose,seeds:purpose==='dataset'?d.seeds.map(e=>({id:e.id,question:e.question,answer:e.answer,...(e.system?{system:e.system}:{}),approved:true})):[],target_count:purpose==='preview'?3:[12,24,48].includes(Number(d.generationTarget))?Number(d.generationTarget):12};
}
export function trainingInput(d,modelId,preset='quick'){
  const error=goalError(d)||sourceError(d)||datasetReadyError(d);if(error)throw Error(error);
  if(!modelId||!['quick','standard'].includes(preset))throw Error('Choose an available model and training preset.');
  return {model_id:modelId,task:d.task,goal:d.goal,language:d.language,source:d.source,preset,training:d.examples.map(e=>({id:e.id,messages:[{role:'system',content:e.system||d.goal},{role:'user',content:e.question},{role:'assistant',content:e.answer}],approved:true})),evaluation:d.evaluationSamples.map(e=>({id:e.id,question:e.question,expected:e.answer,...(e.system?{system:e.system}:{}),approved:true}))};
}
export function applyGenerationResult(d,job){
  const reference=d.gpuJob;
  if(job.status!=='succeeded'||job.kind!=='generation'||job.id!==reference?.jobId)throw Error('Generation has not completed for this project.');
  if(reference.applied)return false;
  const rows=job.result?.examples;
  if(!Array.isArray(rows)||rows.length!==(reference.purpose==='preview'?3:Number(d.generationTarget))||rows.some(e=>!e||typeof e.id!=='string'||!e.id||e.id.length>128||typeof e.question!=='string'||e.question.length>12000||typeof e.answer!=='string'||e.answer.length>24000||(e.system!==undefined&&(typeof e.system!=='string'||e.system.length>6000)))||new Set(rows.map(e=>e.id)).size!==rows.length||datasetIssues(rows).length)throw Error('Generated examples failed validation. Your existing dataset has been kept.');
  const fresh=rows.map(e=>({id:e.id,question:e.question,answer:e.answer,...(e.system?{system:e.system}:{}),approved:false,origin:'gpu-generated',kind:e.kind||'everyday'}));
  if(reference.purpose==='preview'){
    d.seeds=fresh;d.previewConfirmed=false;d.datasetConfirmed=false;if(d.datasetBuilt)d.datasetStale=true;d.stage=2;
  }else{
    if(d.seeds.some(seed=>!fresh.some(e=>e.question===seed.question&&e.answer===seed.answer&&(e.system||'')===(seed.system||''))))throw Error('The generated dataset did not preserve your corrected examples. Your existing dataset has been kept.');
    const additions=fresh.filter(e=>!d.seeds.some(seed=>e.question===seed.question&&e.answer===seed.answer&&(e.system||'')===(seed.system||'')));
    if(additions.length+d.seeds.length!==fresh.length)throw Error('The generated dataset contains duplicate examples. Your existing dataset has been kept.');
    d.examples=[...structuredClone(d.seeds),...additions];d.datasetBuilt=true;d.datasetStale=false;d.datasetConfirmed=false;d.stage=3;
  }
  d.generation={mode:'gpu',jobId:job.id,baseModel:job.base_model,count:rows.length,provenance:job.result?.provenance||null};
  reference.applied=true;d.status='draft';return true;
}
export function editDatasetRow(d,id,patch){
  const row=d.examples.find(e=>e.id===id);if(!row)throw Error('This row no longer exists.');
  Object.assign(row,patch,{approved:false,needsReplacement:false});
  if(['demo-expanded','gpu-generated'].includes(row.origin))row.origin='user-edited';
  // Keep shared seed identities aligned. The existing expansion is now stale.
  const seed=d.seeds.find(e=>e.id===id);
  if(seed){Object.assign(seed,patch,{approved:false,needsReplacement:false});invalidateDataset(d,{seedChange:true});}
  else invalidateDataset(d);
}
// Recognize wording wrappers in older saved datasets when checking train/test overlap.
const prefixes=['Please respond to this request:','Here is the request:','Could you help with this?','Please handle the following:','A user asks:','Here is what I need:','Can you answer this?','Please read this request:','Please respond clearly:','Here is my question:','Please help me with this request:','How would you respond?','Respond to the following input:','Please process this input:','A new request:'];
export function checkDataset(d){
  const issues=datasetIssues(d.examples),rowProblems=new Map();
  for(const issue of issues){const numbers=issue.match(/\d+/g)||[];for(const n of numbers){const row=d.examples[Number(n)-1];if(row)rowProblems.set(row.id,[...(rowProblems.get(row.id)||[]),issue]);}}
  const tests=d.evaluationSamples||[],testIssues=[];
  if(tests.length<3)testIssues.push('Keep at least three new questions for the held-out evaluation.');
  const testKeys=new Set();
  const canonical=v=>{
    let value=key(v);for(const prefix of prefixes){if(value.startsWith(key(prefix)))value=key(value.slice(prefix.length));}return value;
  };
  const trainingKeys=new Set(d.examples.map(e=>canonical(e.question)));
  for(let i=0;i<tests.length;i++){
    const e=tests[i],k=canonical(e.question);
    if(!clean(e.question)||!clean(e.answer))testIssues.push('Test '+(i+1)+' needs a question and expected answer.');
    if(e.needsReplacement||testPlaceholders.has(key(e.question))||testPlaceholders.has(key(e.answer)))testIssues.push('Replace the placeholder in test '+(i+1)+'.');
    if(trainingKeys.has(k))testIssues.push('Test '+(i+1)+' repeats a training example. Choose a new situation.');
    if(testKeys.has(k))testIssues.push('Test '+(i+1)+' repeats another test question.');
    testKeys.add(k);
  }
  const pending=d.examples.filter(e=>!e.approved).length;
  const testPending=tests.filter(e=>!e.approved).length;
  return {count:d.examples.length,approved:d.examples.filter(e=>e.approved&&!rowProblems.has(e.id)).length,pending,issues,rowProblems,tests:tests.length,testIssues,testPending,
    ready:!!d.datasetBuilt&&!d.datasetStale&&d.examples.length>=3&&!issues.length&&!pending&&!testIssues.length&&!testPending,
    kinds:Object.keys(exampleKinds).filter(k=>d.examples.some(e=>e.kind===k))};
}
export function datasetReadyError(d){
  if(!d.datasetBuilt)return 'Build or import the full dataset before continuing. Preview approval alone is not a dataset.';
  if(d.datasetStale||seedError(d))return 'Your teaching direction changed. Review the preview examples and rebuild the dataset.';
  const report=checkDataset(d);
  if(report.issues.length)return report.issues[0];
  if(report.pending)return 'Review the remaining '+report.pending+' dataset examples.';
  if(report.testIssues.length)return report.testIssues[0];
  if(report.testPending)return 'Review and approve your separate test questions.';
  if(!d.datasetConfirmed)return 'Confirm that you reviewed the complete dataset and test questions.';
  return '';
}
const testFixtures={
  answers:[['It has been one business day. Should my order already have shipped?','Orders ship within 2 business days.'],['How can I reach your support team?','Contact help@example.com for support.'],['Can I return an item that I have already used?','The stated return policy covers unused items. Contact support for help with this situation.']],
  writing:[['Ask Taylor to confirm the meeting time.','Hi Taylor, could you confirm the meeting time? Thanks!'],['Tell Morgan the revised file is attached.','Hi Morgan, the revised file is attached. Let me know if you need anything else.'],['Ask Lee whether Thursday or Friday works better for a call.','Hi Lee, would Thursday or Friday work better for a quick call?']],
  format:[['Kim will send the budget on Monday.','Task: Send the budget\nOwner: Kim\nDue date: Monday'],['We need a list of suppliers by Tuesday, but nobody volunteered.','Task: Prepare a list of suppliers\nOwner: Not specified\nDue date: Tuesday'],['Jordan will check the final numbers. No date was set.','Task: Check the final numbers\nOwner: Jordan\nDue date: Not specified']],
  classify:[['Please send me a copy of my invoice.','Billing'],['My tracking link has stopped updating.','Delivery'],['Is this product available in blue?','Other']]
};
export function suggestedTests(d){
  const rows=testFixtures[d.task]||[['Write a new test question.','Describe a correct answer.'],['Write a second new question.','Describe a correct answer.'],['Write a difficult or incomplete request.','Describe how to handle it.']];
  return rows.map(([question,answer],i)=>({id:'check-'+i,question,answer,approved:false,origin:'illustrative-test',needsReplacement:d.task==='custom'}));
}
export function lessonBrief(d,model){
  const error=goalError(d)||sourceError(d)||datasetReadyError(d)||(!model?'Choose a model before training.':'');if(error)throw Error(error);
  const brief=trainingBrief({...d,test:d.evaluationSamples[0]},model);
  return {...brief,name:projectName(d),flowVersion:FLOW_VERSION,sourceMode:d.sourceMode,dataset:{...brief.dataset,reviewed:true,generation:d.generation||{mode:'imported'},previewCount:d.seeds.length},evaluation:{heldOut:true,examples:d.evaluationSamples.map(e=>({question:e.question,expected:e.answer,approved:e.approved})),count:d.evaluationSamples.length}};
}
export const trainingFile=d=>datasetJSONL(d);
export const evaluationFile=d=>d.evaluationSamples.map(e=>JSON.stringify({question:e.question,expected:e.answer})).join('\n')+'\n';
