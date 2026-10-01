// Reviewed teaching data and portable project records. GPU execution is server-owned.
export const TEACH_VERSION = 'macfit-teaching-v1';
export const TEACH_STORAGE = 'macfit.teaching-drafts.v1';
export const MAX_EXAMPLES = 500;
export const MAX_FILE_BYTES = 2_000_000;
export const taskTypes = {
  answers: {title:'Answer questions', icon:'chat', description:'Teach it answers, boundaries, and when to say “I don’t know”.', goal:'Answer questions about my shop clearly. Use only the information I provide.', source:'Example shop: Orders ship within 2 business days. Unused items can be returned within 30 days. Contact help@example.com for support.', task:'general'},
  writing: {title:'Write in my style', icon:'pen', description:'Teach it your tone, wording, and the way you write.', goal:'Write short, warm customer emails. Use plain language and finish with one clear next step.', source:'Example style: “Hi Alex, your order is on its way. You can track it with the link below. Thanks for your patience!”', task:'writing'},
  format: {title:'Follow my format', icon:'layout', description:'Turn messy notes into the same useful structure, every time.', goal:'Turn meeting notes into an action list with Task, Owner, and Due date. Say “Not specified” for missing details.', source:'Example format: Task: Send the draft | Owner: Alex | Due date: Friday', task:'general'},
  classify: {title:'Organize & label', icon:'tag', description:'Sort messages into categories you choose.', goal:'Classify customer messages as Billing, Delivery, or Other. Reply with the label only.', source:'Billing: payments, invoices, or refunds. Delivery: shipping and tracking. Other: everything else.', task:'general'},
  custom: {title:'Something else', icon:'spark', description:'Describe one specific thing you want to teach.', goal:'', source:'', task:'general'}
};
export function emptyDraft(modelId = '') {
  return {schema:TEACH_VERSION, id:null, name:'', task:'', goal:'', source:'', language:'en', modelId,
    examples:[], test:{question:'',answer:''}, delivery:'local', deviceId:'', deviceConfirmed:false,
    status:'draft', stage:0, updatedAt:null};
}
const text = v => typeof v === 'string' ? v.trim() : '';
const key = v => text(v).toLowerCase().replace(/\s+/g, ' ');
export function chooseTask(draft, task) {
  if (!taskTypes[task]) throw Error('Choose a task.');
  return {...draft, task, examples:draft.task===task?draft.examples:[], test:draft.task===task?draft.test:{question:'',answer:''}, status:'draft'};
}
export function goalError(draft) {
  if (!taskTypes[draft.task]) return 'Choose the kind of work you want to teach.';
  if (text(draft.goal).length < 12) return 'Describe the result you want in a short sentence.';
  if (draft.goal.length > 6000 || draft.source.length > 12000) return 'Shorten the description or reference notes to fit the displayed limits.';
  return '';
}
export function modelEvidence(model, task, language) {
  const e = model?.recommendation_evidence;
  const valid = e && e.hf_id===model.hf_id && e.revision===model.repository_revision;
  return {task:valid?e.tasks?.[taskTypes[task]?.task||'general']:null, language:valid?e.languages?.[language]:null};
}
export function candidateModels(models, task, language, selected='') {
  const candidates=models.filter(m=>m.pipeline_tag==='text-generation' && m.parameter_count>0 && m.parameter_count<=14e9);
  candidates.sort((a,b)=>{
    const ae=modelEvidence(a,task,language),be=modelEvidence(b,task,language);
    return Number(!!be.task&&!!be.language)-Number(!!ae.task&&!!ae.language) || (be.task?.level||0)-(ae.task?.level||0) || Number(b.weights_access==='public-files-listed')-Number(a.weights_access==='public-files-listed') || a.parameter_count-b.parameter_count || a.id.localeCompare(b.id);
  });
  // A small, reviewed shortlist; one checkpoint per size band when available.
  const chosen=[];
  for(const [min,max] of [[0,2.5e9],[2.5e9,5.5e9],[5.5e9,14e9]]){
    const m=candidates.find(m=>m.parameter_count>min&&m.parameter_count<=max);
    if(m)chosen.push(m);
  }
  const manual=models.find(m=>m.id===selected&&m.pipeline_tag==='text-generation');
  if(manual&&!chosen.some(m=>m.id===manual.id))chosen.push(manual);
  return chosen;
}
const fixtures = {
 answers:[['When will my order ship?','Orders ship within 2 business days.'],['Can I return an unused item?','Yes. Unused items can be returned within 30 days.'],['Do you offer a student discount?','I don’t have a confirmed discount policy. Please contact help@example.com.']],
 writing:[['Tell Alex the draft is ready and ask for feedback by Friday.','Hi Alex, the draft is ready for you. Could you send your feedback by Friday? Thanks!'],['Tell Sam we need one more day to finish the report.','Hi Sam, we need one more day to finish the report. Thanks for your patience. We’ll send it tomorrow.'],['Ask Jordan to send the missing attachment.','Hi Jordan, thanks for your message. Could you send the attachment when you have a moment?']],
 format:[['Alex will send the draft on Friday.','Task: Send the draft\nOwner: Alex\nDue date: Friday'],['Review the proposal. No owner or deadline was decided.','Task: Review the proposal\nOwner: Not specified\nDue date: Not specified'],['Sam will update the slides tomorrow.','Task: Update the slides\nOwner: Sam\nDue date: Tomorrow']],
 classify:[['I was charged twice for my order.','Billing'],['Where can I track my package?','Delivery'],['What colors does this come in?','Other']],
 custom:[['Write a typical request your model should handle.','Replace this with the complete answer you want.'],['Write a different request for the same task.','Replace this with another correct answer.'],['Write a request with missing information.','Replace this with how your model should ask for clarification.']]
};
export function starterExamples(draft) {
  const rows=fixtures[draft.task]||fixtures.custom;
  // Demonstration fixtures, deliberately never attributed to an LLM or user facts.
  return rows.map(([question,answer],i)=>({id:'example-'+(i+1),question,answer,approved:false,origin:'illustrative-template',needsReplacement:draft.task==='custom'}));
}
export function parseDataset(contents) {
  if(new TextEncoder().encode(contents).length>MAX_FILE_BYTES)throw Error('Choose a JSON or JSONL file smaller than 2 MB.');
  let raw;
  try{raw=JSON.parse(contents);}catch{
    try{raw=contents.split(/\r?\n/).filter(s=>s.trim()).map((line,i)=>{try{return JSON.parse(line);}catch{throw Error('Invalid JSON on line '+(i+1)+'.');}});}catch(e){throw Error(e.message||'Choose a valid JSON or JSONL dataset.');}
  }
  if(!Array.isArray(raw)) raw=raw?.examples||(raw?.question||raw?.messages?[raw]:null);
  if(!Array.isArray(raw)||raw.length===0||raw.length>MAX_EXAMPLES)throw Error('Import between 1 and 500 examples.');
  return raw.map((row,i)=>{
    let question=row?.question??row?.input,answer=row?.answer??row?.output,system='';
    if(row?.messages){
      if(!Array.isArray(row.messages))throw Error('Example '+(i+1)+': messages must be an array.');
      const messages=row.messages;
      const pairs=messages.filter(m=>m.role!=='system');
      if(pairs.length!==2||pairs[0]?.role!=='user'||pairs[1]?.role!=='assistant'||messages.filter(m=>m.role==='system').length>1||messages.some(m=>typeof m.content!=='string'))throw Error('Example '+(i+1)+': use one user message and one assistant answer, with an optional system message. Multi-turn and tool conversations are not supported in this preview.');
      question=pairs[0].content;answer=pairs[1].content;system=messages.find(m=>m.role==='system')?.content||'';
    }
    if(!text(question)||!text(answer)||question.length>12000||answer.length>24000||system.length>6000)throw Error('Example '+(i+1)+' needs a question and answer within the length limits.');
    return {id:'import-'+(i+1),question:text(question),answer:text(answer),system:text(system),approved:false,origin:'imported'};
  });
}
export function datasetIssues(examples) {
  const seen=new Map(),issues=[];
  for(let i=0;i<examples.length;i++){
    const e=examples[i];
    if(!text(e.question)||!text(e.answer))issues.push('Example '+(i+1)+' is missing an input or answer.');
    if(e.needsReplacement||fixtures.custom.some(([q,a])=>key(e.question)===key(q)||key(e.answer)===key(a)))issues.push('Replace the placeholder in example '+(i+1)+'.');
    const k=key(e.system)+'|'+key(e.question),previous=seen.get(k);
    if(previous!==undefined)issues.push(key(examples[previous].answer)===key(e.answer)?'Examples '+(previous+1)+' and '+(i+1)+' are duplicates.':'Examples '+(previous+1)+' and '+(i+1)+' give different answers to the same input.');
    seen.set(k,i);
  }
  return [...new Set(issues)];
}
export function examplesError(draft) {
  if(draft.examples.length<3)return 'Add at least 3 examples to preview the teaching flow. This is a preview minimum, not a training-quality guarantee.';
  const issues=datasetIssues(draft.examples);if(issues.length)return issues[0];
  if(draft.examples.some(e=>!e.approved))return 'Review and approve every example before continuing.';
  return '';
}
export function testError(draft) {
  if(!text(draft.test.question)||!text(draft.test.answer))return 'Add a new test question and what a good answer should do.';
  if(draft.examples.some(e=>key(e.question)===key(draft.test.question)))return 'Use a new test question. Training examples cannot test whether the model learned to generalize.';
  return '';
}
export function trainingBrief(draft, model) {
  const error=goalError(draft)||(!model?'Choose a model.':'')||examplesError(draft)||testError(draft);
  if(error)throw Error(error);
  return {schema:TEACH_VERSION,execution:'not-submitted',demo:false,name:draft.name||'My '+(taskTypes[draft.task]?.title.toLowerCase()||'model'),task:draft.task,goal:draft.goal,language:draft.language,referenceNotes:draft.source,
    baseModel:{id:model.id,repository:model.hf_id,revision:model.repository_revision||null},
    target:{delivery:draft.delivery,deviceId:draft.deviceConfirmed?draft.deviceId:null,fit:'requires-post-training-validation'},
    dataset:{count:draft.examples.length,examples:draft.examples.map(e=>({question:e.question,answer:e.answer,...(e.system?{system:e.system}:{}),approved:true,origin:e.origin}))},
    evaluation:{heldOut:true,question:draft.test.question,expected:draft.test.answer},
    training:{method:'lora_sft',preset:draft.trainingPreset||'quick',eligibility:'server-validation-required',fullDatasetReviewRequired:true},
    notifications:{connected:false}};
}
export function datasetJSONL(draft) {
  return draft.examples.map(e=>JSON.stringify({messages:[{role:'system',content:e.system||draft.goal},{role:'user',content:e.question},{role:'assistant',content:e.answer}]})).join('\n')+'\n';
}
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const optionalType = (value, fields, type) => fields.every(field => value[field] === undefined || typeof value[field] === type);
const uuid=value=>typeof value==='string'&&/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(value);
export function validJobReference(value){
  return record(value)&&uuid(value.requestId)&&uuid(value.projectId)&&(value.jobId==null||uuid(value.jobId))
    &&['generation','training'].includes(value.kind)&&typeof value.modelId==='string'&&value.modelId.length>0&&value.modelId.length<=128
    &&typeof value.ownerUid==='string'&&value.ownerUid.length>0&&value.ownerUid.length<=128
    &&['submitting','queued','running','cancelling','succeeded','failed','cancelled'].includes(value.status)
    &&(value.kind==='generation'?['preview','dataset'].includes(value.purpose):['quick','standard'].includes(value.preset))
    &&optionalType(value,['applied'],'boolean')&&optionalType(value,['stage'],'string')
    &&(value.updatedAt==null||typeof value.updatedAt==='string');
}
function validSavedExample(value) {
  return record(value) && typeof value.question === 'string' && typeof value.answer === 'string'
    && optionalType(value,['id','system','origin','kind','seedId'],'string')
    && optionalType(value,['approved','needsReplacement'],'boolean');
}
function validSavedDraft(value) {
  // Missing optional fields belong to older drafts; malformed present fields do not.
  return record(value) && value.schema === TEACH_VERSION && typeof value.id === 'string' && value.id.trim() !== ''
    && optionalType(value,['name','task','goal','source','language','modelId','delivery','deviceId','status','sourceMode','referenceName','datasetName','generationModelId','trainingPreset'],'string')
    && optionalType(value,['deviceConfirmed','datasetBuilt','datasetConfirmed','datasetStale','previewConfirmed'],'boolean')
    && ['stage','flowVersion','generationTarget'].every(field => value[field] === undefined || Number.isSafeInteger(value[field]))
    && (value.updatedAt == null || typeof value.updatedAt === 'string')
    && ['examples','seeds','evaluationSamples'].every(field => value[field] === undefined || (Array.isArray(value[field]) && Array.from(value[field]).every(validSavedExample)))
    && (value.test === undefined || validSavedExample(value.test))
    && (value.gpuProjectId===undefined||uuid(value.gpuProjectId))
    && (value.gpuJob==null||validJobReference(value.gpuJob));
}
const unreadableDrafts = () => Error('Saved teaching projects contain unreadable data. The original data has been kept; saving is blocked to avoid overwriting it.');
export function readDrafts(storage) {
  const value=storage.getItem(TEACH_STORAGE);if(value == null)return [];
  let rows;try{rows=JSON.parse(value);}catch{throw unreadableDrafts();}
  if(!Array.isArray(rows)||!rows.every(validSavedDraft)||new Set(rows.map(d=>d.id)).size!==rows.length)throw unreadableDrafts();
  return rows;
}
export function saveDraft(storage,draft,id,now=new Date().toISOString()) {
  const item={...draft,id:draft.id||id,updatedAt:now};
  if(!validSavedDraft(item))throw Error('This teaching project contains invalid data and could not be saved. Your saved projects have not been changed.');
  const rows=readDrafts(storage);storage.setItem(TEACH_STORAGE,JSON.stringify([item,...rows.filter(d=>d.id!==item.id)]));return item;
}
