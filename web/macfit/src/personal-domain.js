import {recommend,runInstructions} from './recommendations.js';
export const workTypes={coding:'Coding',writing:'Writing',chat:'Chinese conversation',service:'Customer support',documents:'Document summaries',data:'Data analysis',knowledge:'Company knowledge',other:'Something else'};
export const taskFor=work=>({coding:'coding',data:'coding',writing:'writing',documents:'documents',knowledge:'documents'}[work]||'general');
export const storageKey='macfit.personal-models.v1';
export function examplesFromText(text){
 let raw;try{raw=JSON.parse(text);}catch{try{raw=text.split(/\r?\n/).filter(s=>s.trim()).map(s=>JSON.parse(s));}catch{throw Error('Choose a valid JSON or JSONL file.');}}
 if(!Array.isArray(raw))raw=raw?.examples||(raw?.messages||raw?.question?[raw]:null);
 if(!Array.isArray(raw)||raw.length>2000)throw Error('Use an array of up to 2,000 examples.');
 return raw.map((x,i)=>{const messages=Array.isArray(x?.messages)?x.messages:[];const question=x?.question??messages.find(m=>m?.role==='user')?.content;const answer=x?.answer??messages.find(m=>m?.role==='assistant')?.content;if(typeof question!=='string'||typeof answer!=='string'||!question.trim()||!answer.trim()||question.length>12000||answer.length>24000)throw Error(`Example ${i+1} needs a question and answer within the length limits.`);return {question:question.trim(),answer:answer.trim()};});
}
export function analyzeNeeds(draft){
 const count=draft.examples?.length||0,knowledge=draft.work==='knowledge';
 const training=count>=50&&['service','other'].includes(draft.work);
 return {kind:knowledge?'knowledge':training?'training-review':'instructions',title:knowledge?'Give it access to your knowledge first':training?'Try your configuration, then consider training':'Your preferences. No retraining needed to start.',summary:knowledge?'Company knowledge needs your source material. Instructions can guide how it answers, but cannot give it access to your company information.':training?`You added ${count} examples. Start with a few representative answers, then assess whether repeated issues call for a dedicated version.`:'Your description and examples can guide its tone, answer structure and working preferences. Start with a personal configuration.',steps:knowledge?['Create personal instructions','Set rules for sources and unknown answers','Save your knowledge connection plan']:training?['Create instructions you can use now','Keep your complete example set','Save a plan for training evaluation']:['Create personal instructions','Include up to 3 representative examples','Use the model’s default generation settings'],pending:knowledge?'Knowledge connections are coming later. This version includes instructions only.':training?'Training is coming later. This version uses your instructions and leaves model weights unchanged.':''};
}
export function systemPrompt(draft){return `You are the user's ${workTypes[draft.work]||'work'} assistant.\n${draft.instructions.trim()}${draft.work==='knowledge'?'\nAnswer company questions only from supplied sources. Say when information is unavailable; do not invent company facts.':''}`;}
export function fittingOptions(catalog,rows,config,needs,modelId){
 const result=recommend({models:catalog.models,artifacts:catalog.artifacts,assessments:rows,config,needs,benchmarks:catalog.benchmarks,evaluations:catalog.quality_evaluations});
 return [...result.ranked,...result.pending].flatMap(c=>c.options).filter(c=>c.model.id===modelId&&c.fitting&&runInstructions(c.artifact,config,needs.context)).sort((a,b)=>Number(a.row.estimated_required_bytes)-Number(b.row.estimated_required_bytes));
}
export function validateDraft(d){
 if(typeof d?.name!=='string'||!d.name.trim()||d.name.length>80)throw Error('Give your model a name between 1 and 80 characters.');
 if(!Object.hasOwn(workTypes,d.work))throw Error('Choose what you will use it for.');
 if(typeof d.instructions!=='string'||!d.instructions.trim()||d.instructions.length>6000)throw Error('Describe how it should answer in 1–6,000 characters.');
 if(!Array.isArray(d.examples)||d.examples.length>2000)throw Error('You can save up to 2,000 examples.');
 examplesFromText(JSON.stringify(d.examples));
 if(JSON.stringify(d).length>1800000)throw Error('Your examples are too large. Reduce them to under 1.8 MB.');
 return d;
}
function validSavedModel(item){
 try{
  if(!item||typeof item!=='object'||Array.isArray(item)||typeof item.id!=='string'||!/^[A-Za-z0-9_-]{1,128}$/.test(item.id))return false;
  validateDraft(item);
  if(['modelId','artifactId','deviceId','systemPrompt'].some(key=>typeof item[key]!=='string'||!item[key]))return false;
  if(!Number.isInteger(item.version)||item.version<1||!Number.isInteger(item.context)||item.context<512||item.context>65536)return false;
  if(!Number.isFinite(Date.parse(item.createdAt))||!Number.isFinite(Date.parse(item.updatedAt)))return false;
  return true;
 }catch{return false;}
}
export function readModelStore(storage){
 let raw;
 try{storage ||= globalThis.localStorage;raw=storage.getItem(storageKey);}catch{return {items:[],raw:null,blocked:true,warning:'Browser storage is unavailable. Allow storage for this website to open or save local setups.'};}
 if(!raw)return {items:[],raw:null,blocked:false,warning:''};
 let rows;
 try{rows=JSON.parse(raw);}catch{return {items:[],raw,blocked:true,warning:'Your saved setups could not be read. Download a backup below before repairing your browser data. Nothing has been changed.'};}
 if(!Array.isArray(rows))return {items:[],raw,blocked:true,warning:'Your saved setups use an unsupported format. Download a backup below before repairing your browser data. Nothing has been changed.'};
 const seen=new Set(),items=[];
 for(const row of rows){if(validSavedModel(row)&&!seen.has(row.id)){seen.add(row.id);items.push(row);}}
 const skipped=rows.length-items.length;
 return {items,raw,blocked:skipped>0,warning:skipped?`${skipped} saved ${skipped===1?'setup could':'setups could'} not be opened. Your other setups are available. Download a backup below before repairing the saved data; nothing has been removed.`:''};
}
export function loadModels(storage){return readModelStore(storage).items;}
export function saveModel(draft,selection,config,context,{storage,now=new Date().toISOString(),id=crypto.randomUUID()}={}){
 validateDraft(draft);if(!selection?.fitting)throw Error('Choose a version that fits the selected device budget first.');
 try{storage ||= globalThis.localStorage;}catch{throw Error('Browser storage is unavailable. Allow storage for this website before saving local setups.');}
 const stored=readModelStore(storage);if(stored.blocked)throw Error('Your saved setups need attention. Open My models and download a backup before saving changes. Your original data has been kept.');
 const all=stored.items,previous=all.find(m=>m.id===draft.id);const item={...draft,id:previous?.id||id,name:draft.name.trim(),instructions:draft.instructions.trim(),modelId:selection.model.id,artifactId:selection.artifact.id,revision:selection.artifact.revision,deviceId:config.id,context,version:(previous?.version||0)+1,createdAt:previous?.createdAt||now,updatedAt:now,plan:analyzeNeeds(draft),systemPrompt:systemPrompt(draft),status:'configured',execution:'not-run'};
 delete item.analysis;delete item.pendingQuestion;delete item.pendingAnswer;storage.setItem(storageKey,JSON.stringify([item,...all.filter(m=>m.id!==item.id)]));return item;
}
export function runtimeSpec(item,artifact,config){
 if(item.revision!==artifact.revision)throw Error('The model files have changed. Update your configuration and check again.');
 const guide=runInstructions(artifact,config,item.context);if(!guide)throw Error('This version does not have a supported local launcher yet.');
 const messages=[{role:'system',content:item.systemPrompt}];let used=item.systemPrompt.length;
 for(const e of item.examples.slice(0,3)){if(used+e.question.length+e.answer.length>Math.min(9000,item.context))break;messages.push({role:'user',content:e.question},{role:'assistant',content:e.answer});used+=e.question.length+e.answer.length;}
 return {name:item.name,repo:artifact.repo_id,revision:artifact.revision,files:artifact.files.map(f=>f.filename),format:artifact.format,folder:'models/'+artifact.id,context:item.context,cpu:config.chip.memory_kind==='system',messages};
}
