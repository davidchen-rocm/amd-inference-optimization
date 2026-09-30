// Reproducible lexicographic priorities, never a synthetic match percentage.
export const policyVersion='recommendations-v1';
export const tasks={general:'Daily Q&A',coding:'Coding',writing:'Writing',documents:'Document Q&A'};
export const languages={en:'English',zh:'Chinese',multi:'Multilingual'};
export const preferences={quality:'Quality first',balanced:'Balanced',speed:'Speed first'};
export function normalizeNeeds(value={}){return {task:Object.hasOwn(tasks,value.task)?value.task:'general',language:Object.hasOwn(languages,value.language)?value.language:'en',preference:Object.hasOwn(preferences,value.preference)?value.preference:'balanced',context:[4096,8192,16384].includes(Number(value.context))?Number(value.context):8192};}
export function confirmedConfiguration(saved,configs){return saved?.confirmed===true&&typeof saved.confirmedAt==='string'&&Number.isFinite(Date.parse(saved.confirmedAt))?configs.find(c=>c.id===saved.deviceId)||null:null;}
const nominalPrecision={'Q8_0':8,'Q6_K':6,'Q5_K_M':5,'Q4_K_M':4,'Q4_0':4,'MLX int4':4,'MXFP4':4,'UD-Q4_K_M':4};
const memory=row=>Number.isFinite(Number(row.estimated_required_bytes))&&row.estimated_required_bytes!=null?Number(row.estimated_required_bytes):Infinity;
const safeURL=url=>typeof url==='string'&&url.startsWith('https://');
export function measuredOption(row,artifact,config,benchmarks){return benchmarks.find(b=>['community_reported','independent_measurement','measured'].includes(b.evidence_type)&&b.model_id===row.model_id&&b.artifact_id===artifact.id&&b.device_config_id===config.id&&Number(b.context_tokens)===Number(row.context_tokens)&&b.runtime===artifact.runtime_family&&b.runtime_version&&b.workload_id&&b.test_protocol&&b.batch_size!=null&&b.prompt_tokens!=null&&b.output_tokens!=null&&b.measured_at&&safeURL(b.source_url)&&Number(b.generation_tokens_per_second)>0)||null;}
function taskFact(model,needs){const e=model.recommendation_evidence;const f=e?.revision===model.repository_revision?e.tasks?.[needs.task]:null;return f&&['publisher_statement','independent_task_evaluation'].includes(f.basis)&&[1,2,3].includes(f.level)&&safeURL(f.source_url)?f:null;}
function languageFact(model,needs){const e=model.recommendation_evidence;const f=e?.revision===model.repository_revision?e.languages?.[needs.language]:null;return f&&['publisher_statement','publisher_metadata','independent_task_evaluation'].includes(f.basis)&&safeURL(f.source_url)?f:null;}
function chooseOption(options,preference){
 return [...options].sort((a,b)=>{
  if(preference==='quality'){const precision=(nominalPrecision[b.row.quantization]||0)-(nominalPrecision[a.row.quantization]||0);if(precision)return precision;}
  if(preference==='speed'&&a.measured&&b.measured&&speedKey(a.measured)===speedKey(b.measured)){const d=b.measured.generation_tokens_per_second-a.measured.generation_tokens_per_second;if(d)return d;}
  return memory(a.row)-memory(b.row)||a.artifact.id.localeCompare(b.artifact.id);
 })[0];
}
const speedKey=b=>JSON.stringify([b.device_config_id,b.runtime,b.runtime_version,b.workload_id,b.test_protocol,b.context_tokens,b.batch_size,b.prompt_tokens,b.output_tokens]);
function commonQuality(candidates,needs,evaluations){
 const cohorts=new Map();
 for(const e of evaluations||[]){if(e.task!==needs.task||e.eligible_for_ranking!==true||!safeURL(e.source_url)||!e.metric_version||!e.protocol_id||e.model_scope!=='base_checkpoint'||!e.snapshot_date||!Number.isFinite(e.score)||typeof e.higher_is_better!=='boolean')continue;
  if(e.language!==needs.language)continue;
  const key=JSON.stringify([e.leaderboard_id,e.metric,e.metric_version,e.protocol_id,e.snapshot_date,e.language,e.higher_is_better]);
  if(!cohorts.has(key))cohorts.set(key,new Map());cohorts.get(key).set(e.model_id,e);
 }
 return [...cohorts.values()].find(rows=>candidates.length>1&&candidates.every(c=>rows.has(c.model.id)))||null;
}
export function recommend({models,artifacts,assessments,config,needs:rawNeeds,benchmarks=[],evaluations=[]}){
 const needs=normalizeNeeds(rawNeeds),byModel=new Map(models.map(m=>[m.id,m])),byArtifact=new Map(artifacts.map(a=>[a.id,a]));
 const groups=new Map();const rejects={memory:new Set(),context:new Set(),runtime:new Set(),evidence:new Set()};
 const seen=new Set();
 for(const row of assessments){
  if(row.device_id!==config.id||Number(row.context_tokens)!==needs.context)continue;
  const model=byModel.get(row.model_id);if(!model)continue;
  seen.add(model.id);
  if(row.verdict==='insufficient_memory')rejects.memory.add(model.id);
  if(model.context_length&&needs.context>model.context_length){rejects.context.add(model.id);continue;}
  const artifact=byArtifact.get(row.artifact_id);
  if(!artifact||artifact.model_id!==model.id||artifact.identity_status!=='verified'||artifact.quantization!==row.quantization||artifact.runtime_family!==row.runtime){rejects.evidence.add(model.id);continue;}
  const fact=taskFact(model,needs),language=languageFact(model,needs);
  const measured=measuredOption(row,artifact,config,benchmarks);
  const capacity=Number(config.chip.memory_kind==='discrete'?config.vram_gib:config.ram_gib)*1024**3;
  const components=[row.artifact_bytes,row.kv_cache_bytes,row.policy_reserve_bytes];
  const complete=components.every(n=>n!=null&&Number.isFinite(Number(n))&&Number(n)>=0)&&components.reduce((total,n)=>total+Number(n),0)===memory(row);
  const fitting=row.verdict==='planning_candidate'&&complete&&Number.isFinite(memory(row))&&capacity>0&&memory(row)<=capacity&&model.context_length>=needs.context&&Number(row.artifact_bytes)===Number(artifact.actual_size_bytes)&&row.runtime_support_status==='family_backend_documented';
  const candidate={model,row,artifact,task:fact,language,measured,fitting};
  if(!groups.has(model.id))groups.set(model.id,[]);groups.get(model.id).push(candidate);
  if(!row.runtime_support_status)rejects.runtime.add(model.id);
  if(!fact||!language||!fitting)rejects.evidence.add(model.id);
 }
 const ranked=[],pending=[];
 for(const options of groups.values()){
  const fits=options.filter(o=>o.fitting&&o.task&&o.language);
  const possible=options.filter(o=>o.row.verdict!=='insufficient_memory');
  if(!fits.length&&!possible.length)continue;
  const chosen=chooseOption(fits.length?fits:possible,needs.preference);
  chosen.options=options;chosen.alternatives=options.filter(o=>o.artifact.id!==chosen.artifact.id);
  (fits.length?ranked:pending).push(chosen);
 }
 const quality=commonQuality(ranked,needs,evaluations);
 const speed=ranked.length>1&&ranked.every(c=>c.measured)&&new Set(ranked.map(c=>speedKey(c.measured))).size===1;
 const margin=c=>Number(c.row.headroom_bytes)/(memory(c.row)+Number(c.row.headroom_bytes));
 const safetyBand=c=>margin(c)>=.2?2:margin(c)>=.1?1:0;
 const objectiveCompare=(a,b)=>{
  let d=(b.task?.level||0)-(a.task?.level||0);if(d)return d;
  if(needs.preference==='speed'&&speed){d=b.measured.generation_tokens_per_second-a.measured.generation_tokens_per_second;if(d)return d;}
  if(needs.preference!=='speed'&&quality){const x=quality.get(a.model.id),y=quality.get(b.model.id);d=(x.higher_is_better?-1:1)*(x.score-y.score);if(d)return d;}
  d=safetyBand(b)-safetyBand(a);if(d)return d;
  return memory(a.row)-memory(b.row);
 };
 ranked.sort((a,b)=>objectiveCompare(a,b)||a.model.id.localeCompare(b.model.id));
 ranked.forEach((c,i)=>{c.rank=i+1;c.closeEvidence=(i>0&&ranked[i-1].task.level===c.task.level)||ranked[i+1]?.task.level===c.task.level;c.quality=quality?.get(c.model.id)||null;c.speedComparable=speed;});
 pending.sort((a,b)=>Number(b.fitting)-Number(a.fitting)||(b.task?.level||0)-(a.task?.level||0)||memory(a.row)-memory(b.row)||a.model.id.localeCompare(b.model.id));
 return {ranked,pending,needs,orderAvailable:needs.preference==='balanced'||(needs.preference==='quality'&&!!quality)||(needs.preference==='speed'&&speed),qualityAvailable:!!quality,speedAvailable:!!speed,counts:{ranked:ranked.length,pending:pending.length,catalog:models.length,unassessed:models.filter(m=>!groups.has(m.id)).length,...Object.fromEntries(Object.entries(rejects).map(([k,v])=>[k,v.size]))},policyVersion};
}
export function feedbackRecord(candidate,config,needs,outcome){
 if(!['ran','failed','not-suitable'].includes(outcome))throw new Error('Invalid feedback');
 return {model_id:candidate.model.id,artifact_id:candidate.artifact.id,revision:candidate.artifact.revision,device_id:config.id,needs:normalizeNeeds(needs),outcome,policy_version:policyVersion,recorded_at:new Date().toISOString(),evidence_type:'unverified_user_feedback'};
}
export const shellQuote=value=>"'"+String(value).replaceAll("'","'\\''")+"'";
export function runInstructions(artifact,config,context){
 if(!artifact||artifact.identity_status!=='verified'||!artifact.revision||!Array.isArray(artifact.files)||!artifact.files.length)return null;
 if(!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(artifact.id)||![4096,8192,16384].includes(Number(context)))return null;
 if(!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(artifact.repo_id)||!/^[a-f0-9]{40,64}$/.test(artifact.revision))return null;
 const files=artifact.files.map(f=>f.filename);
 if(files.some(f=>typeof f!=='string'||f.startsWith('/')||f.split('/').includes('..')||/[\r\n\0]/.test(f)))return null;
 const folder='models/'+artifact.id;
 const base=`hf download ${shellQuote(artifact.repo_id)} --revision ${shellQuote(artifact.revision)} --local-dir ${shellQuote(folder)}`;
 if(artifact.format==='GGUF'){
  const first=[...files].sort()[0];
  return {download:base+' --include '+files.map(shellQuote).join(' '),launch:`llama-server -m ${shellQuote(folder+'/'+first)} -c ${Number(context)} -np 1 -ctk f16 -ctv f16 -ngl ${config.chip.memory_kind==='system'?0:999} --host 127.0.0.1 --port 8081`,runtimeURL:'https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md',runtime:'llama.cpp',notes:'Install a build with the appropriate Metal, CUDA, HIP or CPU backend. Keep all GGUF shards together; the first shard opens the set. The command is a proposed launch, not a verified device test.'};
 }
 if(artifact.format==='MLX'&&config.chip.memory_kind==='unified')return {download:base+' --include '+[...files,'*.json','tokenizer*','*.model','*.tiktoken','*.jinja'].map(shellQuote).join(' '),launch:`python -m mlx_lm.generate --model ${shellQuote(folder)} --prompt 'Hello' --max-tokens 128 --temp 0.6 --top-p 0.95 --top-k 20`,runtimeURL:'https://github.com/ml-explore/mlx-lm',runtime:'mlx-lm',notes:`Install mlx-lm in a Python environment on a supported Apple Silicon Mac. The non-greedy settings are a smoke-test starting point, not a quality claim. For real tasks follow the publisher sampling guidance; the application must enforce the selected ${context}-token prompt-plus-output budget.`};
 return null;
}
