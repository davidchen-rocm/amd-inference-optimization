export function configurations(hardware){return hardware.items.flatMap(chip=>(chip.configurations||[]).map(config=>({...config,chip})));}
export function profileFromConfiguration(config){
 return {deviceConfigId:config.id,family:config.chip.memory_kind==='unified'?'mac':'pc',device:config.label,chip:config.chip.name.replace(/^Apple /,''),memory:config.ram_gib==null?null:Number(config.ram_gib),vram:config.vram_gib==null?null:Number(config.vram_gib),gpu:config.chip.name,referenceDeviceId:null,inferenceMode:config.chip.memory_kind==='system'?'cpu':'gpu'};
}
export function groupAssessments(items){const groups=new Map();for(const item of items){if(!groups.has(item.model_id))groups.set(item.model_id,[]);groups.get(item.model_id).push(item);}return groups;}
export function preferredOption(options,preference='Balanced'){
 const candidates=options.filter(o=>o.verdict==='planning_candidate'&&o.artifact_id);
 if(!candidates.length)return null;
 if(preference==='Lowest Memory')return [...candidates].sort((a,b)=>Number(a.estimated_required_bytes)-Number(b.estimated_required_bytes))[0];
 const ranks=preference==='Best Quality'?['Q8_0','Q5_K_M','Q4_K_M','MLX int4']:['Q5_K_M','Q4_K_M','MLX int4','Q8_0'];
 const rank=q=>{const i=ranks.indexOf(q);return i<0?99:i;};
 return [...candidates].sort((a,b)=>rank(a.quantization)-rank(b.quantization)||Number(a.estimated_required_bytes)-Number(b.estimated_required_bytes))[0];
}
export function assessmentForBuild(items,build){return items.find(r=>r.model_id===build.model&&r.quantization===build.quant&&(!build.artifact||r.artifact_id===build.artifact)&&Number(r.context_tokens)===build.context&&r.artifact_id)||null;}
export const verdictLabels={planning_candidate:'Planning candidate',insufficient_memory:'Over memory budget',unknown:'Not assessed / incomplete evidence'};

export function assessmentLabel(row){return ['weights_fit_kv_unverified','weights_fit_context_unverified'].includes(row?.memory_status)?'Weights fit; runtime unverified':verdictLabels[row?.verdict]||'Not assessed';}
export function paginateModels(models,page=1,pageSize=24){
 const pages=Math.max(1,Math.ceil(models.length/pageSize));
 const current=Math.min(pages,Math.max(1,Number.isInteger(page)?page:1));
 return {items:models.slice((current-1)*pageSize,current*pageSize),page:current,pages,total:models.length,start:models.length?(current-1)*pageSize+1:0,end:Math.min(current*pageSize,models.length)};
}
export function filterCatalog(models,{search='',category='all',publisher='all',sort='popular'}={}){
 const query=search.trim().toLowerCase();
 return models.filter(m=>(category==='all'||m.task_tags?.includes(category))&&(publisher==='all'||m.owner===publisher)&&`${m.name} ${m.owner} ${m.hf_id}`.toLowerCase().includes(query)).sort((a,b)=>{
  if(sort==='smallest')return (a.parameter_count??Infinity)-(b.parameter_count??Infinity)||a.id.localeCompare(b.id);
  if(sort==='newest'||sort==='updated'){const key=sort==='newest'?'repository_created_at':'repository_last_modified_at';return (b[key]||'').localeCompare(a[key]||'')||a.id.localeCompare(b.id);}
  return (b.downloads_30d||0)-(a.downloads_30d||0)||a.id.localeCompare(b.id);
 });
}
