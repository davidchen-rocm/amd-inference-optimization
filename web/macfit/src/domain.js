// Pure data rules: measured throughput is never scaled or invented.
export const chipMem = {'M1':[8,16],'M1 Pro':[16,32],'M1 Max':[32,64],'M2':[8,16,24],'M2 Pro':[16,32],'M2 Max':[32,64,96],'M3':[8,16,24],'M3 Pro':[18,36],'M3 Max':[36,48,64,96,128],'M4':[16,24,32],'M4 Pro':[24,48,64],'M4 Max':[36,48,64,128]};
export const defaultProfile={family:'mac',device:'MacBook Pro',chip:'M3 Pro',memory:18,vram:null,gpu:null,referenceDeviceId:null,inferenceMode:'gpu'};
export const quants=[
 {id:'MLX int4',bits:4,type:'MLX',note:'An MLX configuration. Only a source-verified artifact and a supported device backend can receive a database assessment.'},
 {id:'Q4_0',bits:4,type:'GGUF',note:'Four-bit weights with block metadata. This is a planner option, not a verified artifact for every model.'},
 {id:'Q4_K_M',bits:4,type:'GGUF',note:'Mixed K-quants. File size is larger than the four-bit theoretical weight floor.'},
 {id:'Q5_K_M',bits:5,type:'GGUF',note:'Five-bit planning floor; block metadata and mixed precision add overhead.'},
 {id:'Q6_K',bits:6,type:'GGUF',note:'Six-bit planning floor; actual model files and runtime usage are larger.'},
 {id:'Q8_0',bits:8,type:'GGUF',note:'Eight-bit planning floor, excluding quantization metadata.'},
 {id:'F16',bits:16,type:'GGUF',note:'Sixteen-bit weight storage. Runtime support and original checkpoint dtype must be checked.'}
];
// Register publisher artifact formats without guessing bit precision from their names.
const recordedArtifacts=new Map();
export function registerArtifactQuants(artifacts){
 for(const artifact of artifacts||[]){
  if(typeof artifact.quantization!=='string'||!artifact.quantization.trim())continue;
  if(artifact.id)recordedArtifacts.set(artifact.id,artifact);
  const metadata=artifact.metadata||artifact;
  const bits=typeof metadata.weight_bits==='number'&&metadata.weight_bits>0?metadata.weight_bits:null;
  const existing=quants.find(q=>q.id===artifact.quantization);
  if(existing){existing.bits=existing.recorded&&existing.bits!==bits?null:bits;existing.recorded=true;continue;}
  quants.push({id:artifact.quantization,bits,type:artifact.format||'Recorded format',recorded:true,note:'A recorded artifact format. Fit uses the actual file size and its stored device assessment; bit precision is shown only when supplied by the artifact metadata.'});
 }
 return quants;
}
export const presets={
 General:{context:16384,temperature:.7,prompt:'You are a helpful, clear, and thoughtful assistant.'},
 Coding:{context:32768,temperature:.2,prompt:'You are a careful coding assistant. Explain tradeoffs and provide concise, working code.'},
 Thinking:{context:16384,temperature:.6,prompt:'Consider the problem carefully and explain the key reasoning steps clearly.'},
 Writing:{context:16384,temperature:.85,prompt:'Help write clear, engaging prose for the intended reader.'},
 Chinese:{context:16384,temperature:.7,prompt:'请使用自然、清晰的中文回答，并根据问题给出具体、实用的解释。'},
 Math:{context:16384,temperature:.6,prompt:'Explain mathematical ideas clearly and check the result.'},
 RAG:{context:32768,temperature:.1,prompt:'Answer using supplied context. Cite relevant passages and acknowledge missing information.'}
};
export const useCases=['General','Coding','Thinking','Writing','RAG'];
export function normalizeProfile(raw){
 if(!raw||typeof raw!=='object')return {...defaultProfile};
 const p={...defaultProfile,...raw};
 if(p.family==='pc')return {...p,memory:Number(p.memory)>0?Number(p.memory):32,vram:Number(p.vram)>0?Number(p.vram):null};
 if(p.referenceDeviceId)return {...p,family:'mac',memory:Number(p.memory)>0?Number(p.memory):null};
 if(!chipMem[p.chip]||!chipMem[p.chip].includes(p.memory))return {...defaultProfile};
 return {...p,family:'mac',vram:null};
}
export const isEmbedding=m=>(m.task_tags||[]).includes('embedding');
export const modelLimit=m=>Number.isInteger(m.context_length)&&m.context_length>0?m.context_length:null;
export function weightsFloor(model,bits){return Number(model.parameter_count)>0&&Number.isFinite(bits)&&bits>0?model.parameter_count*bits/8/(1024**3):null;}
export function memoryPlan(model,quant,context,profile){
 const weights=weightsFloor(model,quant.bits);
 const budget=profile.family==='pc'&&profile.inferenceMode!=='cpu'?profile.vram:profile.memory;
 // A KV-cache estimate is shown only for documented standard full-attention architectures.
 const c=model.config||{};
 const kvSupported=c.memory_estimate_supported===true&&c.num_hidden_layers>0&&c.num_key_value_heads>0&&c.head_dim>0;
 const kv=kvSupported?2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*context*2/(1024**3):null;
 const lowerBound=weights===null?null:weights+(kv||0);
 const over=budget!=null&&lowerBound!=null&&lowerBound>budget;
 return {weights,kv,budget,lowerBound,headroom:budget!=null&&lowerBound!=null?budget-lowerBound:null,
  level:over?'poor':budget==null||weights==null?'unknown':'conditional',
  fit:over?'Exceeds memory budget':budget==null?'Memory budget unknown':weights==null?'Requirements not available':'Within planning budget',
  memoryKind:profile.inferenceMode==='cpu'?'System RAM':profile.family==='pc'?'GPU memory (VRAM)':'Unified memory',
  speed:null};
}
export function defaultBuild(model,profile,preference='Balanced',use='General'){
 const preset=presets[use]||presets.General;
 let quant=preference==='Best Quality'?'Q6_K':preference==='Balanced'?'Q5_K_M':'Q4_K_M';
 const context=modelLimit(model)==null?8192:Math.min(preset.context,modelLimit(model));
 if(memoryPlan(model,quants.find(q=>q.id===quant),context,profile).level==='poor')quant='Q4_K_M';
 return {model:model.id,quant,context,use,temperature:preset.temperature,topP:.9,systemPrompt:preset.prompt,...(modelLimit(model)==null?{contextIsAssumption:true}:{})};
}
export function buildId(b){const canonical={model:b.model,quant:b.quant,...(b.artifact?{artifact:b.artifact}:{}),context:b.context,use:b.use,temperature:b.temperature,topP:b.topP,systemPrompt:b.systemPrompt,...(b.contextIsAssumption===true?{contextIsAssumption:true}:{})};return 'b1_'+btoa(unescape(encodeURIComponent(JSON.stringify(canonical)))).replaceAll('+','-').replaceAll('/','_').replaceAll('=','');}
export function parseBuild(id,models){try{
 if(!id?.startsWith('b1_'))return null;
 const b=JSON.parse(decodeURIComponent(escape(atob(id.slice(3).replaceAll('-','+').replaceAll('_','/')))));
 const m=models.find(m=>m.id===b.model);
 if(!m||isEmbedding(m)||!quants.some(q=>q.id===b.quant)||!presets[b.use])return null;
 if(b.artifact){const a=recordedArtifacts.get(b.artifact);if(!a||a.model_id!==b.model||a.quantization!==b.quant)return null;}
 if(!Number.isInteger(b.context)||b.context<512||b.context>Math.min(65536,modelLimit(m)??65536)||(modelLimit(m)==null&&b.contextIsAssumption!==true)||!Number.isFinite(b.temperature)||b.temperature<0||b.temperature>2||!Number.isFinite(b.topP)||b.topP<0||b.topP>1||typeof b.systemPrompt!=='string'||b.systemPrompt.length>1000)return null;
 return b;
}catch{return null;}}
export function exactMeasurements(model,profile,benchmarks){
 if(!profile.referenceDeviceId)return [];
 return benchmarks.filter(b=>b.model_id===model.id&&b.device_id===profile.referenceDeviceId);
}
export function comparableMeasurements(rows){
 const keys=['test_protocol','model_id','quantization','runtime','runtime_version','context_tokens','prompt_tokens','output_tokens','batch_size','gpu_layers'];
 return rows.length>1&&rows.every(r=>['separate_pp_tg','sequential_generate'].includes(r.test_protocol))&&keys.every(key=>rows[0][key]!=null&&rows.every(row=>row[key]===rows[0][key]));
}
