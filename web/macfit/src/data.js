export let catalog={schema_version:1,categories:[],models:[],devices:[],benchmarks:[],sources:[],artifacts:[]};
export let hardware={items:[],contexts:[]};
export async function apiGet(path,{signal}={}){
 const controller=new AbortController();const abort=()=>controller.abort();
 if(signal?.aborted)controller.abort();else signal?.addEventListener('abort',abort,{once:true});
 let timedOut=false;
 const timeout=setTimeout(()=>{timedOut=true;controller.abort();},12000);
 try{
  const response=await fetch(path,{cache:'no-store',signal:controller.signal,headers:{Accept:'application/json'}});
  let data=null;try{data=await response.json();}catch{}
  if(!response.ok){const error=new Error(data?.error?.message||'The database service is unavailable.');error.status=response.status;throw error;}
  if(data==null)throw new Error('The database service returned an invalid response.');
  return data;
 }catch(error){
  if(timedOut&&!signal?.aborted)throw new Error('This is taking longer than expected. Please try again.');
  if(error instanceof TypeError&&!signal?.aborted)throw new Error('We couldn’t reach the model service. Check your connection and try again.');
  throw error;
 }finally{clearTimeout(timeout);signal?.removeEventListener('abort',abort);}
}
export async function loadCatalog(){
 const [data,chips]=await Promise.all([apiGet('/api/catalog'),apiGet('/api/chips')]);
 if(data.source!=='postgresql'||chips.source!=='postgresql'||data.schema_version!==1||!['categories','models','devices','benchmarks','sources','runtime_requirements','artifacts'].every(k=>Array.isArray(data[k]))||!Array.isArray(chips.items)||!Array.isArray(chips.contexts))throw new Error('The database response format is not supported.');
 catalog=data;hardware=chips;
 return data;
}
// Exhaust keyset pages over model identities. No artifact-row cap is treated as complete.
export async function fetchCompatibility(deviceId,context,{signal,modelId,verdict,onProgress,artifactsOnly=false}={}){
 const base={device_id:deviceId,context:String(context),limit:'200'};
 if(modelId)base.model_id=modelId;
 if(verdict)base.verdict=verdict;
 if(artifactsOnly)base.artifacts_only='true';
 for(let attempt=0;attempt<2;attempt++){
  let cursor=null,first=null;const cursors=new Set(),models=new Set(),items=[];
  try{
   do{
    if(signal?.aborted)throw new DOMException('The request was aborted.','AbortError');
    const params=new URLSearchParams(base);if(cursor)params.set('cursor',cursor);
    const page=await apiGet('/api/compatibility?'+params,{signal});
    if(!Array.isArray(page.items)||typeof page.has_more!=='boolean'||!Number.isInteger(page.model_count)||!Number.isInteger(page.total_models))throw new Error('The assessment service returned an invalid page.');
    if(first&&(page.catalog_revision!==first.catalog_revision||page.assessment_revision!==first.assessment_revision||page.total_models!==first.total_models)){const e=new Error('The model catalog changed while loading. Please refresh.');e.code='catalog_revision_changed';throw e;}
    first??=page;
    const pageModels=new Set(page.items.map(row=>row.model_id));
    if(pageModels.size!==page.model_count||pageModels.has(null)||pageModels.has(undefined))throw new Error('The assessment page is incomplete. Please refresh.');
    for(const id of pageModels){if(models.has(id))throw new Error('The assessment service repeated a model page.');models.add(id);}
    items.push(...page.items);
    onProgress?.({loadedModels:models.size,totalModels:page.total_models,loadedOptions:items.filter(row=>row.artifact_id).length});
    if(!page.has_more){
     if(models.size!==page.total_models)throw new Error('The assessment inventory is incomplete. Please refresh.');
     return {...first,items,has_more:false,next_cursor:null,model_count:models.size};
    }
    if(!page.next_cursor||cursors.has(page.next_cursor)||page.model_count===0)throw new Error('The assessment service returned a repeated or empty cursor.');
    cursor=page.next_cursor;cursors.add(cursor);
   }while(cursor);
  }catch(error){if(error.code==='catalog_revision_changed'&&attempt===0&&!signal?.aborted)continue;throw error;}
 }
 throw new Error('The model catalog changed while loading. Please refresh.');
}
