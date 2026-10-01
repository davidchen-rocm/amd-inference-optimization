import {getAccessToken,getAccount} from './account.js';

export const terminalJob=job=>['succeeded','failed','cancelled'].includes(job?.status);
export const activeJob=job=>!!job&&!terminalJob(job);
export const trainingError=error=>error?.message||'The GPU service could not be reached. Please try again.';
export const validUUID=value=>typeof value==='string'&&/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(value);
const identifier=value=>{if(!validUUID(value))throw Error('This job reference is invalid. Reopen your project.');return encodeURIComponent(value);};
const base='/api/training';

export function createTrainingClient({getToken=getAccessToken,getUserId=()=>getAccount().user?.uid,fetcher=(...args)=>fetch(...args),timeoutMs=15000,binaryTimeoutMs=180000}={}){
 async function request(path,{method='GET',body,signal,publicRequest=false,binary=false}={}){
  const owner=publicRequest?null:getUserId();
  if(!publicRequest&&!owner){const error=Error('Sign in with Google before using GPU generation or training.');error.code='sign_in_required';throw error;}
  const controller=new AbortController(),abort=()=>controller.abort();
  if(signal?.aborted)controller.abort();else signal?.addEventListener('abort',abort,{once:true});
  let timedOut=false;const timer=setTimeout(()=>{timedOut=true;controller.abort();},binary?binaryTimeoutMs:timeoutMs);
  try{
   for(let attempt=0;attempt<2;attempt++){
    if(controller.signal.aborted)throw new DOMException('Request canceled','AbortError');
    const headers={Accept:binary?'application/octet-stream':'application/json'};
    if(body!==undefined)headers['Content-Type']='application/json';
    if(!publicRequest)headers.Authorization='Bearer '+await getToken(attempt===1);
    if(!publicRequest&&getUserId()!==owner)throw Error('Your account changed. Reopen this project before continuing.');
    const response=await fetcher(base+path,{method,headers,body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal,cache:'no-store',credentials:'same-origin'});
    if(!publicRequest&&getUserId()!==owner)throw Error('Your account changed. Reopen this project before continuing.');
    if(response.status===401&&!publicRequest&&attempt===0)continue;
    if(binary&&response.ok){const blob=await response.blob();if(getUserId()!==owner)throw Error('Your account changed. Reopen this project before continuing.');return blob;}
    let value;try{value=await response.json();}catch{throw Error('The GPU service returned an unreadable response. Please try again.');}
    if(!publicRequest&&getUserId()!==owner)throw Error('Your account changed. Reopen this project before continuing.');
    if(!response.ok){const error=Error(value?.error?.message||'The GPU service could not complete this request.');error.code=value?.error?.code;error.status=response.status;throw error;}
    if(!value||typeof value!=='object')throw Error('The GPU service returned an invalid response.');
    return value;
   }
  }catch(error){
   if(timedOut){const failure=Error('The GPU service did not reply in time. Check the existing job before submitting again.');failure.ambiguous=method==='POST';throw failure;}
   if(error instanceof TypeError){const failure=Error('The GPU service could not be reached. Check your connection and try again.');failure.ambiguous=method==='POST';throw failure;}
   throw error;
  }finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
 }
 return {
  capabilities:options=>request('/capabilities',{...options,publicRequest:true}),
  create:(payload,options)=>request('/jobs',{...options,method:'POST',body:payload}),
  get:(id,options)=>request('/jobs/'+identifier(id),options),
  list:({projectId,requestId},options)=>request('/jobs?'+new URLSearchParams(requestId?{request_id:identifier(requestId)}:{project_id:identifier(projectId)}),options),
  cancel:(id,options)=>request('/jobs/'+identifier(id)+'/cancel',{...options,method:'POST'}),
  artifact:(jobId,artifactId,options)=>{if(typeof artifactId!=='string'||!artifactId||artifactId.length>200)throw Error('This output reference is invalid.');return request('/jobs/'+identifier(jobId)+'/artifacts/'+encodeURIComponent(artifactId),{...options,binary:true});}
 };
}

export const trainingClient=createTrainingClient();

// Recover an ambiguous submission first; replaying the same request is safe.
export async function recoverSubmission(client,payload,options){
 const list=await client.list({requestId:payload.request_id},options);
 if(!Array.isArray(list.items))throw Error('The GPU service returned an invalid job list.');
 const found=list.items.find(job=>job.request_id===payload.request_id&&job.project_id===payload.project_id);
 return found||client.create(payload,options);
}

export function jobReference(job,previous){
 if(!job||!validUUID(job.id)||job.request_id!==previous.requestId||job.project_id!==previous.projectId||job.kind!==previous.kind||!['queued','running','cancelling','succeeded','failed','cancelled'].includes(job.status))throw Error('The GPU service returned a mismatched job. Your project has not been changed.');
 return {...previous,jobId:job.id,status:job.status,stage:typeof job.stage==='string'?job.stage:job.status,updatedAt:job.updated_at||null};
}
