import test from 'node:test';
import assert from 'node:assert/strict';
import {createTrainingClient,recoverSubmission,jobReference,terminalJob} from '../src/training-client.js';

const project='10000000-0000-4000-8000-000000000001',request='10000000-0000-4000-8000-000000000002',id='10000000-0000-4000-8000-000000000003';
const payload={request_id:request,project_id:project,kind:'training',input:{model_id:'qwen3-0-6b'}};
const job={id,request_id:request,project_id:project,kind:'training',status:'queued',stage:'queued',updated_at:'2026-10-01T00:00:00Z'};

test('public capabilities work without login; GPU submission requires an authenticated account',async()=>{
 const calls=[];const client=createTrainingClient({getUserId:()=>null,getToken:()=>{throw Error('must not request a token');},fetcher:async(path,options)=>{calls.push({path,options});return Response.json({available:true,models:[]});}});
 assert.equal((await client.capabilities()).available,true);assert.equal(calls[0].options.headers.Authorization,undefined);
 await assert.rejects(client.create(payload),/Sign in with Google/);assert.equal(calls.length,1);
});

test('401 refreshes the ID token once and replays the same idempotent request',async()=>{
 const tokens=[],calls=[];const client=createTrainingClient({getUserId:()=> 'alice',getToken:async force=>{tokens.push(force);return force?'fresh':'old';},fetcher:async(path,options)=>{calls.push({path,options});return calls.length===1?Response.json({error:{message:'Expired'}},{status:401}):Response.json(job,{status:202});}});
 assert.deepEqual(await client.create(payload),job);assert.deepEqual(tokens,[false,true]);
 assert.equal(calls[0].options.body,calls[1].options.body);assert.equal(calls[1].options.headers.Authorization,'Bearer fresh');assert.equal(calls[1].path,'/api/training/jobs');
});

test('an authorization error after refresh is surfaced and never loops',async()=>{
 let calls=0;const client=createTrainingClient({getUserId:()=> 'alice',getToken:async()=> 'token',fetcher:async()=>{calls++;return Response.json({error:{code:'invalid_token',message:'Sign in again.'}},{status:401});}});
 await assert.rejects(client.get(id),error=>error.status===401&&error.message==='Sign in again.');assert.equal(calls,2);
});

test('ambiguous submission recovery returns an existing job without launching another',async()=>{
 let creates=0;const client={list:async query=>{assert.equal(query.requestId,request);return {items:[job]};},create:async()=>{creates++;}};
 assert.equal(await recoverSubmission(client,payload),job);assert.equal(creates,0);
});

test('a never-received submission is replayed with its original request and project IDs',async()=>{
 const received=[];const client={list:async()=>({items:[]}),create:async value=>{received.push(value);return job;}};
 assert.equal(await recoverSubmission(client,payload),job);assert.equal(received[0],payload);
});

test('network loss is ambiguous for submissions but not mistaken for a completed job',async()=>{
 const client=createTrainingClient({getUserId:()=> 'alice',getToken:async()=> 'token',fetcher:async()=>{throw new TypeError('Failed to fetch');}});
 await assert.rejects(client.create(payload),error=>error.ambiguous===true&&/could not be reached/.test(error.message));
});

test('leaving the view aborts only its read request and never sends cancellation',async()=>{
 const paths=[];const client=createTrainingClient({getUserId:()=> 'alice',getToken:async()=> 'token',fetcher:(path,{signal})=>{paths.push(path);return new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));}});
 const controller=new AbortController(),pending=client.get(id,{signal:controller.signal});await new Promise(resolve=>setImmediate(resolve));controller.abort();
 await assert.rejects(pending,{name:'AbortError'});assert.deepEqual(paths,['/api/training/jobs/'+id]);
});

test('cancel keeps the server cancelling state until the server confirms a terminal state',async()=>{
 const client=createTrainingClient({getUserId:()=> 'alice',getToken:async()=> 'token',fetcher:async(path,options)=>{assert.equal(path,'/api/training/jobs/'+id+'/cancel');assert.equal(options.method,'POST');return Response.json({...job,status:'cancelling'});}});
 const response=await client.cancel(id);assert.equal(terminalJob(response),false);assert.equal(terminalJob({...response,status:'cancelled'}),true);
});

test('a response arriving after an account switch is not exposed to the new account',async()=>{
 let uid='alice';const client=createTrainingClient({getUserId:()=>uid,getToken:async()=> 'token',fetcher:async()=>({ok:true,status:200,json:async()=>{uid='bob';return job;}})});
 await assert.rejects(client.get(id),/account changed/);
});

test('job references reject another request or project and preserve immutable submission parameters',()=>{
 const previous={requestId:request,projectId:project,kind:'training',modelId:'qwen3-0-6b',preset:'quick',ownerUid:'alice'};
 const updated=jobReference(job,previous);assert.equal(updated.jobId,id);assert.equal(updated.modelId,previous.modelId);assert.equal(updated.preset,'quick');
 assert.throws(()=>jobReference({...job,project_id:id},previous),/mismatched job/);assert.throws(()=>jobReference({...job,request_id:id},previous),/mismatched job/);
});

test('artifact download is authenticated and does not treat an API error as a model file',async()=>{
 const client=createTrainingClient({getUserId:()=> 'alice',getToken:async()=> 'token',fetcher:async(path,options)=>{assert.equal(options.headers.Authorization,'Bearer token');return Response.json({error:{message:'Output is not ready.'}},{status:409});}});
 await assert.rejects(client.artifact(id,'adapter'),/Output is not ready/);
});
