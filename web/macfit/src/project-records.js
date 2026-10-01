// Portable serialization and state guards. No authentication tokens or SDK state.
export const RECORD_VERSION = 1;
export const MAX_RECORD_BYTES = 3_000_000;
export const CHUNK_BYTES = 120_000;
export const KINDS = ['teaching','configuration'];
export function modelKey(kind,id){
  if(!KINDS.includes(kind)||typeof id!=='string'||!id.match(/^[A-Za-z0-9_-]{1,128}$/))throw Error('This project has an invalid identity.');
  return kind+'--'+id;
}
export function cleanRecord(record){
  if(!record||typeof record!=='object'||Array.isArray(record))throw Error('The project data is invalid.');
  const clean=JSON.parse(JSON.stringify(record));
  delete clean._cloudRevision;delete clean._ownerUid;
  if(typeof clean.id!=='string'||typeof clean.name!=='string'||clean.name.length>80)throw Error('The project name or ID is invalid.');
  if(!Array.isArray(clean.examples))throw Error('The project examples are invalid.');
  return clean;
}
function toBase64(bytes){let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(binary);}
function fromBase64(value){const binary=atob(value),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);return bytes;}
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
export async function encodeRecord(kind,record,revision){
  const clean=cleanRecord(record),key=modelKey(kind,clean.id),bytes=new TextEncoder().encode(JSON.stringify(clean));
  if(bytes.length>MAX_RECORD_BYTES)throw Error('This project is over the 3 MB cloud-save limit. Reduce its examples or export a local copy.');
  const chunks=[];for(let i=0;i<bytes.length;i+=CHUNK_BYTES)chunks.push({part:chunks.length,content:toBase64(bytes.subarray(i,i+CHUNK_BYTES))});
  return {key,record:clean,chunks,manifest:{schemaVersion:RECORD_VERSION,kind,recordId:clean.id,name:clean.name,baseModelId:String(clean.modelId||''),revision,chunkCount:chunks.length,byteLength:bytes.length,sha256:await digest(bytes)}};
}
export async function decodeRecord(manifest,chunks,uid){
  if(manifest?.schemaVersion!==RECORD_VERSION||!KINDS.includes(manifest.kind)||!Number.isInteger(manifest.chunkCount)||manifest.chunkCount<1||manifest.chunkCount>25||manifest.byteLength>MAX_RECORD_BYTES||chunks.length!==manifest.chunkCount)throw Error('A saved project is incomplete. Refresh your models or keep your local backup.');
  const sorted=[...chunks].sort((a,b)=>a.part-b.part),parts=sorted.map((chunk,i)=>{if(chunk.part!==i||typeof chunk.content!=='string'||chunk.content.length>160000)throw Error('A saved project has invalid data parts.');return fromBase64(chunk.content);});
  const length=parts.reduce((sum,b)=>sum+b.length,0);if(length!==manifest.byteLength)throw Error('A saved project has an invalid size.');
  const bytes=new Uint8Array(length);let cursor=0;for(const part of parts){bytes.set(part,cursor);cursor+=part.length;}
  if(await digest(bytes)!==manifest.sha256)throw Error('A saved project failed its integrity check.');
  const record=cleanRecord(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
  if(record.id!==manifest.recordId)throw Error('A saved project has a mismatched identity.');
  return {...record,_cloudRevision:manifest.revision,_ownerUid:uid};
}
export function assertOwner(record,uid){
  if(!uid)throw Error('Sign in to save this project to your account.');
  if(record._ownerUid&&record._ownerUid!==uid)throw Error('Your account changed. Reopen this project in its owner’s account.');
}
export function assertRevision(current,expected){
  if((current?.revision||null)!==(expected||null))throw Error('This project changed on another device. Reopen it before saving; your edits have not been overwritten.');
}
export function migrationId(kind,id){return modelKey(kind,id).replace('--','_');}
