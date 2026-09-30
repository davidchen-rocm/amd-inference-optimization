import {getAccount,initializeAccount,getCloudDatabase} from './account.js';
import {CloudProjects} from './cloud-projects.js';
import {readDrafts,saveDraft} from './teach-domain.js';
import {loadModels,saveModel} from './personal-domain.js';
import {migrationId} from './project-records.js';

export function createWorkspaceService({
  account={getAccount,initializeAccount,getCloudDatabase},projects=null,
  getStorage=()=>globalThis.localStorage,events=globalThis.document
}={}){
  const cloud=projects||new CloudProjects({getUser:()=>account.getAccount().user,database:account.getCloudDatabase});
  events?.addEventListener('macfit:account-will-change',()=>cloud.reset());
  function storage(){try{return getStorage();}catch{throw Error('Browser storage is unavailable. Allow storage for this website to save local projects.');}}
  function checkIdentity(){
    const state=account.getAccount();
    // A remembered account is not a guest, even if its session cannot be restored.
    if(state.phase==='error'&&(state.user||state.knownUid))throw Error('Your account could not be reached. Check your connection and try again. Your projects have not been moved or changed.');
    if(!state.user&&state.knownUid)throw Error('Your account is still being restored. Please retry before opening or saving projects.');
    return state.user;
  }
  async function resolveIdentity({retry=false}={}){
    const state=account.getAccount();
    if(state.phase==='loading'||retry||(state.phase==='error'&&(state.user||state.knownUid)))await account.initializeAccount();
    return checkIdentity();
  }
  function checkDraftOwner(draft,user,kind){
    if(draft._ownerUid&&draft._ownerUid!==user?.uid)throw Error('Your account changed. Reopen this project in its owner’s account before saving.');
    if(!user&&draft._cloudRevision)throw Error('Sign in to the project’s account before saving changes.');
    let localOnly=draft._localOnly;
    if(user&&!localOnly&&draft.id&&!draft._ownerUid&&!draft._cloudRevision){
      try{localOnly=(kind==='teaching'?readDrafts(storage()):loadModels(storage())).some(record=>record.id===draft.id);}catch{}
    }
    if(user&&localOnly)throw Error('This project belongs to this browser. Use Import local projects in My models to add a copy to your account.');
  }
  async function loadWorkspace(options={}){
    const user=await resolveIdentity(options);
    if(user)await cloud.load(options);
  }
  const workspaceLocation=()=>account.getAccount().user?'your account':'this browser';
  function teachingProjects(){
    const user=checkIdentity();
    return user?cloud.list('teaching'):readDrafts(storage()).map(record=>({...record,_localOnly:true}));
  }
  function localSetups(){
    const user=checkIdentity();
    return user?cloud.list('configuration'):loadModels(storage()).map(record=>({...record,_localOnly:true}));
  }
  async function saveTeachingProject(draft,id){
    const user=await resolveIdentity();checkDraftOwner(draft,user,'teaching');
    if(!user){const local={...draft};delete local._localOnly;return {...saveDraft(storage(),local,id),_localOnly:true};}
    return cloud.save('teaching',{...draft,id:draft.id||id,updatedAt:new Date().toISOString()});
  }
  async function saveLocalSetup(draft,selection,config,context){
    const user=await resolveIdentity();checkDraftOwner(draft,user,'configuration');
    if(!user){const local={...draft};delete local._localOnly;return {...saveModel(local,selection,config,context,{storage:storage()}),_localOnly:true};}
    await cloud.load();
    // Do not continue a save if the session changed while the cloud loaded.
    if(account.getAccount().user?.uid!==user.uid)throw Error('Your account changed. Reopen this project before saving.');
    const id=draft.id||crypto.randomUUID();
    let value=JSON.stringify(cloud.list('configuration'));const inMemory={getItem:()=>value,setItem:(_,next)=>{value=next;}};
    const record=saveModel(draft,selection,config,context,{storage:inMemory,id});return cloud.save('configuration',record);
  }
  function localImportCount(){
    let count=0;try{count+=readDrafts(storage()).length;}catch{}try{count+=loadModels(storage()).length;}catch{}return count;
  }
  async function importLocalProjects(){
    const user=checkIdentity();if(!user)throw Error('Sign in before importing local projects.');
    const uid=user.uid;await cloud.load();
    const groups=[['teaching',readDrafts(storage())],['configuration',loadModels(storage())]];let count=0;
    for(const [kind,records] of groups)for(const record of records){
      if(account.getAccount().user?.uid!==uid)throw Error('Your account changed. Import stopped; local copies are untouched.');
      const id=migrationId(kind,record.id);
      if(cloud.list(kind).some(x=>x.id===id))continue;
      const item={...record,id,updatedAt:record.updatedAt||new Date().toISOString()};delete item._cloudRevision;delete item._ownerUid;delete item._localOnly;
      await cloud.save(kind,item);count++;
    }
    return count;
  }
  function stashGuestDraft(draft){
    if(account.getAccount().user||account.getAccount().knownUid||draft._ownerUid||(!draft.goal&&!draft.examples?.length))return;
    const local={...draft};delete local._localOnly;
    saveDraft(storage(),local,draft.id||crypto.randomUUID());
  }
  return {loadWorkspace,workspaceLocation,teachingProjects,localSetups,saveTeachingProject,saveLocalSetup,localImportCount,importLocalProjects,stashGuestDraft};
}
export const {loadWorkspace,workspaceLocation,teachingProjects,localSetups,saveTeachingProject,saveLocalSetup,localImportCount,importLocalProjects,stashGuestDraft}=createWorkspaceService();
