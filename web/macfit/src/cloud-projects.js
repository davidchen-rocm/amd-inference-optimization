import {assertOwner,assertRevision,encodeRecord,decodeRecord,modelKey,KINDS} from './project-records.js';

export class CloudProjects {
  constructor({getUser,database}){this.getUser=getUser;this.database=database;this.reset();}
  reset(){this.epoch=(this.epoch||0)+1;this.uid=null;this.loaded=false;this.loading=null;this.records=new Map(KINDS.map(k=>[k,[]]));}
  session(){const uid=this.getUser()?.uid;if(!uid)throw Error('Sign in to access your models.');if(this.uid!==uid){this.reset();this.uid=uid;}return {uid,epoch:this.epoch};}
  check({uid,epoch}){if(this.getUser()?.uid!==uid||this.epoch!==epoch)throw Error('Your account changed. Reopen your models before continuing.');}
  list(kind){const uid=this.getUser()?.uid;return uid&&uid===this.uid?this.records.get(kind)||[]:[];}
  async load({force=false}={}){
    const session=this.session();if(this.loading)return this.loading;if(this.loaded&&!force)return;
    const operation=(async()=>{
      const {db,sdk}=await this.database();this.check(session);
      const all=await sdk.getDocs(sdk.collection(db,'users',session.uid,'models'));this.check(session);
      const docs=all.docs,results=[];let next=0;
      // Bound concurrent payload reads; the catalog itself is never copied here.
      await Promise.all(Array.from({length:Math.min(4,docs.length)},async()=>{
        while(next<docs.length){const entry=docs[next++],manifest=entry.data();
          if(!KINDS.includes(manifest.kind))continue;
          if(entry.id!==modelKey(manifest.kind,manifest.recordId))throw Error('A saved project has an invalid document identity.');
          const parts=await sdk.getDocs(sdk.collection(db,'users',session.uid,'models',entry.id,'revisions',manifest.revision,'chunks'));
          this.check(session);const record=await decodeRecord(manifest,parts.docs.map(x=>x.data()),session.uid);results.push({kind:manifest.kind,record});
        }
      }));
      this.check(session);for(const kind of KINDS)this.records.set(kind,results.filter(x=>x.kind===kind).map(x=>x.record).sort((a,b)=>String(b.updatedAt).localeCompare(String(a.updatedAt))));this.loaded=true;
    })();this.loading=operation;
    try{await operation;}finally{if(this.loading===operation)this.loading=null;}
  }
  async save(kind,record){
    const session=this.session();assertOwner(record,session.uid);
    const prepared=await encodeRecord(kind,record,crypto.randomUUID());this.check(session);
    const {db,sdk}=await this.database();this.check(session);
    const ref=sdk.doc(db,'users',session.uid,'models',prepared.key);
    try{await sdk.runTransaction(db,async tx=>{
      const current=await tx.get(ref);this.check(session);assertRevision(current.exists()?current.data():null,record._cloudRevision);
      for(const chunk of prepared.chunks){
        const part=sdk.doc(db,'users',session.uid,'models',prepared.key,'revisions',prepared.manifest.revision,'chunks',String(chunk.part).padStart(2,'0'));
        tx.set(part,{...chunk,revision:prepared.manifest.revision});
      }
      tx.set(ref,{...prepared.manifest,updatedAt:sdk.serverTimestamp()});
    });}catch(error){this.loaded=false;throw error;}
    this.check(session);const saved={...prepared.record,_cloudRevision:prepared.manifest.revision,_ownerUid:session.uid};
    this.records.set(kind,[saved,...this.list(kind).filter(x=>x.id!==saved.id)]);return saved;
  }
}
