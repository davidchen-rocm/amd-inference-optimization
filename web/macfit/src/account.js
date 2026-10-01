import {firebaseConfig,FIREBASE_SDK_VERSION} from './firebase-config.js';

const ACCOUNT_HINT='macfit.account-owner';
export function accountError(error){
  const code=error?.code||'';
  if(/failed to fetch dynamically imported module|error loading dynamically imported module|importing a module script failed|^load failed$/i.test(error?.message||''))return 'Sign-in could not be loaded. Check your connection and try again. Your local projects are still available.';
  if(/popup-closed|cancelled-popup/.test(code))return 'Sign-in was canceled. Your draft is still available.';
  if(/popup-blocked/.test(code))return 'Allow pop-ups for this site, then try Google sign-in again.';
  if(/unauthorized-domain|configuration-not-found|operation-not-allowed/.test(code))return 'Sign-in is not available on this website yet. Please try again later.';
  if(/network-request-failed|unavailable|deadline/.test(code))return 'The account service could not be reached. Check your connection and try again.';
  if(/permission-denied/.test(code))return 'Your saved account projects could not be opened. Please try again later.';
  if(code)return 'The account service is unavailable. Please try again.';
  return error?.message||'The account service is unavailable. Please try again.';
}
function withTimeout(promise,ms,message){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error(message)),ms);Promise.resolve(promise).then(v=>{clearTimeout(timer);resolve(v);},e=>{clearTimeout(timer);reject(e);});});}

// Dependencies are injectable so session recovery can be tested without real accounts.
export function createAccountService({
  loadModule=name=>import(`https://www.gstatic.com/firebasejs/${FIREBASE_SDK_VERSION}/firebase-${name}.js`),
  getStorage=()=>globalThis.localStorage,events=globalThis.document,timeoutMs=12000
}={}){
  let knownUid=null;
  try{knownUid=getStorage()?.getItem(ACCOUNT_HINT)||null;}catch{}
  let state={phase:'loading',user:null,knownUid,error:'',busy:false};
  let auth=null,authSDK=null,app=null,initializing=null,unsubscribe=null,generation=0,database=null,databaseLoading=null,actionPending=false;
  const listeners=new Set();
  const getAccount=()=>({...state});
  const subscribeAccount=listener=>{listeners.add(listener);return()=>listeners.delete(listener);};
  const emit=()=>{for(const fn of listeners)fn(getAccount());};
  const stopObserver=()=>{unsubscribe?.();unsubscribe=null;};
  const dispatch=(type,detail)=>events?.dispatchEvent(new CustomEvent(type,{detail}));
  function remember(uid){
    try{const storage=getStorage();uid?storage?.setItem(ACCOUNT_HINT,uid):storage?.removeItem(ACCOUNT_HINT);}catch{}
  }
  function initializeAccount(){
    if(initializing)return initializing;
    if(state.phase==='ready')return Promise.resolve();
    const recovering=state.phase==='error',current=++generation;stopObserver();
    state={...state,phase:'loading',error:''};emit();
    const operation=Promise.resolve().then(async()=>{
      try{
        const [core,sdk]=await withTimeout(Promise.all([loadModule('app'),loadModule('auth')]),timeoutMs,'Could not load sign-in. You can still open projects saved in this browser.');
        if(current!==generation)return;
        authSDK=sdk;app=core.getApps().find(a=>a.name==='macfit-account')||core.initializeApp(firebaseConfig,'macfit-account');auth=sdk.getAuth(app);
        await withTimeout(new Promise((resolve,reject)=>{
          let first=true;
          unsubscribe=sdk.onAuthStateChanged(auth,user=>{
            if(current!==generation)return;
            const previousUid=state.user?.uid||null,nextUid=user?.uid||null;
            const recovered=recovering&&first,changed=previousUid!==nextUid||recovered;
            if(changed)dispatch('macfit:account-will-change',{previousUid,nextUid,initial:first,recovered});
            remember(nextUid);
            state={phase:'ready',busy:actionPending,error:'',knownUid:nextUid,user:user?{uid:user.uid,displayName:user.displayName||'Your account',email:user.email||'',photoURL:user.photoURL||''}:null};
            emit();if(changed)dispatch('macfit:account-changed',{previousUid,nextUid,initial:first,recovered});
            if(first){first=false;resolve();}
          },error=>{
            if(current!==generation)return;
            if(!first){++generation;stopObserver();state={...state,phase:'error',error:accountError(error),busy:false};emit();}
            reject(error);
          });
        }),timeoutMs,'Your sign-in session could not be restored. Check your connection and try again.');
      }catch(error){
        if(current!==generation)return;
        // Invalidates callbacks queued by a timed-out observer before a later retry.
        ++generation;stopObserver();state={...state,phase:'error',error:accountError(error),busy:false};emit();
      }
    });
    initializing=operation;
    operation.finally(()=>{if(initializing===operation)initializing=null;});
    return operation;
  }
  async function signInGoogle(){
    if(actionPending)return;
    actionPending=true;
    try{
      if(!auth||state.phase!=='ready')await initializeAccount();
      if(!auth||state.phase!=='ready')return;
      state={...state,busy:true,error:''};emit();
      const provider=new authSDK.GoogleAuthProvider();provider.setCustomParameters({prompt:'select_account'});
      await authSDK.signInWithPopup(auth,provider);
    }catch(error){state={...state,error:accountError(error)};}
    finally{actionPending=false;state={...state,busy:false};emit();}
  }
  async function signOutAccount(){
    if(!auth||actionPending)return;
    actionPending=true;state={...state,busy:true,error:''};emit();
    try{await authSDK.signOut(auth);}catch(error){state={...state,error:accountError(error)};}
    finally{actionPending=false;state={...state,busy:false};emit();}
  }
  async function getCloudDatabase(){
    await initializeAccount();
    if(state.phase==='error')throw Error(state.error);
    if(!state.user||!auth?.currentUser)throw Error('Sign in to access your models.');
    if(!database){
      databaseLoading ||= withTimeout(loadModule('firestore-lite'),timeoutMs,'Your saved account projects could not be loaded. Check your connection and retry.').then(sdk=>({sdk,db:sdk.getFirestore(app)}));
      try{database=await databaseLoading;}finally{databaseLoading=null;}
    }
    return database;
  }
  async function getAccessToken(forceRefresh=false){
    await initializeAccount();
    const user=auth?.currentUser,uid=state.user?.uid;
    if(state.phase!=='ready'||!uid||user?.uid!==uid){const error=Error('Sign in with Google before using GPU generation or training.');error.code='sign_in_required';throw error;}
    const token=await withTimeout(user.getIdToken(forceRefresh),timeoutMs,'Your sign-in token could not be refreshed. Please try again.');
    if(state.phase!=='ready'||state.user?.uid!==uid||auth?.currentUser?.uid!==uid)throw Error('Your account changed. Reopen this project before continuing.');
    return token;
  }
  return {getAccount,subscribeAccount,initializeAccount,signInGoogle,signOutAccount,getCloudDatabase,getAccessToken};
}
export const {getAccount,subscribeAccount,initializeAccount,signInGoogle,signOutAccount,getCloudDatabase,getAccessToken}=createAccountService();
