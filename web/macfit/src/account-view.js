import {getAccount} from './account.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const person='<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5.5 20v-1.5a6.5 6.5 0 0 1 13 0V20"/></svg>';
const bookmark='<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M6 3h12v18l-6-4-6 4Z"/></svg>';
export const googleButton=()=>{const s=getAccount();return `<button class="button google-signin" data-action="google-signin" ${s.busy||s.phase==='loading'?'disabled':''}><span class="google-g" aria-hidden="true">G</span>${s.busy?'Opening Google…':s.phase==='loading'?'Loading sign-in…':'Continue with Google'}</button>`;};
const avatar=user=>user?.photoURL&&/^https:\/\/[^/]*googleusercontent\.com\//.test(user.photoURL)?`<img src="${esc(user.photoURL)}" alt="" referrerpolicy="no-referrer">`:user?.displayName?`<span>${esc(user.displayName.slice(0,1).toUpperCase())}</span>`:person;
export function accountControl(){
  const s=getAccount(),u=s.user;
  return `<div class="header-actions account-menu"><button class="account-trigger" id="account-trigger" data-action="account" aria-label="${u?'Your account: '+esc(u.displayName):'Your account'}" aria-expanded="false" aria-controls="account-panel">${avatar(u)}</button><section class="account-panel" id="account-panel" aria-label="Your account" hidden><div class="account-identity"><span class="account-avatar">${avatar(u)}</span><div><strong>${esc(u?.displayName||'Your account')}</strong><span>${esc(u?.email||'Keep your models with you')}</span></div></div><a class="account-menu-link" href="/my-models">${bookmark}<span><strong>My models</strong><small>Teaching projects & saved setups</small></span><span aria-hidden="true">→</span></a>${u?`<button class="account-signout" data-action="google-signout" ${s.busy?'disabled':''}>${s.busy?'Signing out…':'Sign out'}</button><p class="account-preview-note">Signed in with Google. Saved projects are private to your account.</p>`:`<div class="account-login">${googleButton()}</div><p class="account-preview-note">Sign in to save and open your models on any device. Existing local drafts stay local until you import them.</p>`}${s.error?`<p class="account-error" role="alert">${esc(s.error)}</p>`:''}</section></div>`;
}
export function accountGate(){
  const s=getAccount();return `<div class="cloud-signin-card card"><span class="cloud-account-symbol">${person}</span><span class="eyebrow">YOUR OWN LITTLE MODEL LIBRARY</span><h1>Your models.<br/>Wherever you work.</h1><p>Sign in to save your teaching projects, examples, and local setups to your account.</p>${googleButton()}${s.error?`<p class="account-error" role="alert">${esc(s.error)}</p>`:''}<p class="cloud-account-note">Already made a draft? You can import projects saved in this browser after signing in.</p><a class="inline-link" href="/teach">Keep exploring without an account →</a></div>`;
}
export function refreshAccountUI(){
  const el=document.querySelector('.account-menu');if(el){const open=el.querySelector('#account-panel')?.hidden===false;el.outerHTML=accountControl();if(open){document.querySelector('#account-panel').hidden=false;document.querySelector('#account-trigger').setAttribute('aria-expanded','true');}}
  const gate=document.querySelector('#account-gate');if(gate)gate.innerHTML=accountGate();
}
