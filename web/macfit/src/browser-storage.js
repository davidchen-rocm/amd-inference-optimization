// Only replaceable interface preferences may use this session fallback.
// User-created models and other durable records must surface failed saves.
const sessionValues=new Map();

export function readPreference(key,fallback){
 if(sessionValues.has(key))return sessionValues.get(key)??fallback;
 try{return JSON.parse(localStorage.getItem(key))??fallback;}catch{return fallback;}
}

export function writePreference(key,value){
 try{
  localStorage.setItem(key,JSON.stringify(value));
  sessionValues.delete(key);
  return true;
 }catch{
  sessionValues.set(key,value);
  return false;
 }
}

export function removePreference(key){
 try{
  localStorage.removeItem(key);
  sessionValues.delete(key);
  return true;
 }catch{
  sessionValues.set(key,null);
  return false;
 }
}
