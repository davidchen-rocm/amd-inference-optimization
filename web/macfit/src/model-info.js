// Display specifications separately from repository popularity metrics.
export const parameterLabel=model=>{
 const value=model.parameter_count;
 if(!Number.isFinite(value)||value<=0)return 'Unverified';
 const approximate=model.field_evidence?.parameter_count?.approximate||model.field_evidence?.parameter_count?.basis==='architecture-config-calculation';
 return (approximate?'≈ ':'')+new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(value);
};
export const contextLabel=model=>{
 const value=model.context_length;
 if(!Number.isInteger(value)||value<=0)return model.context_variants?.length?'Varies by checkpoint':'Unverified';
 const label=model.field_evidence?.context_length?.declared_label||(value>=1024&&Number.isInteger(Math.log2(value))?`${value/1024}K`:new Intl.NumberFormat('en').format(value));
 return (model.field_evidence?.context_length?.approximate?'≈ ':'')+label;
};
export const licenseLabel=value=>({mit:'MIT','apache-2.0':'Apache 2.0','bsd-3-clause':'BSD 3-Clause','bsd-2-clause':'BSD 2-Clause','cc-by-4.0':'CC BY 4.0','cc-by-nc-4.0':'CC BY-NC 4.0'}[value]||(!value||value==='unknown'?'Unspecified':value));
export function fieldNote(model,field){
 const evidence=model.field_evidence?.[field];
 if(field==='context_length'&&model.context_variants?.length)return model.context_variants.map(v=>`${v.checkpoint}: ${v.declared_context}`).join(' · ');
 if(field==='parameter_count'&&model.parameter_count_unverified_reason)return model.parameter_count_unverified_reason;
 if(evidence?.basis==='publisher-tokenizer-limit')return 'Publisher tokenizer limit; this is not a verified architecture maximum.';
 if(evidence?.basis==='publisher-training-context')return 'Publisher training sequence length; this is not a runtime-independent maximum.';
 if(evidence?.basis==='architecture-config-calculation')return 'Calculated from the official architecture configuration; checkpoint tensors were not inspected.';
 if(evidence)return `${evidence.basis}: ${evidence.field}`;
 if(model[field])return model[field+'_basis']||'Previously sourced model specification.';
 return model.metadata_status?.[field]==='access-restricted'?'Publisher metadata requires access approval.':'Not verified in the available publisher metadata.';
}
