# MacFit training implementation contract

This extends the existing AMD inference repository with reusable LoRA/SFT code and a real MacFit training service. The service targets an MI300X 192GB host. Only signed-in Firebase users may submit website jobs.

For current usage, operational limits and validation evidence, read the
[training guide](README.md) and [deployment guide](../../deploy/macfit-training/README.md).

## Deployment

A dedicated unprivileged systemd service on the GPU host runs HTTP API bound to 127.0.0.1:8791 plus one supervised GPU worker. SQLite WAL job metadata, datasets and artifacts live under /srv/macfit-training/data (persistent host storage). Model cache is separate. OpenShift adds a small training bridge Service with an outbound SSH tunnel using a dedicated port-forward-only key. Frontend Nginx proxies /api/training/ to that bridge, while catalog API remains as deployed. The bridge injects a private X-Training-Gateway secret; GPU API requires it for every application request, plus Firebase user authentication for user jobs. No GPU HTTP port is opened to the internet.

## API

GET /api/training/capabilities is public behind the gateway. Return available, auth:{required:true,provider:'firebase'}, models:[{id,repo_id,revision,name}], supported tasks/presets, limits, generation:true, training:true, evaluation:true, cancel:true. Generation uses an allowed model, with its resolved immutable revision. Availability reflects a working GPU and worker.

POST /api/training/jobs accepts {request_id:UUID,project_id:UUID,kind:'generation'|'training',input:{...}}. Return 202 with the job. Idempotency is scoped by verified Firebase UID + request_id; mismatched input returns409. Owner UID is derived only from a verified ID token. Any owner-supplied field is rejected/ignored as identity. Bound request body to 3MiB. Enforce bounded dataset/model/hyperparameters, one active job per owner and a global queue limit. No arbitrary paths, model repositories, commands or Python from browser input.

GET /api/training/jobs/:id; GET /api/training/jobs?project_id=... or request_id=...; POST /api/training/jobs/:id/cancel; GET /api/training/jobs/:id/artifacts/:artifact_id. Every operation owner-scoped; use 404 for another owner's job.

Job: {id,request_id,project_id,kind,status:'queued'|'running'|'cancelling'|'succeeded'|'failed'|'cancelled',stage,progress:{completed,total,unit}|null,created_at,updated_at,input_hash,base_model:{repo_id,revision},error:null|{code,message,retryable},result:null|object,artifacts:[{id,type,name,size_bytes,sha256}]}. Never manufacture progress or model improvements. Store immutable input snapshot before returning. Cancellation is supervised, finishes only after process cleanup. Restart marks interrupted jobs failed/retryable; it must not claim success or silently duplicate training.

Generation input: {model_id,task:'answers'|'writing'|'format'|'classify'|'custom',goal,language:'en'|'zh'|'multi',source,purpose:'preview'|'dataset',seeds:[{id,question,answer,system?,approved:true}],target_count}. Preview requires goal and optional source, returns3 diverse proposed examples. Dataset requires3 approved corrected seeds, count12/24/48; preserve corrected seeds, generate real additional samples. All new samples are unapproved. Result:{examples:[{id,question,answer,system?,origin:'gpu-generated',approved:false}],provenance:{...}}. Generate with actual LM inference, validate JSON and dedupe, fail recoverably rather than pretending templates are AI output.

Training input: {model_id,task,goal,language,source,training:[{id,messages:[{role:'system'|'user'|'assistant',content}],approved:true}],evaluation:[{id,question,expected,system?,approved:true}],preset:'quick'|'standard'}. Require >=3 train rows and >=3 disjoint held-out test questions, <=500 train rows, max input content bound. Supervised data is user-reviewed, assistant-only causal loss. Both the public API and reusable CLI restrict model IDs to the immutable registry; arbitrary repositories and continued training from an earlier adapter are not supported. Presets: LoRA bf16, rank16/alpha32/dropout0.05, attention+MLP targets, max_seq_length<=2048, max_epochs<=3, steps<=300, walltime<=3600s. One GPU run at a time.

Training result: {method:'lora_sft',training:{steps,loss,trainable_parameters,total_parameters,...},evaluation:{samples:[{id,question,expected,before,after,...}],metrics:{...}},base_model:{repo_id,revision},provenance:{...}}. Evaluate base and trained adapter on identical held-out questions with actual generation, and measured loss metrics. Do not claim better quality from training loss alone. Artifacts include adapter bundle (not a standalone Mac executable model), resolved config, evaluation JSON and manifest with hashes and usage notes. Optional merged model export supported by CLI; website may omit it initially and explain the adapter artifact.

## Worker interface

API launches current Python: `python -m macfit_training.worker --job-dir ABSOLUTE_PRIVATE_JOB_DIR --kind generation|training`. Directory input.json contains the API job input (input field only); kind supplied by CLI. Worker writes events.jsonl (bounded events {stage,progress?,message?}), result.json and artifacts/ files. API polls events file and verifies produced artifacts (regular files, size/hash, no symlink or traversal), registers them after success only. Worker stdout/stderr are private bounded logs and never expose auth tokens. GPU lock uses existing amd_inference_opt.resource_lock; process supervision reuses process_group where possible.

Core public helpers for service: `macfit_training.config.validate_job_input(kind,input)->dict` and `macfit_training.config.capabilities()->dict`; sanitized input returned by helper has base_model registry resolution and presets pinned. Registry uses known immutable Hugging Face revisions, no trust_remote_code.

## Frontend

Add getAccessToken(forceRefresh=false) to account module, never persist/log token. Same-origin client attaches Authorization header and handles401 once with refresh. Submit stable request_id and save project job reference before request; ambiguous response recovers through request_id. Draft owns job reference and last known state; reload resumes polling; leaving page stops polling without cancelling GPU task. Edit yields new immutable request. Explicit cancel waits for terminal backend state. Replace all simulated generation/training paths. Only show returned before/after outputs and actual artifacts. Availability/auth errors are clear, with no demo-success fallback. Online hosting is unavailable initially; delivery presents LoRA adapter download and truthful use requirements.

## Ownership boundaries

API agent: src/macfit_training/service/** and tests/training/test_service*.py; communicate config interface changes.
Framework/ML agent: src/macfit_training/{config,data,trainer,generation,evaluation,artifacts,worker,cli}.py, examples/training/**, tests/training/test_core*.py and tests/training/test_worker*.py. Keep torch imports lazy so CPU service tests don't require it.
Frontend agent: existing frontend src/lesson-view.js, dataset-flow.js, teach-domain.js (job-ref validation), account.js (getAccessToken), new training-client.js; frontend behavioral tests. Root reviews deployments, dependency pins, package metadata, docs, integration and real GPU tests.
