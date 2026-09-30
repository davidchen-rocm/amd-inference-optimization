// Generates a local Python runner; no code or examples are executed by the website.
export function runner(spec){
 const encoded=btoa(Array.from(new TextEncoder().encode(JSON.stringify(spec)),b=>String.fromCharCode(b)).join(''));
 return `#!/usr/bin/env python3
# MacFit local runner. Requires Python 3 and the Hugging Face CLI (hf).
# GGUF: install llama-server with the backend for your GPU/CPU.
# MLX: install mlx-lm on Apple Silicon. Model license terms apply.
import base64, json, pathlib, shutil, socket, subprocess, sys, time, urllib.request, urllib.error
CONFIG = json.loads(base64.b64decode('${encoded}'))
root = pathlib.Path(__file__).resolve().parent
folder = root / CONFIG['folder']
if not shutil.which('hf'):
    sys.exit('Install the Hugging Face CLI first: https://huggingface.co/docs/huggingface_hub/guides/cli')
if CONFIG['format'] == 'GGUF' and not shutil.which('llama-server'):
    sys.exit('Install llama.cpp first: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md')
if CONFIG['format'] == 'MLX':
    try:
        from mlx_lm import load, generate
    except ImportError:
        sys.exit('Install mlx-lm first: python3 -m pip install mlx-lm')
print('Downloading the pinned model version. Files are reused on later runs.')
files = CONFIG['files'] + (['*.json', 'tokenizer*', '*.model', '*.tiktoken', '*.jinja'] if CONFIG['format'] == 'MLX' else [])
subprocess.run(['hf','download',CONFIG['repo'],'--revision',CONFIG['revision'],'--local-dir',str(folder),'--include',*files],check=True)
messages = CONFIG['messages']
process = None
try:
    if CONFIG['format'] == 'GGUF':
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0))
            port = probe.getsockname()[1]
        logfile = open(root / 'macfit-runtime.log','w')
        first = sorted(CONFIG['files'])[0]
        command = ['llama-server','-m',str(folder/first),'-c',str(CONFIG['context']),'-np','1','-ctk','f16','-ctv','f16','-ngl','0' if CONFIG['cpu'] else '999','--host','127.0.0.1','--port',str(port)]
        process = subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT)
        endpoint = 'http://127.0.0.1:'+str(port)
        for _ in range(300):
            if process.poll() is not None: sys.exit('Runtime stopped. Read macfit-runtime.log for details.')
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=1) as response:
                    if response.status == 200: break
            except (OSError,urllib.error.URLError): pass
            time.sleep(1)
        else: sys.exit('Model loading timed out. Read macfit-runtime.log.')
    else:
        model, tokenizer = load(str(folder))
    print(CONFIG['name']+' is ready. Type /exit to stop. Each question starts a fresh conversation.')
    while True:
        try: question = input('You > ').strip()
        except EOFError: break
        if question == '/exit': break
        if not question: continue
        current = [*messages,{'role':'user','content':question}]
        if CONFIG['format'] == 'GGUF':
            body = json.dumps({'messages':current,'max_tokens':512,'stream':False}).encode()
            req = urllib.request.Request(endpoint+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=300) as response:
                    answer=json.load(response)['choices'][0]['message']['content']
                print('AI > '+answer)
            except urllib.error.HTTPError as error:
                print('Request failed (HTTP '+str(error.code)+'). Try a shorter question; check the runtime log.')
        else:
            prompt = tokenizer.apply_chat_template(current,tokenize=False,add_generation_prompt=True)
            if len(tokenizer.encode(prompt))+512>CONFIG['context']:
                print('Question and examples exceed your context budget. Use a shorter question.');continue
            print('AI > '+generate(model,tokenizer,prompt=prompt,max_tokens=512,verbose=False))
except KeyboardInterrupt:
    print('Stopped.')
finally:
    if process is not None:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill();process.wait()
`;
}
