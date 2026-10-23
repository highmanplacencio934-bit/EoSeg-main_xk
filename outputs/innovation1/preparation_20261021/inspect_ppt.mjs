import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/xiakun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root=String.raw`\\sshfs.r\sk@anhuiuniversity.icu!21103\Disk\HDD\Sk\EoSeg-main_Synapse_acdc\EoSeg-main_synapse`;
const out=root+String.raw`\outputs\innovation1\preparation_20261021`;
const p=await PresentationFile.importPptx(await FileBlob.load(root+String.raw`\ASEE算法改进方案_v2.pptx`));
const snap=await p.inspect({kind:'slide,textbox,shape,image,notes',maxChars:150000});
await fs.writeFile(out+'\\ppt_inspect.ndjson',snap.ndjson);
for(const n of [5,8,10,11,13]) {
 const blob=await p.export({slide:p.slides.items[n-1],format:'png',scale:1});
 await fs.writeFile(out+`\\slide-${n}.png`,new Uint8Array(await blob.arrayBuffer()));
}
console.log('Imported 14 slides; rendered method diagrams.');
