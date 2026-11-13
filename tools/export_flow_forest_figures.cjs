'use strict';
// Pure deterministic figure export. No model, network or training is invoked.
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const M=require(path.join(root,'docs/project/flow-forest-math.js'));
const V=require(path.join(root,'docs/project/flow-forest-views.js'));
const out=path.join(root,'paper/figures');fs.mkdirSync(out,{recursive:true});
for(const tab of M.TABS)fs.writeFileSync(path.join(out,`flow-forest-${tab}.svg`),V.renderSvg(tab,M.parseState('')));
fs.writeFileSync(path.join(out,'flow-forest-reward.svg'),V.rewardSvg(1));
console.log('Exported eight exact reference SVG figures; zero model or optimizer execution.');
