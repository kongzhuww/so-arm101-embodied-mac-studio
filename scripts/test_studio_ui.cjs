// Browser-free DOM tests. All fetch calls are mocks; no camera/serial/motion.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = JSON.parse(fs.readFileSync(0,'utf8')).html;
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(code);
const elements = new Map();
for (const m of html.matchAll(/id="([^"]+)"/g)) {
  elements.set(m[1], {value:'',textContent:'',innerText:'',disabled:false,
    checked:false,style:{},className:'',options:[],open:false,attrs:{},
    getAttribute(key){return this.attrs[key];},setAttribute(key,value){this.attrs[key]=value;},removeAttribute(key){delete this.attrs[key];},
    replaceChildren(){this.options=[];},add(option){this.options.push(option);}});
}
assert(!elements.has('gripper-opening'));
assert(!elements.has('version-name'));
const stored={p1:'p1_standby',p2:'p2_grasp',p3:'p3_lift',p4:'p4_release'};
const basePoints={p1_standby:[2000,2000,2000,2000,2000,2000],p2_grasp:[2200,2200,2200,2200,2200,2200],
 p3_lift:[2300,2300,2300,2300,2300,2300],p4_release:[2400,2400,2400,2400,2400,2400]};
const versions=[1,2,3].map(i=>({id:String(i),name:'版本'+i,revision:1,episodes:0,
 history_episodes:0,open_gripper:2850,updated_at:'stamp'+i,complete:i===1,waypoints:i===1?basePoints:{}}));
let state={teleop_active:false,recording:false,training:false,motion_active:false,
 camera_enabled:true,camera_active:true,camera_released:false,standby_running:false,
 total_episodes:0,status_msg:'就绪',versions,active_version:versions[0],
 workbench_mode:'teach',act_running:false,act_status:'未运行',selected_act_model:'act_new30',
 act_models:[{id:'act_new30',label:'新模型 · 30条',path:'/TEST/new/final'},
             {id:'act_old10',label:'旧模型 · 10条',path:'/TEST/old/final'}],
 waypoints:Object.fromEntries(Object.entries(stored).map(([key,value])=>[key,basePoints[value]]))};
const requests=[],payloads=[];
const parameterIds=['point-copy-source','edit-point',...Array.from({length:6},(_,i)=>'joint-'+(i+1)),
 'copy-p1','copy-p2','copy-p3','copy-p4'];
function refreshState(){
 state.waypoints=Object.fromEntries(Object.entries(stored).map(([key,value])=>[key,state.active_version.waypoints[value]||null]));
}
const context=vm.createContext({
 document:{getElementById(id){assert(elements.has(id),id);return elements.get(id);},
  querySelectorAll(selector){
   if(selector==='.wp-btn')return [1,2,3,4].map(i=>elements.get('wp-'+i));
   if(selector==='[data-version-edit]')return [elements.get('version-select')];
   if(selector==='[data-parameter-edit]')return parameterIds.map(id=>elements.get(id));
   if(selector==='[data-teach-panel]')return ['versions-card','waypoints-card','teleop-card','dataset-card'].map(id=>elements.get(id));
   return [];
  }},
 window:{addEventListener(){}},setInterval(){},
 Option:function(text,value){this.text=text;this.value=value;},
 confirm(){return true;},alert(){},encodeURIComponent,
 fetch:async(url,options)=>{
  requests.push(url);
  if(url==='/api/standby')return {json:async()=>({ok:true,source:'TEST version 2',target:[2000,2000,2000,2000,2000,2000],context:'test-teach-2'})};
  if(url.startsWith('/api/standby/start?')){state.standby_running=true;state.motion_active=true;}
  if(url==='/api/standby/stop'){state.standby_running=false;state.motion_active=false;}
  if(url.startsWith('/api/camera?enabled=')){
   state.camera_enabled=url.endsWith('=true');
   state.camera_active=state.camera_enabled;state.camera_released=!state.camera_enabled;
  }
  if(url.startsWith('/api/workbench/mode?mode='))state.workbench_mode=url.split('mode=')[1];
  if(url.startsWith('/api/act/select?ident='))state.selected_act_model=url.split('ident=')[1];
  if(url.startsWith('/api/act/start?ident=')){
   assert.equal(state.workbench_mode,'act');
   assert.equal(url.split('ident=')[1],state.selected_act_model);
   state.act_running=true;state.motion_active=true;state.act_status='TEST RUNNING';
  }
  if(url==='/api/act/stop'){
   state.act_running=false;state.motion_active=false;state.act_status='TEST HELD';
  }
  if(url.startsWith('/api/versions/select')){
   state.active_version=versions[1];refreshState();
  }
  if(url==='/api/waypoints/edit'||url==='/api/waypoints/copy'){
   const data=JSON.parse(options.body);payloads.push({url,data});
   assert.equal(options.headers['Content-Type'],'application/json');
   assert.equal(data.target_version_id,state.active_version.id);
   assert.equal(data.expected_updated_at,state.active_version.updated_at);
   if(url.endsWith('edit'))state.active_version.waypoints[stored[data.point]]=data.values;
   else{
    const source=versions.find(v=>v.id===data.source_version_id);
    for(const point of data.points)state.active_version.waypoints[stored[point]]=[...source.waypoints[stored[point]]];
   }
   state.active_version.updated_at+='-saved';refreshState();
  }
  return {json:async()=>url==='/api/state'?state:{ok:true,msg:'ok'}};
 }
});
(async()=>{
 vm.runInContext(code,context);
 await vm.runInContext('syncState()',context);
 assert.equal(elements.get('btn-camera').innerText,'关闭摄像头');
 await vm.runInContext('toggleCamera()',context);
 assert.equal(elements.get('btn-camera').innerText,'开启摄像头');
 assert.equal(elements.get('live-feed').getAttribute('src'),undefined);
 assert.equal(elements.get('camera-placeholder').style.display,'');
 await vm.runInContext('toggleCamera()',context);
 assert.equal(elements.get('live-feed').getAttribute('src'),'/video_feed');
 assert.equal(elements.get('version-select').options.length,3);
 assert.equal(elements.get('version-select').value,'1');
 assert.equal(elements.get('wp-1-val').innerText,'已保存');
 assert.match(elements.get('wp-1-val').title,/2000/);
 assert.equal(elements.get('version-select').options[0].text,'版本1');
 await vm.runInContext('createVersion(false)',context);
 assert(requests.includes('/api/versions/create?copy_current=false'));
 assert(!requests.some(url=>url.includes('name=')));
 elements.get('point-tools').open=true;
 vm.runInContext('loadPointEditor()',context);
 assert.equal(elements.get('joint-1').value,'2000');
 elements.get('joint-1').value='2010';
 vm.runInContext('parameterDirty=true',context);
 await vm.runInContext('syncState()',context);
 assert.equal(elements.get('joint-1').value,'2010','poll must not overwrite unsaved edits');
 await vm.runInContext('saveParameterEdit()',context);
 assert.equal(payloads.at(-1).data.values[0],2010);
 const before=payloads.length;
 elements.get('joint-1').value='1.5';
 await vm.runInContext('saveParameterEdit()',context);
 assert.equal(payloads.length,before,'invalid decimal must not be submitted');
 await vm.runInContext("switchVersion('2')",context);
 assert.equal(elements.get('version-select').value,'2');
 assert.equal(elements.get('wp-1-val').innerText,'未录制');
 assert.equal(elements.get('point-tools').open,false,'switch closes stale target editor');
 elements.get('point-copy-source').value='1';
 vm.runInContext('updateCopyAvailability()',context);
 for(const point of ['p1','p2','p3','p4'])elements.get('copy-'+point).checked=['p1','p4'].includes(point);
 await vm.runInContext('copySelectedPoints()',context);
 assert.deepEqual(payloads.at(-1).data.points,['p1','p4']);
 assert.equal(state.waypoints.p1[0],2010);
 assert.equal(state.waypoints.p2,null);
 assert.equal(state.waypoints.p3,null);
 assert.equal(state.waypoints.p4[0],2400);
 state.motion_active=true;
 await vm.runInContext('syncState()',context);
 assert.equal(elements.get('version-select').disabled,true);
 assert.equal(elements.get('joint-1').disabled,true);
 assert.equal(elements.get('copy-p1').disabled,true);
 assert.equal(elements.get('btn-auto-record').disabled,true);
 state.motion_active=false;
 await vm.runInContext("changeWorkbenchMode('act')",context);
 assert.equal(elements.get('versions-card').style.display,'none');
 assert.equal(elements.get('act-panel').style.display,'');
 assert.equal(elements.get('btn-auto-record').style.display,'none');
 assert.equal(elements.get('act-model-select').options.length,2);
 await vm.runInContext("chooseActModel('act_old10')",context);
 assert.equal(elements.get('act-model-select').value,'act_old10');
 assert.match(elements.get('run-context').textContent,/旧模型/);
 assert(!requests.some(url=>url.startsWith('/api/act/start')),'selecting does not start');
 await vm.runInContext('runOnce()',context);
 assert(requests.includes('/api/act/start?ident=act_old10'));
 assert.equal(elements.get('act-stop').disabled,false);
 assert.equal(elements.get('mode-teach').disabled,true);
 assert.equal(state.active_version.id,'2');
 await vm.runInContext("versionRequest('/api/act/stop')",context);
 assert.equal(elements.get('act-stop').disabled,true);
 await vm.runInContext("changeWorkbenchMode('teach')",context);
 assert.equal(elements.get('versions-card').style.display,'');
 assert.equal(elements.get('act-panel').style.display,'none');
 assert.match(elements.get('btn-auto-test').innerText,/示教脚本/);
 await vm.runInContext('returnToStandby()',context);
 assert(requests.includes('/api/standby/start?context=test-teach-2'));
 assert.equal(elements.get('btn-standby').disabled,true);
 assert.equal(elements.get('btn-standby-stop').disabled,false);
 assert.equal(requests.filter(url=>url.startsWith('/api/act/start')).length,1,'standby never starts ACT');
 await vm.runInContext("versionRequest('/api/standby/stop')",context);
 assert.equal(elements.get('btn-standby-stop').disabled,true);
 assert(!requests.some(url=>url.includes('auto_playback')||url.includes('waypoint/save')));
 console.log('PASS: UI parses; point editing/copy; teach vs ACT visibility; selected model dispatch; stop; no real hardware');
})().catch(e=>{console.error(e);process.exit(1);});
