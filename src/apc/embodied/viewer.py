"""Write an offline, rotatable 3D trajectory replay (no CDN, server or GPU required)."""
# ruff: noqa: E501
# Embedded HTML/JavaScript retains its own line layout.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HTML = r'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>APC / Embodied 3D</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e4ebf4;font:15px system-ui}
header{padding:20px 28px;border-bottom:1px solid #334355;display:flex;gap:24px;align-items:center}
h1{font-size:21px;margin:0}small{color:#a2b3c5}main{max-width:1280px;margin:auto;padding:20px}
canvas{width:100%;height:65vh;min-height:320px;background:#16212f;border:1px solid #334355;
 border-radius:12px;touch-action:none}nav{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:16px 0}
button,select{padding:9px 12px;background:#25364a;color:#e4ebf4;border:1px solid #53677f;border-radius:6px}
input{flex:1;min-width:160px}#status{padding:12px;background:#1b2a3c;border-radius:6px;min-height:46px}
.note{color:#a2b3c5;line-height:1.7}b{color:#b8d8fe}
</style>
<header><h1>APC / Embodied 3D</h1><small>Continuous control · Primitive composition · Reuse</small></header>
<main><nav><label>Episode <select id="episode"></select></label><button id="play">Play</button>
<label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
<input id="seek" type="range" min="0" value="0"><span id="clock"></span></nav>
<canvas id="world"></canvas><p id="status"></p><p class="note">ドラッグ: 視点回転 / ホイール: 拡大縮小。
球が身体、立方体が障害物、番号付きの点が指定された経由地点です。軌跡は記録された実際の状態を再生します。
<b>Reference dynamics / privileged state / explicit waypoints</b> — 実機検証・自律的経路発見の証明ではありません。</p></main>
<script>
"use strict";
const data=__DATA__;
const canvas=document.getElementById('world'), ctx=canvas.getContext('2d');
const selector=document.getElementById('episode'), seek=document.getElementById('seek');
const play=document.getElementById('play'), speed=document.getElementById('speed');
let current=0,index=0,running=false,last=0,accumulated=0,yaw=-0.85,pitch=0.52,zoom=1;
const center=[0,0,2.5];
data.episodes.forEach((e,i)=>{let o=document.createElement('option');o.value=i;
 o.textContent=e.label+' / '+(e.result.success?'SUCCESS':e.result.reason);selector.add(o)});
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
 let a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;
 let h=Math.cos(pitch)*z-Math.sin(pitch)*b,d=Math.sin(pitch)*z+Math.cos(pitch)*b;
 let scale=Math.min(canvas.width,canvas.height)*0.72*zoom/(14+d);
 return [canvas.width/2+a*scale,canvas.height/2-h*scale,scale,d];}
function line(points,color,width=1){if(points.length<2)return;ctx.beginPath();points.forEach((p,i)=>{
 let v=project(p);if(i)ctx.lineTo(v[0],v[1]);else ctx.moveTo(v[0],v[1]);});ctx.strokeStyle=color;
 ctx.lineWidth=width*devicePixelRatio;ctx.stroke();}
function sphere(p,r,color,label){const v=project(p);ctx.beginPath();ctx.arc(v[0],v[1],Math.max(3,r*v[2]),0,2*Math.PI);
 ctx.fillStyle=color;ctx.fill();if(label){ctx.fillStyle='#e4ebf4';ctx.font=(13*devicePixelRatio)+'px system-ui';
 ctx.fillText(label,v[0]+9*devicePixelRatio,v[1]-8*devicePixelRatio);}}
function box(b){let a=b.lower,c=b.upper,pts=[];for(let i=0;i<8;i++)pts.push([i&1?c[0]:a[0],i&2?c[1]:a[1],i&4?c[2]:a[2]]);
 const faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]];
 faces.sort((a,b)=>b.reduce((s,k)=>s+project(pts[k])[3],0)-a.reduce((s,k)=>s+project(pts[k])[3],0));
 faces.forEach(face=>{ctx.beginPath();face.forEach((k,i)=>{let v=project(pts[k]);if(i)ctx.lineTo(v[0],v[1]);else ctx.moveTo(v[0],v[1]);});
 ctx.closePath();ctx.fillStyle='#48596dcc';ctx.fill();ctx.strokeStyle='#8a9eb5';ctx.stroke();});}
function draw(){const e=data.episodes[current];if(!e)return;const tr=e.result.trace;if(!tr.length)return;
 const state=tr[index];let rect=canvas.getBoundingClientRect(),w=Math.round(rect.width*devicePixelRatio),h=Math.round(rect.height*devicePixelRatio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}ctx.clearRect(0,0,w,h);
 const lo=data.world.lower,hi=data.world.upper;
 for(let x=Math.ceil(lo[0]);x<=hi[0];x++)line([[x,lo[1],lo[2]],[x,hi[1],lo[2]]],'#2a3d51');
 for(let y=Math.ceil(lo[1]);y<=hi[1];y++)line([[lo[0],y,lo[2]],[hi[0],y,lo[2]]],'#2a3d51');
 line([[0,0,0],[2,0,0]],'#d17e81',2);line([[0,0,0],[0,2,0]],'#80b799',2);line([[0,0,0],[0,0,2]],'#84aee4',2);
 e.result.task.obstacles.forEach(box);
 line(tr.slice(0,index+1).map(s=>s.position),'#85c9ef',2);
 e.result.task.goals.forEach((g,i)=>sphere(g,0.13,'#e6c774',String(i+1)));
 sphere(e.result.task.start,0.09,'#8498af','start');sphere(state.position,data.world.radius,'#b5e1fa','body');
 const q=state.orientation,xdir=[1-2*(q[2]*q[2]+q[3]*q[3]),2*(q[1]*q[2]+q[3]*q[0]),2*(q[1]*q[3]-q[2]*q[0])];
 line([state.position,state.position.map((p,i)=>p+0.5*xdir[i])],'#f09582',2);
 document.getElementById('status').textContent='Primitive: '+(state.option||'—')+' | position (m): '+
 state.position.map(v=>v.toFixed(2)).join(', ')+' | outcome: '+(e.result.success?'success':e.result.reason);
 document.getElementById('clock').textContent=state.t.toFixed(2)+' / '+tr[tr.length-1].t.toFixed(2)+' s';seek.value=index;}
function choose(){current=Number(selector.value);index=0;accumulated=0;seek.max=Math.max(0,data.episodes[current].result.trace.length-1);draw();}
selector.onchange=choose;seek.oninput=()=>{index=Number(seek.value);accumulated=0;draw();};
play.onclick=()=>{running=!running;if(index>=Number(seek.max))index=0;play.textContent=running?'Pause':'Play';};
let pointer=null;canvas.onpointerdown=e=>{pointer=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!pointer)return;yaw+=(e.clientX-pointer[0])*0.008;pitch=Math.max(-0.1,Math.min(1.3,pitch+(e.clientY-pointer[1])*0.005));pointer=[e.clientX,e.clientY];draw();};
canvas.onpointerup=()=>pointer=null;canvas.onpointercancel=()=>pointer=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.5,Math.min(3,zoom*Math.exp(-e.deltaY*0.001)));draw();},{passive:false});
window.onresize=draw;
function tick(now){if(running&&last){accumulated+=(now-last)/1000*Number(speed.value);const tr=data.episodes[current].result.trace;
 while(index<tr.length-1&&accumulated>=tr[index+1].t-tr[index].t){accumulated-=tr[index+1].t-tr[index].t;index++;}
 if(index===tr.length-1){running=false;play.textContent='Play';}draw();}last=now;requestAnimationFrame(tick);}
if(data.episodes.length){choose();requestAnimationFrame(tick);}
</script></html>'''


def write_replay(path: Path, world: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    data = json.dumps({"world": world, "episodes": episodes}, allow_nan=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(HTML.replace("__DATA__", data))
