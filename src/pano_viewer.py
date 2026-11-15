"""PLAN_V5 milestone 4: the Inside view as a panorama viewer (no depth mesh, so nothing can tear).

The v4 inside view was a depth mesh seen from (near) the panorama centre: any camera translation exposes holes and
stretches thin furniture into ribbons. The panorama itself is a perfect picture of the room from ONE point, so the
default Inside view now keeps the camera at that point and only lets the user turn and zoom:

    one self-contained HTML page (no CDN, no three.js, works offline): a WebGL2 full-screen quad whose fragment
    shader casts, for every screen pixel, the pinhole ray of a level camera
        d_view = ((x - w/2) / f, (y - h/2) / f, 1)            (X right, Y down, Z forward; f from the horizontal FOV)
        d_B    = R_yaw R_pitch d_view                           (frame B: gravity-aligned, wall-aligned room frame)
        d_cam  = G d_B,  G = R^T Q                              (room box rotations, src/room_box.fit_box)
    and samples the 4096 x 2048 equirectangular panorama at (lon, lat) of d_cam with textureLod at the mip level of
    the current zoom (no seam line from derivative jumps at lon = +-180°; mipmaps against aliasing).
    The ray formula is the same as src/design_shots.render_view, so at the hero yaw and FOV the viewer shows exactly the
    hero design shot (checked with screenshots in scripts/eval_v5_viewer.py).
    Colours: the JPEG's pixel values are written to the canvas unchanged (no tone mapping, no exposure, no lighting).
    Controls: drag = turn (pitch limited to +-85°), wheel / pinch / +- keys = zoom (horizontal FOV 30-110°),
    arrow keys = turn, R / double click = reset to the hero direction.
    For scripted checks: window.viewerReady, window.setView(yawDeg, pitchDeg, hfovDeg), window.getView().
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

HFOV0 = 80.0          # opening horizontal FOV (the hero shot's FOV)

_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#111;overflow:hidden;font-family:system-ui,sans-serif}
canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}
#hint{position:absolute;left:50%;bottom:10px;transform:translateX(-50%);color:#fff;background:rgba(0,0,0,.45);
padding:4px 10px;border-radius:6px;font-size:12px;pointer-events:none;transition:opacity 1s}
#err{position:absolute;inset:0;display:none;color:#ddd;align-items:center;justify-content:center;text-align:center;padding:20px}
</style></head><body>
<canvas id="c"></canvas><div id="hint">__HINT__</div><div id="err">This browser has no WebGL2, so the 360&deg; view cannot be shown.</div>
<script>
(function(){
const YAW0=__YAW0__, PITCH0=0, HFOV0=__HFOV0__, G=__G__;
const cv=document.getElementById('c'); const gl=cv.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
if(!gl){document.getElementById('err').style.display='flex';return;}
const vs=`#version 300 es
in vec2 p; void main(){gl_Position=vec4(p,0.,1.);}`;
const fs=`#version 300 es
precision highp float; uniform sampler2D tex; uniform mat3 G; uniform float yaw,pitch,f,lod; uniform vec2 res; out vec4 o;
void main(){
 vec2 q=gl_FragCoord.xy; vec3 d=vec3((q.x-0.5*res.x)/f,-(q.y-0.5*res.y)/f,1.0);
 float cp=cos(pitch),sp=sin(pitch); d=vec3(d.x,cp*d.y-sp*d.z,sp*d.y+cp*d.z);
 float cy=cos(yaw),sy=sin(yaw); d=vec3(cy*d.x+sy*d.z,d.y,-sy*d.x+cy*d.z);
 d=normalize(G*d);
 float lon=atan(d.x,d.z), lat=asin(clamp(-d.y,-1.,1.));
 vec2 uv=vec2((lon+3.14159265)/6.2831853,(1.5707963-lat)/3.14159265);
 o=vec4(textureLod(tex,uv,lod).rgb,1.0);
}`;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);
 if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(x);return x;}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(pr);gl.useProgram(pr);
const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
const lp=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,2,gl.FLOAT,false,0,0);
const U=n=>gl.getUniformLocation(pr,n);
gl.uniformMatrix3fv(U('G'),false,new Float32Array(G));
let yaw=YAW0*Math.PI/180,pitch=PITCH0,hfov=HFOV0,W=4096,ready=false;
const tex=gl.createTexture();
function draw(){
 const dpr=window.devicePixelRatio||1, w=Math.round(cv.clientWidth*dpr), h=Math.round(cv.clientHeight*dpr);
 if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
 gl.viewport(0,0,w,h);
 const f=0.5*w/Math.tan(hfov*Math.PI/360);
 gl.uniform1f(U('yaw'),yaw);gl.uniform1f(U('pitch'),pitch);gl.uniform1f(U('f'),f);
 gl.uniform1f(U('lod'),Math.max(0,Math.log2(W/(2*Math.PI*f))));gl.uniform2f(U('res'),w,h);
 if(ready)gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
}
let pending=false;function req(){if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;draw();});}}
const img=new Image();
img.onload=function(){
 gl.bindTexture(gl.TEXTURE_2D,tex);gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);gl.generateMipmap(gl.TEXTURE_2D);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
 W=img.width;ready=true;draw();window.viewerReady=true;};
img.src="__SRC__";
const clampP=()=>{pitch=Math.max(-85*Math.PI/180,Math.min(85*Math.PI/180,pitch));};
const pts=new Map();let pinch0=0,fov0=hfov;
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);pts.set(e.pointerId,[e.clientX,e.clientY]);
 if(pts.size===2){const a=[...pts.values()];pinch0=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);fov0=hfov;}});
cv.addEventListener('pointermove',e=>{if(!pts.has(e.pointerId))return;const [x0,y0]=pts.get(e.pointerId);
 pts.set(e.pointerId,[e.clientX,e.clientY]);
 if(pts.size===1){const k=(hfov*Math.PI/180)/cv.clientWidth;yaw-=(e.clientX-x0)*k;pitch+=(e.clientY-y0)*k;clampP();}
 else if(pts.size===2&&pinch0>0){const a=[...pts.values()];const d=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);
  hfov=Math.max(30,Math.min(110,fov0*pinch0/Math.max(d,1)));}
 req();document.getElementById('hint').style.opacity=0;});
const up=e=>{pts.delete(e.pointerId);pinch0=0;};cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',up);
cv.addEventListener('wheel',e=>{e.preventDefault();hfov=Math.max(30,Math.min(110,hfov*Math.exp(e.deltaY*0.001)));req();},{passive:false});
const reset=()=>{yaw=YAW0*Math.PI/180;pitch=PITCH0;hfov=HFOV0;req();};
cv.addEventListener('dblclick',reset);
window.addEventListener('keydown',e=>{const s=5*Math.PI/180;
 if(e.key==='ArrowLeft')yaw-=s;else if(e.key==='ArrowRight')yaw+=s;else if(e.key==='ArrowUp')pitch+=s;
 else if(e.key==='ArrowDown')pitch-=s;else if(e.key==='+'||e.key==='=')hfov=Math.max(30,hfov/1.1);
 else if(e.key==='-')hfov=Math.min(110,hfov*1.1);else if(e.key==='r'||e.key==='R')reset();else return;clampP();req();});
window.addEventListener('resize',req);
window.setView=(y,p,f)=>{yaw=y*Math.PI/180;pitch=p*Math.PI/180;if(f)hfov=f;clampP();draw();};
window.getView=()=>({yaw:yaw*180/Math.PI,pitch:pitch*180/Math.PI,hfov:hfov});
})();
</script></body></html>"""


def view_matrix(box: dict | None) -> np.ndarray:
    """G = R^T Q: frame B (room, gravity- and wall-aligned) -> panorama camera frame. Identity without a box."""
    if box is None:
        return np.eye(3)
    return np.asarray(box["R"]).T @ np.asarray(box["Q"])


def jpeg_b64(pano: np.ndarray, quality: int = 92, max_w: int = 4096) -> str:
    if pano.shape[1] > max_w:
        pano = cv2.resize(pano, (max_w, max_w // 2), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(pano, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def viewer_html(pano: np.ndarray | None = None, box: dict | None = None, yaw_deg: float = 0.0, hfov_deg: float = HFOV0,
                title: str = "360° room", src: str | None = None,
                hint: str = "Drag to look around, scroll to zoom, double-click to reset.") -> str:
    """Self-contained viewer page. pano: RGB uint8 (embedded as JPEG), or src: an image URL instead.
    yaw_deg: opening direction in frame B (the hero shot's yaw)."""
    G = view_matrix(box)
    g = ",".join(f"{v:.6f}" for v in G.T.reshape(-1))          # column-major for uniformMatrix3fv
    return (_HTML.replace("__TITLE__", title).replace("__HINT__", hint).replace("__YAW0__", f"{yaw_deg:.3f}")
            .replace("__HFOV0__", f"{hfov_deg:.2f}").replace("__G__", f"[{g}]")
            .replace("__SRC__", src if src is not None else jpeg_b64(pano)))


def save_viewer(path, pano: np.ndarray, box: dict | None, yaw_deg: float, hfov_deg: float = HFOV0,
                title: str = "360° room") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(viewer_html(pano, box, yaw_deg, hfov_deg, title), encoding="utf-8")
    return str(path)
