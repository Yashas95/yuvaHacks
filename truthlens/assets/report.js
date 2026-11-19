
document.querySelectorAll('.ev').forEach(card=>{
  card.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
    card.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
    card.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.dataset.p===t.dataset.t));
  });
});
document.querySelectorAll('.fchip').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.fchip').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('.ev').forEach(c=>c.style.display=(b.dataset.f==='all'||c.dataset.type===b.dataset.f)?'':'none');
});
function setMode(m){
  document.body.classList.toggle('simple',m==='simple');document.body.classList.toggle('expert',m==='expert');
  document.querySelectorAll('.mode button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  try{localStorage.setItem('tl-mode',m)}catch(e){}
}
(function(){let m='simple';try{m=new URLSearchParams(location.search).get('mode')||localStorage.getItem('tl-mode')||'simple'}catch(e){}setMode(m==='expert'?'expert':'simple')})();
document.querySelectorAll('.mode button,.tomode').forEach(b=>b.addEventListener('click',e=>{e.preventDefault();setMode(b.dataset.m);window.scrollTo(0,0)}));
function openFor(id){
  const el=document.getElementById(id);if(!el)return;
  if(el.closest('.x')&&document.body.classList.contains('simple'))setMode('expert');
  if(el.closest('.s')&&document.body.classList.contains('expert'))setMode('simple');
  const card=el.closest('.ev');
  if(card){card.style.display='';const p=el.closest('.panel');
    if(p&&!p.classList.contains('on')){card.querySelector('.tab[data-t="'+p.dataset.p+'"]').click()}}
  el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1600);
}
document.addEventListener('click',e=>{
  const a=e.target.closest('a[href^="#"]');if(!a)return;
  const id=a.getAttribute('href').slice(1);const el=document.getElementById(id);if(!el)return;
  e.preventDefault();openFor(id);el.scrollIntoView({behavior:'smooth',block:'start'});history.replaceState(null,'','#'+id);
});
function lb(img){const l=document.getElementById('lb');l.querySelector('img').src=img.src;l.querySelector('.cap').textContent=img.alt;l.classList.add('on')}
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lb').classList.remove('on')});
function cp(t,b){navigator.clipboard&&navigator.clipboard.writeText(t);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}
const io=new IntersectionObserver(es=>es.forEach(en=>{if(en.isIntersecting){
  document.querySelectorAll('.side nav a').forEach(a=>a.classList.toggle('cur',a.getAttribute('href')==='#'+en.target.id))}}),{rootMargin:'-30% 0px -60% 0px'});
document.querySelectorAll('main section[id]').forEach(s=>io.observe(s));
window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
