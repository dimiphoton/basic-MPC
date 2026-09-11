const I18N = {
  fr: {
    nav: "Machine learning · Bâtiment · <a href='../slides/presentation-recruteur-fr.html'>slides</a> · <a href='https://github.com/dimiphoton/basic-MPC'>code</a>",
    title: "Peux-tu chauffer avant que la maison ait froid ?",
    lede: "Un mur, c’est un condensateur. Tourne l’inertie, vois le déphasage, puis affronte le thermostat.",
    rcTitle: "Le modèle RC, en une phrase",
    rcBody: "R dit à quelle vitesse la chaleur fuit vers l’extérieur. C dit combien les murs encaissent avant que l’air bouge. Le capteur ne voit que l’air — la masse est cachée.",
    rcHint: "R1C1 = un seul seau. R2C2 = air + murs. C’est pour ça qu’un thermostat « trop tard, trop fort » existe.",
    phaseTitle: "À quoi sert le déphasage",
    phaseBody: "Le soleil et le chauffage d’aujourd’hui n’arrivent au confort que plus tard. Plus les murs sont lourds, plus le retard grandit. On préchauffe en heures creuses parce que 7 h, c’est déjà trop tard.",
    massLabel: "Inertie des murs (× capacité C)",
    zTitle: "Vecteur Z des fits (24 h)",
    zBody: "Chaque flèche est Z(jω) / Z(0) du modèle identifié sur la maison. L’angle, c’est le retard.",
    bodeTitle: "Phase vs période",
    bodeBody: "Vers 24 h, la phase plonge : la journée thermique n’est pas synchrone de l’apport.",
    arenaTitle: "Arène : trois stratégies, même maison",
    arenaBody: "Clique pour superposer. Le score, c’est la facture HP/HC et les heures vraiment trop froides (> 0,1 °C).",
    delay: (h) => `Retard ≈ ${h.toFixed(1)} h entre le pic extérieur et l’air`,
    strat: {
      hysteresis: { t: "Hystérésis", p: "Le thermostat. Il attend." },
      preheat: { t: "Préchauffage 2 h", p: "Règle bête : allumer à 5 h." },
      mpc: { t: "MPC", p: "Modèle + facture + confort." },
    },
    bill: "facture",
    cold: "h trop froid",
    lang: "EN",
  },
  en: {
    nav: "Machine learning · Buildings · <a href='../slides/presentation-recruteur-en.html'>slides</a> · <a href='https://github.com/dimiphoton/basic-MPC'>code</a>",
    title: "Can you heat before the house feels cold?",
    lede: "A wall is a capacitor. Crank the mass, watch the lag, then take on the thermostat.",
    rcTitle: "The RC model, in one line",
    rcBody: "R is how fast heat leaks outdoors. C is how much the walls soak up before the air moves. The sensor only sees air — mass is hidden.",
    rcHint: "R1C1 = one bucket. R2C2 = air + walls. That is why a thermostat is always late, then too hard.",
    phaseTitle: "Why phase lag matters",
    phaseBody: "Today’s sun and heating show up later as comfort. Heavier walls, longer delay. We preheat on the cheap night rate because 7 a.m. is already too late.",
    massLabel: "Wall inertia (× capacitance C)",
    zTitle: "Fitted Z vector (24 h)",
    zBody: "Each arrow is Z(jω) / Z(0) of the model fitted on the house. The angle is the delay.",
    bodeTitle: "Phase vs period",
    bodeBody: "Around 24 h the phase drops: the thermal day is not in sync with the input.",
    arenaTitle: "Arena: three strategies, same house",
    arenaBody: "Click to overlay. Score = peak/off-peak bill and hours actually too cold (> 0.1 °C).",
    delay: (h) => `Lag ≈ ${h.toFixed(1)} h from outdoor peak to indoor air`,
    strat: {
      hysteresis: { t: "Hysteresis", p: "The thermostat. It waits." },
      preheat: { t: "2 h preheat", p: "Dumb rule: turn on at 5 a.m." },
      mpc: { t: "MPC", p: "Model + bill + comfort." },
    },
    bill: "bill",
    cold: "h too cold",
    lang: "FR",
  },
};

const PALETTE = {
  hysteresis: "#8a7e6e",
  preheat: "#8c4a32",
  mpc: "#3d6b6b",
  ink: "#2c2416",
  bg: "#f4efe6",
};

let DATA = null;
let LANG = "fr";
let ON = { hysteresis: true, preheat: true, mpc: true };

function t() {
  return I18N[LANG];
}

function applyCopy() {
  const c = t();
  document.getElementById("nav").innerHTML = c.nav;
  document.getElementById("title").textContent = c.title;
  document.getElementById("lede").textContent = c.lede;
  document.getElementById("rcTitle").textContent = c.rcTitle;
  document.getElementById("rcBody").textContent = c.rcBody;
  document.getElementById("rcHint").textContent = c.rcHint;
  document.getElementById("phaseTitle").textContent = c.phaseTitle;
  document.getElementById("phaseBody").textContent = c.phaseBody;
  document.getElementById("massLabel").textContent = c.massLabel;
  document.getElementById("zTitle").textContent = c.zTitle;
  document.getElementById("zBody").textContent = c.zBody;
  document.getElementById("bodeTitle").textContent = c.bodeTitle;
  document.getElementById("bodeBody").textContent = c.bodeBody;
  document.getElementById("arenaTitle").textContent = c.arenaTitle;
  document.getElementById("arenaBody").textContent = c.arenaBody;
  document.getElementById("langBtn").textContent = c.lang;
  renderStratBtns();
  renderScore();
}

function layout() {
  return {
    paper_bgcolor: PALETTE.bg,
    plot_bgcolor: "#fbf8f2",
    font: { color: PALETTE.ink, family: "Segoe UI, sans-serif" },
    margin: { t: 24, r: 20, b: 44, l: 48 },
    legend: { orientation: "h", y: 1.12 },
  };
}

function peakLagHours(hours, outdoor, indoor) {
  const iOut = outdoor.indexOf(Math.max(...outdoor));
  const iIn = indoor.indexOf(Math.max(...indoor));
  return Math.abs(hours[iIn] - hours[iOut]);
}

function runLab(cmScale) {
  const p = DATA.plant;
  const dt = p.dt_seconds;
  const n = Math.round((48 * 3600) / dt);
  const hours = [];
  const text = [];
  const ta = [];
  const tm = [];
  let xTa = 12;
  let xTm = 12;
  const ca = p.ca;
  const cm = p.cm * cmScale;
  for (let k = 0; k < n; k += 1) {
    const h = (k * dt) / 3600;
    const hod = h % 24;
    const tExt = 5 + 6 * Math.sin((2 * Math.PI * (hod - 9)) / 24);
    const dTa =
      (xTm - xTa) / (p.ram * ca) + (tExt - xTa) / (p.rae * ca);
    const dTm = (xTa - xTm) / (p.ram * cm);
    xTa += dt * dTa;
    xTm += dt * dTm;
    if (k % 2 === 0) {
      hours.push(h);
      text.push(tExt);
      ta.push(xTa);
      tm.push(xTm);
    }
  }
  return { hours, text, ta, tm, lag: peakLagHours(hours, text, ta) };
}

function drawLab() {
  const scale = Number(document.getElementById("mass").value);
  const sim = runLab(scale);
  document.getElementById("delayReadout").textContent = t().delay(sim.lag);
  Plotly.react(
    "phaseChart",
    [
      { x: sim.hours, y: sim.text, name: LANG === "fr" ? "extérieur" : "outdoor", line: { color: PALETTE.ink, width: 2 } },
      { x: sim.hours, y: sim.ta, name: LANG === "fr" ? "air" : "air", line: { color: PALETTE.mpc, width: 2.4 } },
      { x: sim.hours, y: sim.tm, name: LANG === "fr" ? "murs" : "walls", line: { color: PALETTE.preheat, width: 1.6, dash: "dot" } },
    ],
    { ...layout(), xaxis: { title: LANG === "fr" ? "heures" : "hours" }, yaxis: { title: "°C" } },
    { responsive: true, displaylogo: false },
  );
}

function drawZ() {
  const z = DATA.fitted_z_24h;
  const arrow = (key, color, name) => {
    const s = z[key];
    return [
      {
        x: [0, s.re],
        y: [0, s.im],
        mode: "lines+markers",
        name: `${name} (${s.delay_hours.toFixed(1)} h)`,
        line: { color, width: 3 },
        marker: { size: [1, 10], color },
      },
    ];
  };
  Plotly.react(
    "zChart",
    [...arrow("r1c1", PALETTE.hysteresis, "R1C1"), ...arrow("r2c2", PALETTE.mpc, "R2C2")],
    {
      ...layout(),
      xaxis: { title: "Re Z/Z(0)", zeroline: true, scaleanchor: "y" },
      yaxis: { title: "Im Z/Z(0)", zeroline: true },
    },
    { responsive: true, displaylogo: false },
  );
}

function drawBode() {
  const b = DATA.bode;
  Plotly.react(
    "bodeChart",
    [
      { x: b.period_hours, y: b.phase_r1_deg, name: "R1C1", line: { color: PALETTE.hysteresis } },
      { x: b.period_hours, y: b.phase_r2_deg, name: "R2C2", line: { color: PALETTE.mpc } },
    ],
    {
      ...layout(),
      xaxis: { title: LANG === "fr" ? "période (h)" : "period (h)", type: "log" },
      yaxis: { title: LANG === "fr" ? "phase (°)" : "phase (°)" },
      shapes: [{ type: "line", x0: 24, x1: 24, y0: 0, y1: 1, yref: "paper", line: { dash: "dot", color: PALETTE.ink } }],
    },
    { responsive: true, displaylogo: false },
  );
}

function renderStratBtns() {
  const box = document.getElementById("stratBtns");
  box.innerHTML = "";
  const names = Object.keys(DATA.arena.strategies);
  names.forEach((key) => {
    const meta = t().strat[key];
    const btn = document.createElement("button");
    btn.className = `strat${ON[key] ? " on" : ""}`;
    btn.type = "button";
    btn.innerHTML = `<h3>${meta.t}</h3><p>${meta.p}</p>`;
    btn.addEventListener("click", () => {
      ON[key] = !ON[key];
      drawArena();
      renderStratBtns();
      renderScore();
    });
    box.appendChild(btn);
  });
}

function renderScore() {
  const board = document.getElementById("scoreboard");
  board.innerHTML = "";
  Object.entries(DATA.arena.strategies).forEach(([key, strat]) => {
    if (!ON[key]) return;
    const m = strat.metrics;
    const el = document.createElement("div");
    el.className = "pill";
    el.innerHTML = `<strong>${t().strat[key].t}</strong> · ${m.bill_eur.toFixed(2)} € ${t().bill} · ${m.hours_under_conf.toFixed(1)} ${t().cold}`;
    board.appendChild(el);
  });
}

function drawArena() {
  const traces = [
    {
      x: DATA.arena.hours,
      y: DATA.arena.t_conf,
      name: "T_conf",
      line: { color: PALETTE.ink, width: 1.4, dash: "dash" },
    },
  ];
  Object.entries(DATA.arena.strategies).forEach(([key, strat]) => {
    if (!ON[key]) return;
    traces.push({
      x: strat.hours,
      y: strat.ta,
      name: t().strat[key].t,
      line: { color: PALETTE[key] || PALETTE.mpc, width: 2.2 },
    });
  });
  Plotly.react(
    "arenaChart",
    traces,
    {
      ...layout(),
      xaxis: { title: LANG === "fr" ? "heures depuis 18 h" : "hours from 6 p.m." },
      yaxis: { title: "°C" },
    },
    { responsive: true, displaylogo: false },
  );
}

async function main() {
  const res = await fetch("data.json");
  DATA = await res.json();
  if (!DATA.arena.strategies.mpc) ON.mpc = false;
  applyCopy();
  drawLab();
  drawZ();
  drawBode();
  drawArena();
  document.getElementById("mass").addEventListener("input", drawLab);
  document.getElementById("langBtn").addEventListener("click", () => {
    LANG = LANG === "fr" ? "en" : "fr";
    applyCopy();
    drawLab();
    drawZ();
    drawBode();
    drawArena();
  });
}

main().catch((err) => {
  document.getElementById("lede").textContent = `data.json manquant — python -m basic_mpc export-simulator (${err})`;
});
