const I18N = {
  fr: {
    nav: "Machine learning · Bâtiment · <a href='../slides/presentation-recruteur-fr.html'>slides</a> · <a href='https://github.com/dimiphoton/basic-MPC'>code</a>",
    title: "Un mur, c’est un condensateur",
    lede: "Le capteur ne voit que l’air. Tourne l’inertie : le pic intérieur retarde. Pour comparer des stratégies de chauffe, lance le labo Streamlit (météo tirée, maison déjà habitée).",
    rcTitle: "Le modèle RC, en une phrase",
    rcBody: "R dit à quelle vitesse la chaleur fuit vers l’extérieur. C dit combien les murs encaissent avant que l’air bouge. Le capteur ne voit que l’air — la masse est cachée.",
    rcHint: "R1C1 = un seul seau. R2C2 = air + murs. C’est pour ça qu’un thermostat « trop tard, trop fort » existe.",
    phaseTitle: "À quoi sert le déphasage",
    phaseBody: "Sans chauffage, l’air suit l’extérieur en retard. On jette 24 h de transitoire ; le retard est une corrélation croisée, pas le premier pic. On préchauffe en heures creuses parce que 7 h, c’est déjà trop tard.",
    massLabel: "Inertie des murs (× capacité C)",
    zTitle: "Vecteur Z des fits (24 h)",
    zBody: "Chaque flèche est Z(jω)/Z(0) du modèle identifié. L’angle, c’est le retard à 24 h — lecture DS, pas le labo de stratégies.",
    bodeTitle: "Phase vs période",
    bodeBody: "Vers 24 h, la phase plonge : la journée thermique n’est pas synchrone de l’apport.",
    labTitle: "Labo de stratégies : Streamlit",
    labBody: "Ici on n’enchaîne pas une arène figée. En local : météo au hasard, 2–7 jours, burn-in 24 h, une stratégie (hystérésis / préchauffage / MPC).",
    delay: (h) => `Retard ≈ ${h.toFixed(1)} h (corrélation T_ext → air, après 24 h)`,
    lang: "EN",
  },
  en: {
    nav: "Machine learning · Buildings · <a href='../slides/presentation-recruteur-en.html'>slides</a> · <a href='https://github.com/dimiphoton/basic-MPC'>code</a>",
    title: "A wall is a capacitor",
    lede: "The sensor only sees air. Crank the mass: indoor peaks lag. Strategy comparison lives in the Streamlit lab (drawn weather, a house already lived in).",
    rcTitle: "The RC model, in one line",
    rcBody: "R is how fast heat leaks outdoors. C is how much the walls soak up before the air moves. The sensor only sees air — mass is hidden.",
    rcHint: "R1C1 = one bucket. R2C2 = air + walls. That is why a thermostat is always late, then too hard.",
    phaseTitle: "Why phase lag matters",
    phaseBody: "With no heating, air follows outdoor with delay. We drop 24 h of transient; lag is a cross-correlation, not the first peak. We preheat on the night rate because 7 a.m. is already too late.",
    massLabel: "Wall inertia (× capacitance C)",
    zTitle: "Fitted Z vector (24 h)",
    zBody: "Each arrow is Z(jω)/Z(0) of the identified model. The angle is the 24 h delay — a DS readout, not the strategy lab.",
    bodeTitle: "Phase vs period",
    bodeBody: "Around 24 h the phase drops: the thermal day is not in sync with the input.",
    labTitle: "Strategy lab: Streamlit",
    labBody: "This page is not a frozen arena. Locally: random weather, 2–7 days, 24 h burn-in, one strategy (hysteresis / preheat / MPC).",
    delay: (h) => `Lag ≈ ${h.toFixed(1)} h (T_ext → air correlation, after 24 h)`,
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
  document.getElementById("labTitle").textContent = c.labTitle;
  document.getElementById("labBody").textContent = c.labBody;
  document.getElementById("langBtn").textContent = c.lang;
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

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i += 1) s += arr[i];
  return s / arr.length;
}

function lagHoursXcorr(outdoor, indoor, dtHours, discardHours) {
  const skip = Math.round(discardHours / dtHours);
  const x = outdoor.slice(skip);
  const y = indoor.slice(skip);
  if (x.length < 8) return 0;
  const mx = mean(x);
  const my = mean(y);
  const xc = x.map((v) => v - mx);
  const yc = y.map((v) => v - my);
  const maxLag = Math.round(18 / dtHours);
  const minOverlap = Math.max(8, Math.round(12 / dtHours));
  let bestLag = 0;
  let bestC = -Infinity;
  const n = xc.length;
  for (let lag = 0; lag <= maxLag; lag += 1) {
    if (n - lag < minOverlap) break;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < n - lag; i += 1) {
      const a = xc[i];
      const b = yc[i + lag];
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    const c = dot / (Math.sqrt(na * nb) + 1e-12);
    if (c > bestC) {
      bestC = c;
      bestLag = lag;
    }
  }
  return bestLag * dtHours;
}

function runLab(cmScale) {
  const p = DATA.plant;
  const dt = p.dt_seconds;
  const nDays = p.n_lab_days || 5;
  const discard = p.discard_hours || 24;
  const n = Math.round((nDays * 24 * 3600) / dt);
  const hours = [];
  const text = [];
  const ta = [];
  const tm = [];
  const tExt0 = 5 + 6 * Math.sin((2 * Math.PI * (0 - 9)) / 24);
  let xTa = tExt0;
  let xTm = tExt0;
  const ca = p.ca;
  const cm = p.cm * cmScale;
  for (let k = 0; k < n; k += 1) {
    const h = (k * dt) / 3600;
    const hod = h % 24;
    const tExt = 5 + 6 * Math.sin((2 * Math.PI * (hod - 9)) / 24);
    const dTa = (xTm - xTa) / (p.ram * ca) + (tExt - xTa) / (p.rae * ca);
    const dTm = (xTa - xTm) / (p.ram * cm);
    xTa += dt * dTa;
    xTm += dt * dTm;
    hours.push(h);
    text.push(tExt);
    ta.push(xTa);
    tm.push(xTm);
  }
  const dtHours = dt / 3600;
  const lag = lagHoursXcorr(text, ta, dtHours, discard);
  const stride = 4;
  return {
    hours: hours.filter((_, i) => i % stride === 0),
    text: text.filter((_, i) => i % stride === 0),
    ta: ta.filter((_, i) => i % stride === 0),
    tm: tm.filter((_, i) => i % stride === 0),
    lag,
    discard,
  };
}

function drawLab() {
  const scale = Number(document.getElementById("mass").value);
  const sim = runLab(scale);
  document.getElementById("delayReadout").textContent = t().delay(sim.lag);
  const shapes = [
    {
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: 0,
      x1: sim.discard,
      y0: 0,
      y1: 1,
      fillcolor: PALETTE.ink,
      opacity: 0.06,
      line: { width: 0 },
    },
  ];
  Plotly.react(
    "phaseChart",
    [
      { x: sim.hours, y: sim.text, name: LANG === "fr" ? "extérieur" : "outdoor", line: { color: PALETTE.ink, width: 2 } },
      { x: sim.hours, y: sim.ta, name: LANG === "fr" ? "air" : "air", line: { color: PALETTE.mpc, width: 2.4 } },
      { x: sim.hours, y: sim.tm, name: LANG === "fr" ? "murs" : "walls", line: { color: PALETTE.preheat, width: 1.6, dash: "dot" } },
    ],
    {
      ...layout(),
      xaxis: { title: LANG === "fr" ? "heures (bande = transitoire jeté)" : "hours (band = dropped transient)" },
      yaxis: { title: "°C" },
      shapes,
    },
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

async function main() {
  const res = await fetch("data.json");
  DATA = await res.json();
  applyCopy();
  drawLab();
  drawZ();
  drawBode();
  document.getElementById("mass").addEventListener("input", drawLab);
  document.getElementById("langBtn").addEventListener("click", () => {
    LANG = LANG === "fr" ? "en" : "fr";
    applyCopy();
    drawLab();
    drawZ();
    drawBode();
  });
}

main().catch((err) => {
  document.getElementById("lede").textContent = `data.json manquant — python -m basic_mpc export-simulator (${err})`;
});
