// Does this p5.brush sketch crash?
//
// A cheap fake browser. It gives the sketch a pretend p5.js and a pretend
// p5.brush (every function p5.brush 1.1.4 exports, all doing nothing), runs
// the code, and calls setup(). Nothing is drawn. We only care whether it throws.
//
// Reads one JSON object per line on stdin:   {"id": "...", "code": "..."}
// Writes one JSON object per line on stdout: {"id": "...", "ok": true, "error": null}
//
//   echo '{"id":"a","code":"let brush; function setup(){ brush = createBrush(); }"}' | node check.js

const vm = require("vm");
const readline = require("readline");

// Everything p5.brush 1.1.4 exports. Anything else, like brush.wash(), crashes.
const BRUSH_METHODS = `add addField arc beginShape beginStroke bleed box circle clip colorCache endShape
  endStroke erase field fill fillAnimatedMode fillTexture flowLine gravity hatch hatchArray instance line
  listFields load noClip noErase noField noFill noGravity noHatch noStroke pick plot polygon pop preload push
  reBlend reDraw rect refreshField remove rotate scale scaleBrushes seed segment set setHatch spline stroke
  strokeWeight vertex`.split(/\s+/);

// p5 functions that just need to exist. Drawing does nothing.
const P5_NOOPS = `createCanvas resizeCanvas background clear translate rotate scale push pop noLoop loop
  redraw frameRate pixelDensity angleMode colorMode randomSeed noiseSeed noiseDetail print
  ellipse rect line point triangle quad arc circle square vertex beginShape endShape curveVertex
  bezierVertex fill noFill stroke noStroke strokeWeight text textSize image blendMode`.split(/\s+/);

function fakeColor() {
  return { levels: [0, 0, 0, 255], setAlpha() {}, setRed() {}, setGreen() {}, setBlue() {}, toString: () => "#000000" };
}

function makeWorld() {
  const world = {
    width: 512, height: 512, windowWidth: 512, windowHeight: 512, frameCount: 0,
    WEBGL: "webgl", P2D: "p2d", RGB: "rgb", HSB: "hsb", HSL: "hsl", DEGREES: "degrees", RADIANS: "radians",
    CENTER: "center", CORNER: "corner", CLOSE: "close", BLEND: "blend", MULTIPLY: "multiply",
    PI: Math.PI, TWO_PI: 2 * Math.PI, HALF_PI: Math.PI / 2, QUARTER_PI: Math.PI / 4, TAU: 2 * Math.PI,
    random(a, b) {
      if (Array.isArray(a)) return a[Math.floor(Math.random() * a.length)];
      if (a === undefined) return Math.random();
      if (b === undefined) return Math.random() * a;
      return a + Math.random() * (b - a);
    },
    noise: () => Math.random(),
    map: (v, a, b, c, d) => c + ((v - a) / (b - a)) * (d - c),
    lerp: (a, b, t) => a + (b - a) * t,
    constrain: (v, lo, hi) => Math.min(Math.max(v, lo), hi),
    dist: (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1),
    radians: (d) => (d * Math.PI) / 180,
    degrees: (r) => (r * 180) / Math.PI,
    sq: (x) => x * x,
    int: (x) => Math.trunc(x),
    float: (x) => parseFloat(x),
    str: (x) => String(x),
    shuffle: (arr) => [...arr].sort(() => Math.random() - 0.5),
    millis: () => 0,
    color: fakeColor,
    lerpColor: fakeColor,
    red: () => 0, green: () => 0, blue: () => 0, alpha: () => 255,
    hue: () => 0, saturation: () => 0, brightness: () => 0,
    createVector: (x = 0, y = 0) => ({ x, y, add() { return this; }, mult() { return this; } }),
    p5: {}, // so `new p5.Brush()` fails the way it does in a real browser
    brush: {},
    console,
  };
  for (const name of ["sin", "cos", "tan", "atan", "atan2", "sqrt", "pow", "abs", "floor", "ceil", "round", "min", "max", "exp", "log"]) {
    world[name] = Math[name];
  }
  for (const name of P5_NOOPS) world[name] = () => {};
  for (const name of BRUSH_METHODS) world.brush[name] = () => {};
  return world;
}

function check(code) {
  const world = vm.createContext(makeWorld());
  try {
    vm.runInContext(code, world, { timeout: 2000 });
    vm.runInContext(`
      if (typeof setup === "function") setup();
      if (typeof draw === "function") draw();
    `, world, { timeout: 2000 });
    return null;
  } catch (e) {
    return `${e && e.name}: ${e && e.message}`;
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  if (!line.trim()) return;
  const { id, code } = JSON.parse(line);
  const error = check(code);
  process.stdout.write(JSON.stringify({ id, ok: error === null, error }) + "\n");
});
