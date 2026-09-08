const fs = require("fs");

const read = (file) => (fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim() : "");
const gib = (bytes) => `${(Number(bytes) / 2 ** 30).toFixed(2)} GiB`;

try {
  process.kill(Number(process.env.STATE_sampler_pid));
} catch {}
const cgroup = process.env.STATE_cgroup;
const peak = read(`${cgroup}/memory.peak`) || read(process.env.STATE_peak_file) || "0";
const max = read(`${cgroup}/memory.max`);
const oomKills = (read(`${cgroup}/memory.events`).match(/^oom_kill (\d+)/m) || [, "0"])[1];
const line = `Peak job memory: ${gib(peak)}, cap ${max === "max" || !max ? "none" : gib(max)}, OOM kills ${oomKills}`;
console.log(line);
fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${line}\n`);
