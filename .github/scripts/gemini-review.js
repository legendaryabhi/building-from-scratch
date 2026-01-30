const axios = require("axios");
const fetch = require("node-fetch");
const { execSync } = require("child_process");

// ---------- CONFIG ----------
const SCORE_TO_MERGE = 8;
const MODEL = "gemini-1.5-flash";
const LINK_TIMEOUT_MS = 8000;
// ----------------------------

const prNumber = process.env.GITHUB_REF.split("/")[2];

// Collect Gemini keys
const GEMINI_KEYS = Object.entries(process.env)
  .filter(([k]) => k.startsWith("GEMINI_API_KEY"))
  .map(([, v]) => v)
  .filter(Boolean);

if (GEMINI_KEYS.length === 0) {
  console.error("❌ No Gemini API keys found");
  process.exit(1);
}

function pickKey() {
  return GEMINI_KEYS[Math.floor(Math.random() * GEMINI_KEYS.length)];
}

function run(cmd) {
  return execSync(cmd, { encoding: "utf8" }).trim();
}

function comment(body) {
  run(`gh pr comment ${prNumber} --body "${body.replace(/"/g, '\\"')}"`);
}

function closePR(reason) {
  run(`gh pr close ${prNumber} --comment "${reason}"`);
  process.exit(1);
}

function mergePR() {
  run(`gh pr merge ${prNumber} --squash --auto`);
}

// ---------- BASIC VALIDATION ----------

// Only README.md allowed
const files = run(
  `gh pr view ${prNumber} --json files --jq '.files[].path'`
).split("\n");

if (!(files.length === 1 && files[0] === "README.md")) {
  closePR("❌ Only README.md changes are allowed.");
}

// Get diff
const diff = run(`gh pr diff ${prNumber}`);
if (diff.length < 100) {
  closePR("❌ Change is too small or meaningless.");
}

// ---------- LINK EXTRACTION ----------

// Extract added lines only
const addedLines = diff
  .split("\n")
  .filter(l => l.startsWith("+") && !l.startsWith("+++"))
  .join("\n");

// Extract URLs
const urlRegex = /(https?:\/\/[^\s)>\]]+)/g;
const newLinks = [...new Set(addedLines.match(urlRegex) || [])];

if (newLinks.length === 0) {
  closePR("❌ No valid links added.");
}

// ---------- DUPLICATE CHECK ----------

// Get current README (main branch)
const baseReadme = run(`git show origin/main:README.md`);
const existingLinks = new Set(baseReadme.match(urlRegex) || []);

const duplicates = newLinks.filter(l => existingLinks.has(l));
if (duplicates.length > 0) {
  closePR(
    `❌ Duplicate link(s) already exist:\n\n${duplicates.join("\n")}`
  );
}

// ---------- LINK HEALTH CHECK ----------

async function checkLink(url) {
  try {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), LINK_TIMEOUT_MS);

    const res = await fetch(url, {
      method: "HEAD",
      redirect: "follow",
      signal: controller.signal
    });

    return res.status >= 200 && res.status < 400;
  } catch {
    return false;
  }
}

async function validateLinks() {
  const results = await Promise.all(
    newLinks.map(async link => ({
      link,
      ok: await checkLink(link)
    }))
  );

  const dead = results.filter(r => !r.ok).map(r => r.link);

  if (dead.length > 0) {
    closePR(
      `❌ Dead or unreachable link(s):\n\n${dead.join("\n")}`
    );
  }
}

// ---------- GEMINI REVIEW ----------

async function geminiReview() {
  const apiKey = pickKey();

  const prompt = `
You are a strict open-source maintainer.

Review this README change.

Criteria:
1. Relevant to AI systems (foundations → models → agents → IDEs → trends)
2. Technically correct
3. Not hype or marketing
4. Useful for developers
5. High-signal resource

Score 1–10.

Return ONLY valid JSON:
{
  "overall_score": number,
  "decision": "merge" | "needs_changes" | "reject",
  "reason": "short explanation"
}

README DIFF:
${diff}
`;

  const res = await axios.post(
    `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${apiKey}`,
    {
      contents: [{ parts: [{ text: prompt }] }]
    }
  );

  const text = res.data.candidates[0].content.parts[0].text;
  return JSON.parse(text);
}

// ---------- EXECUTION ----------

(async () => {
  try {
    await validateLinks();

    const result = await geminiReview();

    if (result.decision === "merge" && result.overall_score >= SCORE_TO_MERGE) {
      comment(`✅ **Gemini Approved**\n\n${result.reason}`);
      mergePR();
    } else if (result.decision === "needs_changes") {
      comment(`🟡 **Changes Needed**\n\n${result.reason}`);
    } else {
      closePR(`❌ **Rejected**\n\n${result.reason}`);
    }
  } catch (e) {
    closePR("❌ Review failed (invalid response or network issue).");
  }
})();
