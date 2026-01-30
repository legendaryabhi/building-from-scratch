const fs = require("fs");
const axios = require("axios");
const { execSync } = require("child_process");

// ---------- PR CONTEXT ----------
const event = JSON.parse(
  fs.readFileSync(process.env.GITHUB_EVENT_PATH, "utf8")
);

const prNumber = event.pull_request.number;

// ---------- CONFIG ----------
const SCORE_TO_MERGE = 8;
const MODEL = "gemini-2.5-flash";

// ---------- GEMINI KEYS ----------
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

// ---------- HELPERS ----------
function run(cmd) {
  return execSync(cmd, { encoding: "utf8" }).trim();
}

function comment(body) {
  run(`gh pr comment ${prNumber} --body "${body.replace(/"/g, '\\"')}"`);
}

function closePR(reason) {
  console.log("PR rejected:", reason);
  run(`gh pr close ${prNumber} --comment "${reason}"`);
  process.exit(0); // important: workflow success
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
  closePR("❌ Change is too small or low-signal.");
}

// ---------- DUPLICATE LINK CHECK (TEXT ONLY) ----------

// Extract added lines
const addedLines = diff
  .split("\n")
  .filter(l => l.startsWith("+") && !l.startsWith("+++"))
  .join("\n");

const urlRegex = /(https?:\/\/[^\s)>\]]+)/g;
const newLinks = [...new Set(addedLines.match(urlRegex) || [])];

if (newLinks.length === 0) {
  closePR("❌ No new links added in README.");
}

// Existing README from main
const baseReadme = run(`git show origin/main:README.md`);
const existingLinks = new Set(baseReadme.match(urlRegex) || []);

const duplicates = newLinks.filter(l => existingLinks.has(l));
if (duplicates.length > 0) {
  closePR(
    `❌ Duplicate link(s) already exist:\n\n${duplicates.join("\n")}`
  );
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
5. High-signal contribution

Score from 1–10.

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
    const result = await geminiReview();

    if (result.decision === "merge" && result.overall_score >= SCORE_TO_MERGE) {
      comment(`✅ **Gemini Approved**\n\n${result.reason}`);
      mergePR();
    } else if (result.decision === "needs_changes") {
      comment(`🟡 **Changes Requested**\n\n${result.reason}`);
    } else {
      closePR(`❌ **Rejected**\n\n${result.reason}`);
    }
  } catch (e) {
    closePR("❌ Review failed (Gemini error or invalid JSON).");
  }
})();
