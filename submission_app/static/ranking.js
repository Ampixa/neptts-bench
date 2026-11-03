/* Exact evaluator identities are the boundary for every displayed rank. */
function rankWithinProfile(entries, profile) {
  if (!profile) return [];
  return entries.filter(entry => entry.profile === profile)
    .sort((a, b) => a.report.cer - b.report.cer);
}
function generationLabel(manifest) {
  const count = manifest.candidate_count;
  const selector = manifest.selector_model;
  if (!Number.isInteger(count) || count < 1 || count > 10000 ||
      typeof selector !== 'string' || !selector.trim() ||
      typeof manifest.judge_family_used !== 'boolean' ||
      typeof manifest.judge_family_details !== 'string' || !manifest.judge_family_details.trim()) {
    return 'Unknown / legacy';
  }
  if (manifest.generation_mode === 'single_take' && count === 1 && selector.toLowerCase() === 'none') {
    return 'Single-take · N=1';
  }
  if (manifest.generation_mode === 'best_of_n' && count >= 2 && selector.toLowerCase() !== 'none') {
    return `Best-of-N · max ${count} attempts/prompt`;
  }
  return 'Unknown / legacy';
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {rankWithinProfile, generationLabel};
}
