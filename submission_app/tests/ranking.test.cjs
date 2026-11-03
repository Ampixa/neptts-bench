const test = require('node:test');
const assert = require('node:assert/strict');
const {rankWithinProfile, generationLabel} = require('../static/ranking.js');

test('judges may reverse systems; every displayed rank belongs to one exact profile', () => {
  const entries = [
    {id: 'a-good', profile: 'judge-a', report: {cer: 0.1}},
    {id: 'b-bad', profile: 'judge-b', report: {cer: 0.8}},
    {id: 'a-bad', profile: 'judge-a', report: {cer: 0.5}},
    {id: 'b-good', profile: 'judge-b', report: {cer: 0.01}},
  ];
  const original = [...entries];
  assert.deepEqual(rankWithinProfile(entries, 'judge-a').map(e => e.id), ['a-good', 'a-bad']);
  assert.deepEqual(rankWithinProfile(entries, 'judge-b').map(e => e.id), ['b-good', 'b-bad']);
  assert.deepEqual(rankWithinProfile(entries, ''), []);
  assert.deepEqual(rankWithinProfile(entries, 'unknown'), []);
  assert.deepEqual(entries, original);
});


test('generation labels describe declared total attempts and preserve unknown legacy records', () => {
  const declaration = {
    generation_mode: 'best_of_n', candidate_count: 12, selector_model: 'declared-selector',
    judge_family_used: true, judge_family_details: 'Selection only; no RL',
    selection_policy: 'Eight selection candidates; twelve total attempts including retries',
  };
  assert.equal(generationLabel(declaration), 'Best-of-N · max 12 attempts/prompt');
  assert.equal(generationLabel({selection_policy: 'best-of-eight with retries'}), 'Unknown / legacy');
  assert.equal(generationLabel({generation_mode: 'single_take', candidate_count: 1}), 'Unknown / legacy');
  assert.equal(generationLabel({...declaration, generation_mode: 'single_take', candidate_count: 1, selector_model: 'none'}), 'Single-take · N=1');
});
