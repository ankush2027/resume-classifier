"""Stage 3 remains authoritative for its own score; this is a separate evidence view."""
from src.matching import match_candidate_to_job
from src.matching.matcher import WEIGHTS, _skills
from src.resume_intelligence.parser import extract_skills
from .extractor import candidate_sources, extract_evidence, snippets
from .models import CandidateAssessment, Evidence, ExperienceRelevance, RequirementAssessment
from .rules import MULTIPLIERS, classify, domains, mentioned


def _requirement(requirement, candidate, sources):
    evidence = extract_evidence(requirement, sources)
    # A structured skill without recoverable text is a mention, not a fabricated quote.
    if not any(item.relevance == 'direct' for item in evidence):
        for index, skill in enumerate(candidate.skills):
            if requirement in _skills([skill]):
                evidence.insert(0, Evidence(requirement, requirement, 'skills', skill, 'weak',
                                           'skills', 'direct', f'skills[{index}]', True,
                                           'Structured skill entry only; no contextual snippet available.'))
                break
    direct = [item for item in evidence if item.relevance == 'direct' and item.supports_requirement]
    transferable = [item for item in evidence if item.relevance == 'transferable' and item.supports_requirement]
    result = RequirementAssessment(requirement, 'missing', evidence=evidence,
                                   direct_match=bool(direct), transferable=bool(transferable))
    if direct:
        best = max(direct, key=lambda item: MULTIPLIERS[item.strength])
        result.evidence_strength = best.strength
        if best.strength != 'weak':
            result.status = 'evidenced'
            result.explanation = f'{requirement}: {best.strength} direct evidence from {best.source_type}. {best.reason}'
        elif all(item.source_path.startswith('skills[') for item in direct):
            result.status = 'present_no_context'
            result.explanation = f'{requirement} is listed in structured skills; supporting context is unavailable.'
        else:
            result.status = 'weak_evidence'
            result.explanation = f'{requirement} is mentioned, but concrete supporting use was not found. {best.reason}'
    elif transferable:
        result.status = 'transferable'
        result.evidence_strength = max(transferable, key=lambda item: MULTIPLIERS[item.strength]).strength
        concepts = ', '.join(dict.fromkeys(item.concept for item in transferable))
        result.explanation = f'Related {concepts} evidence exists; {requirement} itself is not directly evidenced. Related does not mean equivalent.'
    elif evidence:
        result.status = 'unknown'
        result.explanation = f'{requirement} appears only in unsupported/negated context; capability remains unknown.'
    else:
        result.explanation = f'No {requirement} evidence was found in the supplied profile; this does not prove absence of the skill.'
    return result


def _experience_relevance(job, sources):
    results = []
    for requirement in job.experience_requirements:
        wanted_skills = extract_skills(requirement, explicit_skills=True)
        wanted_domains = domains(requirement)
        full, partial = [], []
        if wanted_skills or wanted_domains:
            for kind, _, _, text in sources:
                if kind != 'experience':
                    continue
                # Keep relevance local to the same snippet, not distributed over unrelated roles.
                for snippet in snippets(text):
                    strength, supports, _ = classify(kind, snippet)
                    if not supports or strength != 'strong':
                        continue
                    checks = [mentioned(skill, snippet) for skill in wanted_skills]
                    checks += [domain in domains(snippet) for domain in wanted_domains]
                    if all(checks):
                        full.append(snippet)
                    elif any(checks):
                        partial.append(snippet)
        status = 'relevant' if full else 'partial' if partial else 'unknown'
        reason = {'relevant': 'Concrete work text contains all supported skill/domain markers.',
                  'partial': 'Some markers occur in work text, but the requested context is incomplete.',
                  'unknown': 'No supported context markers or sufficient work snippets establish relevance.'}[status]
        results.append(ExperienceRelevance(requirement, status, list(dict.fromkeys(full or partial)),
                                            reason + ' This does not verify duration or employment.'))
    return results


def _adjusted_score(result):
    """Weight direct text evidence; keep Stage 3 duration/education components intact."""
    baseline = result.match_result
    values = {}
    for kind, assessments in [('required', result.required_requirements), ('preferred', result.preferred_requirements)]:
        values[kind] = (sum(MULTIPLIERS[item.evidence_strength] for item in assessments
                            if item.direct_match) / len(assessments)
                        if assessments else None)
    for name in ['experience', 'education']:
        status = getattr(baseline, name + '_match')
        values[name] = 1.0 if status == 'satisfied' else 0.0 if status == 'not_satisfied' else None
    result.score_coverage = round(sum(WEIGHTS[name] for name, value in values.items() if value is not None), 10)
    if result.score_coverage:
        result.evidence_adjusted_match_score = round(
            100 * sum(WEIGHTS[name] * value for name, value in values.items() if value is not None)
            / result.score_coverage, 2)


def assess_candidate(candidate_profile, job_profile) -> CandidateAssessment:
    """Add auditable evidence to a single candidate/job match, without modifying either."""
    candidate, job = candidate_profile, job_profile
    baseline = match_candidate_to_job(candidate, job)
    result = CandidateAssessment(candidate.candidate_name, baseline.overall_match_score, baseline)
    sources = candidate_sources(candidate)
    result.required_requirements = [_requirement(skill, candidate, sources) for skill in _skills(job.required_skills)]
    result.preferred_requirements = [_requirement(skill, candidate, sources) for skill in _skills(job.preferred_skills)]
    result.experience_relevance = _experience_relevance(job, sources)
    result.evidence_strength_summary = {'strong': 0, 'moderate': 0, 'weak': 0,
                                        'transferable': 0, 'missing': 0, 'unknown': 0}
    # Count unique requirements, not occurrences or duplicated required/preferred entries.
    seen = set()
    for item in result.required_requirements + result.preferred_requirements:
        if item.requirement in seen:
            continue
        seen.add(item.requirement)
        bucket = item.evidence_strength if item.direct_match else item.status
        result.evidence_strength_summary[bucket] += 1
        if item.status == 'evidenced':
            result.strengths.append(item.explanation)
        else:
            result.weaknesses.append(item.explanation)
            result.concerns.append(f'Verify {item.requirement}: supporting direct use is limited or unavailable.')
        if item.direct_match and item.requirement not in baseline.candidate_skills:
            result.concerns.append(f'{item.requirement} appears in source text but not in the extracted skill list; Stage 3 score is unchanged.')
        # Neutral repetition flag based only on raw weak/keyword lines, not duplicate representations.
        weak_lines = [line for line in candidate.raw_text.splitlines()
                      if mentioned(item.requirement, line) and classify('experience', line)[0] == 'weak']
        if len(weak_lines) >= 5:
            result.concerns.append(f'{item.requirement} occurs on at least five weak-context lines; repetition adds no evidence credit.')
    if job.experience_requirements and (baseline.experience_match == 'unknown' or
                                       any(item.status != 'relevant' for item in result.experience_relevance)):
        result.concerns.append('Experience duration and/or relevance could not be fully verified.')
    if job.education_requirements and baseline.education_match == 'unknown':
        result.concerns.append('Education information does not establish the stated requirement.')
    _adjusted_score(result)
    result.explanation = (
        f'Stage 3 rule-based match score: {result.overall_match_score}. '
        f'Evidence-adjusted match score: {result.evidence_adjusted_match_score}. '
        'Direct matched skills use strong=1.0, moderate=0.7, weak=0.3; only the strongest '
        'supporting snippet per requirement counts. Transferable evidence adds no direct credit. '
        'Experience/education components retain Stage 3 checks; relevance is reported separately. '
        f'Assessed weight coverage: {result.score_coverage:.0%}. '
        'Unknown/unspecified Stage 3 components remain excluded. These are deterministic '
        'text-evidence rules, not probabilities, verified experience, or hiring recommendations.')
    return result
