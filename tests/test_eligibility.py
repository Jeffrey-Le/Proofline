from proofline.eligibility import Profile, check, grounded
from proofline.schemas import Constraint, Extraction

POSTING = """Minimum qualifications: Currently pursuing a BS or MS in Computer Science or a related field.
Must graduate by June 2026. Based in Sunnyvale, CA.
Preferred: Returning to your degree program after the internship."""


def prof(**kw):
    base = dict(graduation="2026-12", degree_level="BS", fields=["Mathematics of Computation"],
                willing_locations=["Sunnyvale", "Los Angeles"], open_to_relocation=False,
                needs_sponsorship=False, next_program=None)
    return Profile(**(base | kw))


def C(category, kind="minimum", quote="Currently pursuing a BS or MS", **kw):
    base = dict(summary=category, grad_after=None, grad_before=None, degree_levels=[], locations=[],
                remote_ok=None, sponsorship_available=None, source_quote=quote)
    return Constraint(category=category, kind=kind, **(base | kw))


def X(*cs, start="2027-06", end="2027-09"):
    return Extraction(company="X", title="Intern", program_start=start, program_end=end,
                      constraints=list(cs), qualifications=[])


def one(c, p=None, **kw):
    return check(X(c, **kw), p or prof(), POSTING)[0]


def test_grad_window_blocks_with_clear_reason():
    f = one(C("graduation_window", grad_before="2026-06", quote="Must graduate by June 2026."))
    assert f.status == "blocker" and "Jun 2026" in f.reason and "Dec 2026" in f.reason


def test_grad_window_pass():
    assert one(C("graduation_window", grad_after="2026-09", grad_before="2027-06",
                 quote="Must graduate by June 2026.")).status == "pass"


def test_enrollment_after_grad_without_next_program_blocks():
    assert one(C("enrollment")).status == "blocker"


def test_enrollment_depends_on_pending_admission_is_warning():
    p = prof(next_program={"level": "MS", "start": "2027-06", "status": "applying"})
    assert one(C("enrollment"), p).status == "warning"


def test_return_to_school_preferred_is_warning_not_blocker():
    f = one(C("return_to_school", kind="preferred", quote="Returning to your degree program"))
    assert f.status == "warning"


def test_return_to_school_passes_when_admitted():
    p = prof(next_program={"level": "MS", "start": "2027-09", "status": "admitted"})
    assert one(C("return_to_school", quote="Returning to your degree program"), p).status == "pass"


def test_location():
    assert one(C("location", locations=["Sunnyvale, CA"], quote="Based in Sunnyvale, CA.")).status == "pass"
    assert one(C("location", locations=["London"], quote="Based in Sunnyvale, CA.")).status == "blocker"


def test_degree_level_counts_admitted_ms():
    c = C("degree_level", degree_levels=["MS"])
    assert one(c).status == "blocker"
    assert one(c, prof(next_program={"level": "MS", "start": "2027-09", "status": "admitted"})).status == "pass"


def test_ungrounded_quote_never_hard_blocks():
    f = one(C("graduation_window", grad_before="2026-06", quote="Must be a PhD wizard"))
    assert f.status == "warning" and not f.quote_grounded


def test_grounded_is_whitespace_and_quote_style_insensitive():
    assert grounded("currently   pursuing a BS", POSTING)
    assert grounded("Returning to your degree program.", POSTING)
    assert not grounded("pursuing a PhD", POSTING)
    assert not grounded("", POSTING)


def test_grounded_tolerates_model_wrapping_but_not_new_words():
    line = "Built a Python service that geocodes 40k delivery addresses a day and caches results in Redis"
    for q in ['"Built a Python service"', "“Built a Python service”", "<Built a Python service>",
              "[s-01] Built a Python service", "s-01: built a python service.",
              "Built a Python service ... caches results in Redis", "Built a Python service… in Redis"]:
        assert grounded(q, line), q
    for q in ["Built a Go service", "caches results in Redis ... Built a Python service",  # wrong order
              "Built a Python microservice", '""', "..."]:
        assert not grounded(q, line), q


def test_open_field_wording_passes():
    from proofline.eligibility import _degree_field, Profile
    from proofline.schemas import Constraint
    p = Profile(graduation="2026-12", degree_level="BS", fields=["Mathematics of Computation"],
                willing_locations=["US"], open_to_relocation=True, needs_sponsorship=False)
    mk = lambda q: Constraint(category="degree_field", kind="minimum", summary="field", degree_levels=[],
                              locations=[], source_quote=q)
    for q in ["A field relevant to the role as demonstrated through coursework, training, or professional experience",
              "BS in Computer Science or equivalent practical experience",
              "Bachelor's in Computer Science, Electrical Engineering, Robotics, or related field"]:
        assert _degree_field(mk(q), p, None)[0] == "pass", q
    assert _degree_field(mk("BS in Mechanical Engineering"), p, None)[0] == "warning"
