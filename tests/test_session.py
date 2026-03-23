import plp


def test_session_records_multiple_calls(llm):
    sess = plp.session(llm=llm, system="You are concise.")
    first = sess.ask("State one word.")
    second = sess.ask("State another word.")

    assert first.strip()
    assert second.strip()
    assert len(sess.trace()) == 2
    assert sess.trace()[0].primitive == "session.ask"


def test_session_fork_and_reset(llm):
    sess = plp.session(llm=llm)
    sess.ask("Provide a short greeting.")
    forked = sess.fork()

    forked.ask("Provide a short farewell.")
    assert len(sess.trace()) == 1
    assert len(forked.trace()) == 2

    forked.reset()
    assert forked.trace() == []
