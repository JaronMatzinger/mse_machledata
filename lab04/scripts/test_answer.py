import pytest

class TestClass:
    def test_answer(self):
        assert 42 == answer_to_life_universe_everything()

    def test_answer_fail(self):
        assert 47 != answer_to_life_universe_everything()

    def test_validate_answer_raises_value_error(self):
        with pytest.raises(ValueError, match='Answer 47 is wrong'):
            validate_answer(47)
    
def answer_to_life_universe_everything():
    return 42

def validate_answer(answer):
    if 42 != answer:
        raise ValueError(f'Answer {answer} is wrong.')