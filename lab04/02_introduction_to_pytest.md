# Introduction to unit testing with `pytest`

[`pytest`](https://docs.pytest.org/) is the de-facto standard for unit testing in python.
Why are we learning about unit testing in a class about MLOps you wonder? Well, ML _is_ a lot about code (data processing, model definitions, training loops - it's all code!) and code requires testing for many reasons:

1. **Identifying Bugs and Defects**: Testing helps to identify bugs and defects in the software early in the development process.

2. **Ensuring Reliability and Stability**: Testing ensures that the software behaves reliably and stably under different conditions.

3. **Maintaining Code Quality**: Testing encourages developers to write clean, modular, and maintainable code.

4. **Supporting Refactoring and Changes**: Testing provides a safety net for making changes to the codebase.

5. **Enhancing Collaboration and Communication**: Testing fosters collaboration and communication within development teams. Test cases serve as a common language between developers, testers, and other stakeholders, helping to clarify requirements and expectations. It is also the best documentation for code you can possibly wish for!

In summary, testing is essential for delivering high-quality software that meets user needs, maintains reliability, and supports continuous improvement. It's not just a phase in the development process but a fundamental practice that underpins the entire software lifecycle. For a masterclass in ML code testing, take a loot at the [`test` directory in the hugging face `transformers` repository](https://github.com/huggingface/transformers/tree/main/tests). They have test for every model, every optimization strategy, pipeline etc.

Now that I have convinced you, that testing is good practice, let's take a look at `pytest`.

## Get started

As for almost every other python package, you install it via `pip`:

```shell
pip install -U pytest
```

However, we have already taken care of this for you in the `environment.yaml` for this lab.

Let's write our first test. Create a file named `test_answer.py`, containing the following snippet:

```python
def answer_to_life_universe_everything():
    return 42

def test_answer():
    assert 42 == answer_to_life_universe_everything()
```

Then, run `pytest` in the same directory. You should see the test case succeed with output similar to the following:

```raw
============================= test session starts ==============================
collected 1 item

test_answer.py .                                                         [100%]

============================== 1 passed in 0.01s ===============================

```

Next, let's add the following test case:

```python
def test_answer_fail():
    assert 47 == answer_to_life_universe_everything()
```

<details>
    <summary>Why 47?</summary>

[Glad you asked.](https://web.archive.org/web/20180823042044/https://memory-alpha.wikia.com/wiki/47)
</details>

You will see that this new test case fails:

```raw
============================= test session starts ==============================
collected 2 items

test_answer.py .F                                                        [100%]

=================================== FAILURES ===================================
_______________________________ test_answer_fail _______________________________

    def test_answer_fail():
>       assert 47 == answer_to_life_universe_everything()
E       assert 47 == 42
E        +  where 42 = answer_to_life_universe_everything()

test_answer.py:8: AssertionError
=========================== short test summary info ============================
FAILED test_answer.py::test_answer_fail - assert 47 == 42
========================= 1 failed, 1 passed in 0.02s ==========================
```

The `[100%]` refers to the overall progress of running all test cases. After it finishes, `pytest` then shows a failure report because 47 is the wrong answer.

These simple examples already demonstrate a few important features.

1. `pytest` will run all files of the form `test_*.py` or `*_test.py` in the current directory and all its subdirectories.
2. `pytest` considers functions of the form `test_*` test cases.
3. You can use the `assert` statement to verify test expectations.

### More on assertions

Sometimes, we expect code to raise an exception. `pytest` allows you to handle such situations using `pytest.raises`:

```python
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
```

---

**Your turn**: Amend `test_answer.py` with the following snippet:

```python
def validate_answer(answer):
    if 42 != answer:
        raise ValueError(f"Answer {answer} is wrong.")
```

Add a test case for `validate_answer` that assures that `validate_answer` raises a `ValueError`.

---

### Grouping tests

To keep your test suites nice and tidy, you can group tests into classes:

```python
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
```

There are no other requirements than prefixing the name of the class with `Test*`.
Beyond test organization, using test classes is also beneficial when combined with two advanced features we are introducing the the next sections: fixtures and marks.

Note that each test case has its own unique instance of the class. Sharing instances would be detrimental to test isolation!

### Interlude: Anatomy of a test

A test is meant to check the result of a particular behavior and ensure that the result aligns with expectations. By behavior, one refers to the way in which as system acts in response to a particular situation of stimulus. Generally, the _what_ is more important than the _how_ or _why_.

In a test, you can identify four steps:

1. Arrange: This is where you prepare everything for running your test.
2. Act: This is the action that triggers the "behavior". Ideally, this is a single action, with everything else already set up by the _arrange_ step.
3. Assert: Here, you compare the observed behavior against the expect behavior.
4. Cleanup: Finally, you free any resources used by the test.

So far, we've really only seen the _act_ and _assert_ steps. In the next section, you will be introduced to _fixtures_, `pytest`'s idiom for the _arranging_ the context of a test.

### Fixtures

A fixture provides a defined, reliable and consistent context for the tests. This could include environment (for example a database configured with known parameters) or content (such as a dataset). Fixtures define the steps and data that constitute the _arrange_ phase of a test.

If you have ever written unit tests in another programming language, chances are that you have come across xUnit (e.g. in Java or C#). In this case, you may be wondering why one would ever want to use fixtures over setup/teardown functions. [You can find a comparison of the two in the docs](https://docs.pytest.org/en/7.1.x/explanation/fixtures.html#improvements-over-xunit-style-setup-teardown-functions)!

In `pytest`, the services, state, or other operating environments (the _context_ of a test) set up by fixtures are accessed by test functions through arguments. For each fixture used by a test function there is typically a parameter (named after the fixture) in the test function’s definition.

`pytest` fixtures are functions decorated with [`@pytest.fixture`](https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest.fixture). Consider the following example taken from the `pytest` documentation:

```python
import pytest


class Fruit:
    """
    This is just a class representing fruit.
    """
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name


@pytest.fixture
def my_fruit():
    return Fruit("apple")


@pytest.fixture
def fruit_basket(my_fruit):
    return [Fruit("banana"), my_fruit]


def test_my_fruit_in_basket(my_fruit, fruit_basket):
    assert my_fruit in fruit_basket
```

Here, `my_fruit` and `fruit_basket` are fixtures. As you can see, the `test_my_fruit_in_basket` test case has arguments `my_fruit` and `fruit_basket`, both of which are fixtures! `pytest` automagically passes in the correct fixture.


### Marks

Marks are additional metadata on your test functions. Using the `pytest.mark` helper, you can easily set metadata on your test functions. `pytest` comes with [a range of predefined markers](https://docs.pytest.org/en/7.1.x/reference/reference.html#marks-ref), which includes:

Here are some of the builtin markers:

- `usefixtures` - use fixtures on a test function or class
- `filterwarnings` - filter certain warnings of a test function
- `skip` - always skip a test function
- `skipif` - skip a test function if a certain condition is met
- `xfail` - produce an “expected failure” outcome if a certain condition is met
- `parametrize` - perform multiple calls to the same test function.

You can of course also [create your own markers](https://docs.pytest.org/en/7.1.x/example/markers.html#mark-examples). In the next section you will see how `@pytest.mark.parametrize` can be used to parameterize test functions.

#### Parametrizing test functions

Test parametrization using `pytest` is easy:

```python
import pytest

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected

```

`pytest.mark.parametrize` takes the name(s) of the arguments it should parametrize as a single string, in this example `"test_input,expected". The second argument is a range of tuples - one element per parametrized function argument.

The example above produces the following output:

```raw
=========================== test session starts ============================
collected 3 items

test_expectation.py ..F                                              [100%]

================================= FAILURES =================================
____________________________ test_eval[6*9-42] _____________________________

test_input = '6*9', expected = 42

    @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
    def test_eval(test_input, expected):
>       assert eval(test_input) == expected
E       AssertionError: assert 54 == 42
E        +  where 54 = eval('6*9')

test_expectation.py:6: AssertionError
========================= short test summary info ==========================
FAILED test_expectation.py::test_eval[6*9-42] - AssertionError: assert 54...
======================= 1 failed, 2 passed in 0.12s ========================
```

As you can see, each parameter combination created one test "item".

## Testing PyTorch code

With all the testing goodness that you learnt in the previous section, we can now venture on to a more applied topic: testing PyTorch code. Here, it is important to note that we are not (yet) testing models, we are solely testing the code that defines the model. This is an important distinction!

PyTorch code (and any other machine learning code) tends to manipulate floating-point numbers, which can lead to nasty surprises. For instance, comparing two floating point numbers using `==` is a bad idea!
For instance, what do you think is the result of the last line in the following snippet?

```python
num1 = 0.1 + 0.2
num2 = 0.3
num1 == num2  # <--- What does this return?
```

<details>
    <summary> Solution </summary>
As you probably guessed from the overly suggestive wording, `(0.1 + 0.2) == 0.3` evaluates to `False`.
</details>

As you can see, even though mathematically the values should be equal, they are not considered equal when compared using ==. This is because of the limited precision with which floating-point numbers are represented in computers, leading to tiny differences in the actual stored values. For more information on the peculiarities of floating-point numbers, please refer to [What every computer scientist should know about floating-point arithmetic](https://dl.acm.org/doi/10.1145/103162.103163) (an excellent read!).

How do we circumvent this? To compare floating-point numbers accurately, it's better to check if the absolute difference between the numbers is within a small tolerance instead of using `==`:

```python
num1 = 0.1 + 0.2
num2 = 0.3

# Define a tolerance level
tolerance = 1e-10  # Adjust this tolerance based on your application's requirements

# Compare the absolute difference between the numbers with tolerance
if abs(num1 - num2) < tolerance:
    print("The numbers are approximately equal.")
else:
    print("The numbers are not equal.")
```

This approach allows for small discrepancies due to floating-point arithmetic and provides more reliable comparisons for floating-point numbers. PyTorch comes with it's own utility for testing "closeness":

> [`torch.testing.assert_close(actual, expected, *, allow_subclasses=True, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, check_layout=True, check_stride=False, msg=None)`](https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close),

which asserts that actual and expected are close in the sense that $|\mathrm{actual} - \mathrm{expected}| \leq \mathrm{atol} + \mathrm{rtol}\cdot|\mathrm{expected}|$.

### What to test?

When writing tests for machine learning code, _what_  behavior should you test? Here are some ideas:

#### Correctness of Computation

- Verify that mathematical operations (e.g., matrix multiplications, activations functions) are implemented correctly.
- Test custom loss functions, ensuring they compute gradients accurately and match expected values.
- Check that gradients are computed correctly using numerical differentiation or, if feasible, against analytical gradients.
- Test custom layers or modules to ensure they produce correct outputs according to their mathematical definitions.

#### Correctness of Output Dimensions

- Ensure that the dimensions of output tensors from layers and modules match the expected dimensions.
- Verify that the output dimensions of convolutional, pooling, and other layers are calculated correctly.
- Test reshaping operations to ensure they produce output tensors with the correct shape.
- Check that the final output of the model matches the expected shape and size for the given task.

#### Consistency of Model Outputs

This is especially important when using random numbers.

- Test model outputs across multiple runs with the same input data to ensure consistency.
- Verify that the model produces consistent predictions for the same input across different hardware or software environments.
- Test model outputs against known benchmarks or reference implementations to ensure correctness.
- Always seed your random number generator, especially in tests! Even better, do not use random numbers in tests!

#### Numerical stability

- Test implementations against inputs of varying size - this can expose numerical instabilities.
- Test stability by introducing small perturbations to input data and verifying that outputs remain consistent.

---

These are of course only ideas, and the _what_ and _how_  depend a lot on what you are implementing.

### Your turn: ~~Riding on~~ Testing the tensor train

Now it is your turn. You are provided with the implementation of a custom layer called `TensorTrainLayer`. This layer performs tensor contractions in a tensor train format. Your task is to write testing code to ensure that the implementation of `TensorTrainLayer` is correct and behaves as expected.

1. Write a `pytest` test suite to cover the following scenarios:
    - Test with random input tensors of various sizes and shapes.
    - Test with different combinations of input modes, output modes, and ranks.
    - Test with known input-output pairs to verify the correctness of the tensor contractions.

    ```python
    class TensorTrainLayer(nn.Module):
        def __init__(self, in_modes, out_modes, ranks):
            super(TensorTrainLayer, self).__init__()
            assert len(in_modes) == len(out_modes) == len(ranks) + 1  # Check dimensions
            self.in_modes = in_modes
            self.out_modes = out_modes
            self.ranks = ranks
            self.weights = nn.ParameterList()
            for i in range(len(ranks)):
                self.weights.append(nn.Parameter(torch.randn(in_modes[i], out_modes[i], ranks[i], ranks[i+1])))

        def forward(self, x):
            # Reshape input tensor to tensor train format
            tensor_train = x.view(self.in_modes[0], -1, self.out_modes[-1])
            for i in range(len(self.weights)):
                # Apply tensor contraction
                tensor_train = torch.einsum('mnr, nrm -> nm', tensor_train, self.weights[i])
            return tensor_train
    ```

2. Extend the demo or write a new GitHub Actions workflow that executes your test suite.

<details>
    <summary>What is a tensor train (layer)?</summary>

The  "Tensor Train" is a tensor decomposition technique that can be used to represent high-dimensional tensors in a compact format by decomposing them into a series of smaller, low-dimensional tensors.

In a Tensor Train Layer, each neuron represents a tensor in the tensor train format. The input to the layer is decomposed into a series of smaller tensors, and the layer applies a series of operations on these tensors to transform the input data. This can be particularly useful for tasks involving high-dimensional data, such as image or video processing, where traditional neural network architectures may struggle due to the curse of dimensionality.

Tensor Train Layers have been explored in research for tasks such as image super-resolution, video classification, and medical image analysis. They offer advantages in terms of memory efficiency and computational speed compared to traditional fully connected or convolutional layers, especially when dealing with large-scale datasets or complex data structures.

Tensor trains are also frequently used in modern quantum chemistry, a field dear to the heart of the author of this document. :)
</details>
