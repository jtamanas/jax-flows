<img align="right" width="300" src="assets/flows.gif">

# Normalizing Flows in JAX

<a href="https://circleci.com/gh/ChrisWaites/jax-flows">
    <img alt="Build" src="https://img.shields.io/circleci/build/github/ChrisWaites/jax-flows/master">
</a>
<a href="https://github.com/ChrisWaites/jax-flows/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/ChrisWaites/jax-flows.svg?color=blue">
</a>
<a href="https://jax-flows.readthedocs.io/en/latest/">
    <img alt="Documentation" src="https://img.shields.io/website/http/jax-flows.readthedocs.io.svg?down_color=red&down_message=offline&up_message=online">
</a>

<p>Implementations of normalizing flows (RealNVP, GLOW, MAF) in the <a href="https://github.com/google/jax/">JAX</a> deep learning framework.</p>

## What are normalizing flows?

[Normalizing flow models](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html) are _generative models_. That is, they infer the underlying probability distribution which generated a given dataset. With that distribution we can do a number of interesting things, namely query probability densities and sample new realistic points.

## How are things structured?

For a more thorough description, check out the [documentation](https://jax-flows.readthedocs.io/).

### Bijections

A `bijection` is a parameterized invertible function.

```python
init_fun = flows.MADE()

params, direct_fun, inverse_fun = init_fun(rng, input_shape)

# Transform inputs
transformed_inputs, log_det_jacobian_direct = direct_fun(params, inputs)

# Reconstruct original inputs
reconstructed_inputs, log_det_jacobian_inverse = inverse_fun(params, inputs)

assert np.array_equal(inputs, reconstructed_inputs)
```

We can construct a sequence of bijections using `flows.serial`. The result is just another bijection, and adheres to the exact same interface.

```python
init_fun = flows.serial(
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse()
)

params, direct_fun, inverse_fun = init_fun(rng, input_shape)
```

### Distributions

A `distribution` is characterized by a probability density querying function, a sampling function, and its parameters.

```python
init_fun = flows.Normal()

params, log_pdf, sample = init_fun(rng, input_shape)

# Query probability density of points
log_pdfs = log_pdf(params, inputs)

# Draw new points
samples = sample(rng, params, num_samples)
```

### Normalizing Flow Models

Under this definition, a normalizing flow model is just a `distribution`. But to retrieve one, we have to give it a `bijection` and another `distribution` to act as a prior.

```python
bijection = flows.serial(
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse(),
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse(),
)

prior = flows.Normal()

init_fun = flows.Flow(bijection, prior)

params, log_pdf, sample = init_fun(rng, input_shape)
```

### How do I train a model?

To train our model, we would typically define an appropriate loss function and parameter update step.

```python
def loss(params, inputs):
  return -log_pdf(params, inputs).mean()

@jit
def step(i, opt_state, inputs):
  params = get_params(opt_state)
  gradient = grad(loss)(params, inputs)
  return opt_update(i, gradient, opt_state)
```

Given these, we can go forward and execute a standard JAX training loop.

```python
batch_size = 32

itercount = itertools.count()
for epoch in range(num_epochs):
  npr.shuffle(X)
  for batch_index in range(0, len(X), batch_size):
    opt_state = step(
      next(itercount),
      opt_state,
      X[batch_index:batch_index+batch_size]
    )

optimized_params = get_params(opt_state)
```

Now that we have our trained model parameters, we can query and sample as regular.

```python
log_pdfs = log_pdf(optimized_params, inputs)

samples = sample(rng, optimized_params, num_samples)
```

_Magic!_

## Interested in contributing?

Yay! Check out our [contributing guidelines](https://github.com/ChrisWaites/jax-flows/blob/master/.github/CONTRIBUTING.md).

## Inspiration

This repository is largely modeled after the [`pytorch-flows`](https://github.com/ikostrikov/pytorch-flows) repository by [Ilya Kostrikov](https://github.com/ikostrikov) and the [`nf-jax`](https://github.com/ericjang/nf-jax) repository by [Eric Jang](http://evjang.com/).

The implementations are modeled after the work of the following papers:

  > [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)\
  > Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio\
  > _arXiv:1605.08803_

  > [Improving Variational Inference with Inverse Autoregressive Flow
](https://arxiv.org/abs/1606.04934)\
  > Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling\
  > _arXiv:1606.04934_

  > [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)\
  > Diederik P. Kingma, Prafulla Dhariwal\
  > _arXiv:1807.03039_

  > [Flow++: Improving Flow-Based Generative Models
  with Variational Dequantization and Architecture Design](https://openreview.net/forum?id=Hyg74h05tX)\
  > Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel\
  > _OpenReview:Hyg74h05tX_

  > [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)\
  > George Papamakarios, Theo Pavlakou, Iain Murray\
  > _arXiv:1705.07057_

