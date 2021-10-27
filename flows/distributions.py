import jax.numpy as np
from jax import random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm, multivariate_normal
from jax.experimental import stax


def Normal():
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return norm.logpdf(inputs).sum(1)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples, input_dim))

        return (), log_pdf, sample

    return init_fun


def GMM(means, covariances, weights):
    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            cluster_lls = []
            for log_weight, mean, cov in zip(np.log(weights), means, covariances):
                cluster_lls.append(
                    log_weight + multivariate_normal.logpdf(inputs, mean, cov)
                )
            return logsumexp(np.vstack(cluster_lls), axis=0)

        def sample(rng, params, num_samples=1):
            cluster_samples = []
            for mean, cov in zip(means, covariances):
                rng, temp_rng = random.split(rng)
                cluster_sample = random.multivariate_normal(
                    temp_rng, mean, cov, (num_samples,)
                )
                cluster_samples.append(cluster_sample)
            samples = np.dstack(cluster_samples)
            idx = random.categorical(rng, weights, shape=(num_samples, 1, 1))
            return np.squeeze(np.take_along_axis(samples, idx, -1))

        return (), log_pdf, sample

    return init_fun


def MLPEmbedding(
    rng,
    input_dim,
    embedding_dim=32,
    hidden_dim=128,
    num_layers=2,
    act=None,
    use_context_embedding=True,
):
    assert (
        use_context_embedding
    ), "Initializing embedding with use_context_embedding==False"

    if act is None:
        act = stax.Selu

    layers = [lyr for _ in range(num_layers) for lyr in (stax.Dense(hidden_dim), act)]
    layers += [stax.Dense(embedding_dim)]
    init_random_params, embed = stax.serial(*layers)
    _, init_params = init_random_params(rng, (-1, input_dim))
    return init_params, embed


def Flow(
    transformation,
    prior=Normal(),
    context_embedding_kwargs=None,
):
    """
    Args:
        transformation: a function mapping ``(rng, input_dim)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_dim)`` to a
            ``(params, log_pdf, sample)`` triplet
        context_kwargs:
            Additional keyword arguments to pass to the context embedding function.
            use_context_embedding: whether to use the context embedding
            embedding_dim: the dimension of the context embedding (output_dim)
            hidden_layers: Number of hidden layers in the context embedding.
            hidden_dim: Number of hidden units in the hidden layers.
            act: Activation function to use in the hidden layers.

    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.

    Examples:
        >>> import flows
        >>> input_dim, rng = 3, random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> flow_params, log_pdf, sample = init_fun(rng, input_dim)
    """

    assert "embedding_dim" in context_embedding_kwargs, "Must specify embedding_dim"
    assert (
        "use_context_embedding" in context_embedding_kwargs
    ), "Must specify use_context_embedding"

    def init_fun(rng, input_dim, context_dim=None, hidden_dim=64):
        if (
            context_embedding_kwargs["embedding_dim"] is None
            or not context_embedding_kwargs["use_context_embedding"]
        ):
            embedding_dim = context_dim
        else:
            embedding_dim = context_embedding_kwargs["embedding_dim"]

        transformation_rng, embedding_rng, prior_rng = random.split(rng, 3)

        if context_embedding_kwargs["use_context_embedding"]:
            embedding_params, context_embedding = MLPEmbedding(
                embedding_rng, context_dim, **context_embedding_kwargs
            )
        else:
            embedding_params = None
            context_embedding = None

        flow_params, direct_fun, inverse_fun = transformation(
            transformation_rng,
            input_dim,
            context_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        params = (embedding_params, flow_params)

        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_dim)

        def embed(embedding_params, context):
            if context is not None and context_embedding is not None:
                assert embedding_params is not None, "embedding_params must be provided"
                context = context_embedding(embedding_params, context)
            return context

        def log_pdf(params, inputs, context=None, embedding_params=None):
            embedding_params, flow_params = params

            context = embed(embedding_params, context)
            u, log_det = direct_fun(flow_params, inputs, context=context)
            log_probs = prior_log_pdf(prior_params, u)
            return log_probs + log_det

        def sample(rng, params, context=None, num_samples=None, embedding_params=None):
            assert (
                num_samples is not None or context is not None
            ), "Only one of num_samples or context must be specified"
            if context is not None:
                num_samples = context.shape[0]

            embedding_params, flow_params = params

            context = embed(embedding_params, context)
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(flow_params, prior_samples, context=context)[0]

        return params, log_pdf, sample

    return init_fun
