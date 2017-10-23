<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Batch Normalization

### Hyper Parameter Tuning

Normalizing the input features can speed up the learning. Normalize t

Something that is more elongated to something that is more normal. Activations A1 A2 etc. Couldn't we normalize the mean and the variance help train more efficiently?

Can we normalize the values of A to train W b fast?

This is what batch normalization does

Normalizing $Z^2$ 
 
## The process

Given some intermediate values in NN $$Z^{(1)},...,Z^{(m)}$$ 

Compute the mean as follows 

$$\mu = \frac{1}{m} \sum_{i}{Z^{(i)}}$$ 

$$\sigma^2 = \frac{1}{m} \sum_{i}{(Z - \mu)} $$

$$Z_{norm}^{(i)} = \frac{Z^{(i)} - \mu}{\sqrt{\sigma^2+\epsilon}} $$

$$\tilde{Z}^{(i)} = \gamma Z_{norm}^{(i)} + \beta $$

$\beta$ and $\gamma$ are learnable parts of the model. When $\beta =\mu$ then  $\tilde{Z}^{(i)}$ would exactly invert: 

$$\frac{Z^{(i)} - \mu}{\sqrt{\sigma^2+\epsilon}} $$

and $$\tilde{Z}^{(i)} = Z^{(i)}$$

These 4 equations are calculating the identify function. Where previously you were using $$Z^{(1)},...,Z^{(m)}$$ 

We use

$$\tilde{Z}^{(1)} ,..., \tilde{Z}^{(m)}$$ 

Normalizing the input features can help learning in a neural network. Batch norm applies that same normlization process not just to the input layer, but to values deep within the hidden layer. Depending on how you have made your NN, you may not want your hidden values to be of $\mu=0$ and $\sigma^2 = 1$. consider if you use a sigmoid function, you may not want all your values to be in the linear portion of the sigmoid curve, but may want to take more advantage of the non-linear tails.