using CUDA
using MLDatasets
using Base.Iterators: partition
using LinearAlgebra
using Distributions
using Flux
using Parameters: @with_kw
include("StatisticalLoss.jl")

function make_loader(batch_size)
	# 1) Load & normalize on CPU
	imgs = Float32.(MLDatasets.MNIST(:train).features)   # 28×28×1×60000
	imgs .= (2.0f0 .* imgs .- 1.0f0)                         # now in [−1,1]

	# 2) Flatten spatial dims into “features”
	#    resulting shape: (784, 60000)
	flat = reshape(imgs, 28 * 28 * 1, :)

	# 3) Partition into vectors of column-index chunks,
	#    then pull out each sub-matrix
	return [flat[:, idxs] for idxs in partition(1:size(flat, 2), batch_size)]
end

# weight initialization as given in the paper https://arxiv.org/abs/1511.06434
dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0

function Discriminator()
	return Chain(
		Conv((4, 4), 1 => 64; stride = 2, pad = 1, init = dcgan_init),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		Conv((4, 4), 64 => 128; stride = 2, pad = 1, init = dcgan_init),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		x -> reshape(x, 7 * 7 * 128, :),
		Dense(7 * 7 * 128, 1))
end

function Generator(latent_dim::Int)
	return Chain(
		Dense(latent_dim, 7 * 7 * 256),
		BatchNorm(7 * 7 * 256, relu),
		x -> reshape(x, 7, 7, 256, :),
		ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2, init = dcgan_init),
		BatchNorm(128, relu),
		ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1, init = dcgan_init),
		BatchNorm(64, relu),
		ConvTranspose((4, 4), 64 => 1; stride = 2, pad = 1, init = dcgan_init),
		x -> tanh.(x),
	)
end



# Check device
if CUDA.functional()
	@info "Training on GPU"
else
	@warn "Training on CPU, this will be very slow!"
end

z_dim = 128
# Mean vector (zero vector of length dim)
mean_vector = zeros(z_dim)

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix = Diagonal(ones(z_dim))

# Create the multivariate normal distribution
noise_model = MvNormal(mean_vector, cov_matrix)

@with_kw struct HParams_sliced
	K::Int = 10
	m::Int = 10 # Number of random directions
	η::Float32 = 1e-2
	τ::Float32 = 1e-2
	epochs::Int = 100
	samples::Int = 1000
	latent_dim::Int = z_dim
	noise_model::Distribution = noise_model
end

hparams = HParams_sliced()


data = make_loader(hparams.samples)
#data = make_loader_FashionMNIST(hparams.samples)
batch_size = 200
loader = Flux.DataLoader(hcat(data...), batchsize = batch_size, shuffle = true, partial = false)


# ─── Instantiate & push your Generator to GPU ─────────────────────────────
latent_dim = 128
gen = gpu(Generator(latent_dim))

# ─── Call your multiscale ISL trainer ─────────────────────────────────────
Ks = [5, 10, 20]
losses = StatisticalLoss.invariant_statistical_loss_multiscale_gpu(gen, loader, hparams, Ks)

#Plot the numbers#
fixed_noise = [gpu(randn(Float32, hparams.latent_dim, 1)) for _ ∈ 1:9*9]
fake_images = @. cpu(gen(fixed_noise))
image_array = reduce(vcat, reduce.(hcat, partition(fake_images, 9)))
image_array = permutedims(dropdims(image_array; dims = (3, 4)), (2, 1))
image_array = @. Gray(image_array + 1.0f0) / 2.0f0
save("MNIST.pdf", image_array)
