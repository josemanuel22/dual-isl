using CUDA
using MLDatasets
using Base.Iterators: partition
using LinearAlgebra
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
	epochs::Int = 10
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


function plot_generated_mnist(gen; n::Int = 25, z_dim::Int, noise_model)
	# 1) Sample latent vectors
	Z = rand(noise_model, z_dim, n)
	# If your generator is on the GPU, push Z to the same device:
	try
		Z = gpu(Z)
	catch
		# ignore if gen is on CPU
	end

	# 2) Generate images
	imgs = gen(Z)              # should be 28×28×1×n
	imgs = cpu(imgs)           # bring back to CPU if needed

	# 3) Plot a sqrt(n)×sqrt(n) grid
	s = Int(sqrt(n))
	plt = plot(layout = (s, s), margin = 1mm, ticks = false)

	for i in 1:n
		# squeeze to 28×28
		img = dropdims(imgs[:, :, 1, i]; dims = 3)
		heatmap!(plt[i], img;
			colorbar = false,
			aspect_ratio = 1,
			axis = false,
			c = :grays,
		)
	end

	display(plt)
end
