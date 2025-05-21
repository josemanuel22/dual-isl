include("StatisticalLoss.jl")
#using StatisticalLoss
using Distributions
using Parameters: @with_kw
using Plots
using LinearAlgebra
using KernelDensity


##Generate 2D data##

abstract type Target end

mutable struct TwoMoons <: Target
	prop_scale::Float32
	prop_shift::Float32
	n_dims::Int
	max_log_prob::Float32

	TwoMoons() = new(6.0, -3.0, 2, 0.0)  # Default values for Two Moons distribution
end

function log_prob(z::AbstractArray)
	a = abs.(z[:, 1])
	norm_z = vec(sqrt.(sum(z .^ 2; dims = 2)))
	return -0.5 .* ((norm_z .- 2) ./ 0.2) .^ 2 .- 0.5 .* ((a .- 2) ./ 0.3) .^ 2 .+
		   log.(1 .+ exp.(-4 .* a ./ 0.09))
end

function rejection_sampling(model::TwoMoons, num_steps::Int)
	eps = rand(Float32, (num_steps, model.n_dims))
	z_ = model.prop_scale .* eps .+ model.prop_shift
	prob = rand(Float32, num_steps)
	prob_ = exp.(log_prob(z_) .- model.max_log_prob)
	accept = prob_ .> prob
	z = z_[accept, :]
	return z
end

function sample(model::TwoMoons, num_samples::Int)
	z = Array{Float32}(undef, 0, model.n_dims)  # Initialize z as an empty 2D array with 0 rows and model.n_dims columns
	while size(z, 1) < num_samples
		z_ = rejection_sampling(model, num_samples)
		ind = min(size(z_, 1), num_samples - size(z, 1))
		z = vcat(z, z_[1:ind, :])
	end
	return z
end


z_dim = 2
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
	noise_model::Distribution = noise_model
end

hparams = HParams_sliced()

hidden_dim = 32
gen = Chain(
	Dense(z_dim, hidden_dim, relu),
	Dense(hidden_dim, hidden_dim, relu),
	#Dense(hidden_dim, hidden_dim, x -> rbf(x, 0.0f0, 1.0f0)),
	Dense(hidden_dim, hidden_dim, relu),
	Dense(hidden_dim, 2),
)

d = TwoMoons()
moons = permutedims(Float32.(sample(d, 10000)))
batch_size = 1000
train_loader = Flux.DataLoader(Float32.(moons); batchsize = batch_size, shuffle = true, partial = false)

StatisticalLoss.invariant_statistical_loss_proj(gen, train_loader, hparams)



# === Plotting ===
# draw 10k new generator samples
Z_plot = Float32.(rand(hparams.noise_model, 10000))
X_gen = gen(Z_plot)'              # 10000×2 array

# true samples already in `moons` as 2×N; transpose to N×2
X_true = permutedims(moons)

# scatter plot
scatter(
	X_true[:, 1], X_true[:, 2];
	markersize = 2, alpha = 0.4,
	label = "True Two-Moons",
)
scatter!(
	X_gen[:, 1], X_gen[:, 2];
	markersize = 2, alpha = 0.4,
	label = "Generator samples",
)

kde_gen = kde((X_gen[:, 2], X_gen[:, 1]))
contour(
	kde_gen.x, kde_gen.y, kde_gen.density;
	levels    = 6,
	linewidth = 2,
	linestyle = :dash,
	label     = "Gen KDE",
)
