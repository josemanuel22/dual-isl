include("StatisticalLoss.jl")
#using StatisticalLoss
using Distributions
using Parameters: @with_kw
using Plots


#noise_model = Normal(0.0f0, 1.0f0)
noise_model = Normal(0.0f0, 1.0f0)
n_samples = 10000

gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
dscr = Chain(
	Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ),
)

@with_kw struct HParams
	K::Int = 10
	η::Float32 = 1e-2
	τ::Float32 = 1e-2
	epochs::Int = 1000
	samples::Int = 1000
	noise_model::Distribution = Normal(0.0f0, 1.0f0)
end

hparams = HParams()

target_model = MixtureModel([Normal(5.0f0, 2.0f0), Normal(-1.0f0, 1.0f0)])
train_set = Float32.(rand(target_model, hparams.samples))
loader = Flux.DataLoader(
	train_set; batchsize = 1000, shuffle = true, partial = false,
)


loss = StatisticalLoss.invariant_statistical_loss(gen, loader, hparams)


# 1) Draw one batch of “generated” data from your generator:
Z_gen = rand(hparams.noise_model, hparams.samples)'     # samples × 1
x_gen = vec(gen(Z_gen))                               # length = hparams.samples

# 2) Prepare a range for plotting the true density
all_data = vcat(train_set, x_gen)
xmin, xmax = extrema(all_data)
xs = range(xmin, xmax; length = 500)

# 3) Plot
histogram(train_set;
	bins = 50, normalize = :pdf,
	alpha = 0.5, label = "Real data",
)
histogram!(x_gen;
	bins = 50, normalize = :pdf,
	alpha = 0.5, label = "Generated",
)
plot!(xs, pdf.(target_model, xs);
	linewidth = 2, linestyle = :dashdot,
	label = "True PDF",
)
xlabel!("x")
ylabel!("Density")
title!("Real vs. Generated Samples")
