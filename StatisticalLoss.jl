module StatisticalLoss

using Flux
using Zygote
using Statistics
using Distributions
using StatsFuns: logistic
using ProgressMeter
using LinearAlgebra

export scalar_diff,
	invariant_statistical_loss,
	invariant_statistical_loss_with_monotonicity_penalty,
	sample_random_direction,
	F_model_smooth_sliced,
	invariant_statistical_loss_proj,
	invariant_statistical_loss_multiscale_gpu

# ————————————————————————————————————————————————————————————————
# compute ℓ² distance of a vector from the uniform “true” Q
@inline scalar_diff(q::AbstractArray{T}) where {T <: AbstractFloat} =
	sum((q .- (T(1.0f0) ./ T(length(q)))) .^ 2)

function F_model_smooth(x_gen::AbstractVector, x_eval::AbstractVector; τ::Real = 1e-2)
	N = length(x_gen)
	# pairwise (x_eval_i - x_gen_j)/τ
	# result is M x N matrix
	D = (x_eval .- x_gen') ./ τ
	K = logistic.(D)                    # σ(D[i,j]) ≈ 1 if x_gen[j] ≤ x_eval[i]
	return sum(K, dims = 2)[:, 1] ./ N     # average over model‐samples
end

# ————————————————————————————————————————————————————————————————
function invariant_statistical_loss(nn_model, loader, hparams)
	K = hparams.K
	τ = hparams.τ           # allow user to set smoothing in hparams
	opt = Flux.setup(Adam(hparams.η), nn_model)
	losses = Float32[]

	@showprogress for epoch in 1:hparams.epochs
		for data in loader
			loss, back = Flux.withgradient(nn_model) do m
				# 1) sample from model
				Z_gen = rand(hparams.noise_model, hparams.samples)'   # samples×1
				x_gen = vec(m(Z_gen))

				# 2) smooth‐CDF at real points
				u_ref = F_model_smooth(x_gen, vec(data); τ = τ)

				# 3) Monte​‐Carlo Q_K
				Q_K = [mean(binomial(K, m) .* u_ref .^ m .* (1 .- u_ref) .^ (K - m)) for m in 0:K]

				# 4) scalar difference loss
				scalar_diff(Q_K)
			end

			Flux.update!(opt, nn_model, back[1])
			push!(losses, loss)
		end
	end

	return losses
end

# avoid tracking ProgressMeter’s methods
Zygote.@nograd ProgressMeter.update!
Zygote.@nograd ProgressMeter.finish!

# ————————————————————————————————————————————————————————————————
function (
	nn_model, loader, hparams, λ,
)
	losses = Float64[]
	optim  = Flux.setup(Adam(hparams.η), nn_model)

	for data in loader
		loss, grads = Flux.withgradient(nn_model) do m
			# accumulate counts
			a_k = zeros(hparams.K + 1)
			num_blocks = fld(hparams.samples, hparams.K)

			# batch‐generate and evaluate
			inputs = [rand(hparams.noise_model) for _ in 1:num_blocks]
			outputs = m(inputs')

			# accumulate per‐block contributions
			for i in 1:num_blocks
				y_k       = outputs[i]
				start_idx = hparams.K * (i - 1) + 1
				end_idx   = hparams.K * i
				values    = reshape(data[start_idx:end_idx], hparams.K, 1)
				a_k       .+= generate_aₖ(values, y_k[1])
			end

			base_loss = scalar_diff(a_k ./ sum(a_k))

			# monotonicity penalty
			sorted_idx = sortperm(inputs)
			penalty = sum(max(0, outputs[sorted_idx[i]][1] -
								 outputs[sorted_idx[i+1]][1])
						  for i in 1:(length(sorted_idx)-1))
			base_loss + λ * (penalty / length(sorted_idx))
		end

		Flux.update!(optim, nn_model, grads[1])
		push!(losses, loss)
	end

	return losses
end

# ————————————————————————————————————————————————————————————————
@inline function sample_random_direction(n::Int)::Vector{Float32}
	v = randn(Float32, n)
	v ./ norm(v)
end

# ————————————————————————————————————————————————————————————————
function F_model_smooth_sliced(
	x_gen1d::AbstractVector{<:Real},
	x_eval1d::AbstractVector{<:Real};
	τ::Real = 1e-2,
)
	N = length(x_gen1d)
	M = length(x_eval1d)
	Xev = reshape(x_eval1d, M, 1)
	Xge = reshape(x_gen1d, 1, N)
	Kmat = σ.((Xev .- Xge) ./ τ)    # M×N
	vec(sum(Kmat, dims = 2)) ./ N
end

# ————————————————————————————————————————————————————————————————
function invariant_statistical_loss_proj(
	nn_model, loader, hparams,
)
	K = hparams.K
	τ = hparams.τ
	Ngen = hparams.samples
	mproj = hparams.m
	opt = Flux.setup(Adam(hparams.η), nn_model)
	losses = Float32[]

	@showprogress for epoch in 1:hparams.epochs
		for real_batch in loader
			x_real = Float32.(permutedims(real_batch))   # M×D_out

			loss, back = Flux.withgradient(nn_model) do m
				# generate & transform
				Z_gen = Float32.(rand(hparams.noise_model, Ngen))
				X_gen = m(Z_gen)
				x_gen = Float32.(permutedims(X_gen))      # Ngen×D_out

				total = 0.0f0
				for _ in 1:mproj
					ω      = sample_random_direction(size(x_gen, 2))
					s_gen  = x_gen * ω
					s_real = x_real * ω
					u_ref  = F_model_smooth_sliced(s_gen, s_real; τ = τ)
					Q_K    = [mean(binomial(K, j) .* u_ref .^ j .* (1 .- u_ref) .^ (K - j))
					for j in 0:K]
					total  += scalar_diff(Q_K)
				end
				total / mproj
			end
			Flux.update!(opt, nn_model, back[1])
			push!(losses, loss)
		end
	end
	return losses
end


function gram_schmidt(X::AbstractMatrix)
	# make sure X lives on the GPU
	Xg = gpu(X)
	n, m = size(Xg)

	# start with an empty n×0 CuArray
	Q = similar(Xg, n, 0)

	for i in 1:m
		# grab the i-th column (copy it so we can overwrite `v`)
		v = copy(@view(Xg[:, i]))

		# subtract off projections onto each previous q
		for j in 1:size(Q, 2)
			q = @view(Q[:, j])
			α = dot(q, v)               # runs on GPU
			v = v .- α .* q             # functional broadcast → new CuArray
		end

		# normalize
		nv = norm(v)                   # GPU norm, returns a scalar
		if nv > zero(eltype(v))
			v_norm = v ./ nv           # functional broadcast → new CuArray
			Q = hcat(Q, v_norm)        # still in‐device
		end
	end

	return Q   # n×r CuArray of orthonormal columns
end

function invariant_statistical_loss_multiscale_gpu(gen, loader, hparams, Ks)
	Ngen = hparams.samples
	m = hparams.m
	τ = Float32(1e-2)
	opt = Flux.setup(Adam(hparams.η), gen)
	losses = Float32[]

	# Precompute (K, ks, binoms) on CPU and move to GPU
	K_info = [
		(K,
			gpu(0:K),                                   # ks: 0…K on GPU
			gpu(Float32.(binomial.(K, collect(0:K)))))  # binoms on GPU
		for K in Ks
	]

	@showprogress for epoch in 1:hparams.epochs
		for real_batch in loader
			# real_batch :: Array{Float32,2} of shape (features, batch)
			# we need x_real of shape (batch, features):
			x_real = gpu(Float32.(permutedims(real_batch)))

			loss, back = Flux.withgradient(gen) do g
				# 1) sample noise and send to GPU
				Z = gpu(Float32.(rand(hparams.noise_model, Ngen)))  # (latent_dim × Ngen)

				# 2) forward pass → raw is H×W×C×Ngen
				raw = g(Z)

				# 3) flatten spatial dims into features, then transpose to Ngen×features
				flat = reshape(raw, :, size(raw, 4))  # (features × Ngen)
				Xg   = flat'                           # (Ngen × features)

				# 4) build random basis in feature-space and orthonormalize
				R = gpu(randn(Float32, size(Xg, 2), m))  # (features × m)
				V = gram_schmidt(R)                      # (features × m)

				total = zero(Float32)

				# 5) loop over each projection direction ω ∈  ℝ^features
				for ω in eachcol(V)
					s_real = x_real * ω    # (batch) projection of real data
					s_gen  = Xg * ω   # (Ngen) projection of generated data

					# 6) compute smoothed slice values u ∈ [0, 1]
					u = F_model_smooth_sliced(s_gen, s_real; τ = τ)
					u = clamp.(u, eps(Float32), 1.0f0 - eps(Float32))

					# 7) for each K accumulate the moment‐error
					for (K, ks, binoms) in K_info
						Umat  = u .^ ks'                               # n×(K+1)
						Ūmat  = (1.0f0 .- u) .^ (K .- ks)'               # n×(K+1)
						P     = Umat .* Ūmat .* reshape(binoms, 1, :)   # n×(K+1)
						Q_gpu = sum(P, dims = 1) ./ Float32(length(u))  # 1×(K+1)
						Q     = vec(Q_gpu)                            # (K+1)
						total += sum(abs.(Q .- 1.0f0 / (K + 1)))
					end
				end

				return total / (length(Ks) * m)
			end

			Flux.update!(opt, gen, back[1])
			push!(losses, loss)
		end
	end
	return losses
end


end # module
