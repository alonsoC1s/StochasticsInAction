using JuMP, GLPK
using Distributions
using LinearAlgebra
using Plots

struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
    TR # sample transition and reward
end

MDP(γ, 𝒮, 𝒜, T, R) = MDP(γ, 𝒮, 𝒜, T, R, nothing)

function MDP(T::Array{Float64,3}, R::Array{Float64,2}, γ::Float64)
    MDP(γ, 1:size(R, 1), 1:size(R, 2), (s, a, s′) -> T[s, a, s′], (s, a) -> R[s, a], nothing)
end

function MDP(T::Array{Float64,3}, R, γ::Float64)
    MDP(γ, 1:size(T, 1), 1:size(T, 2), (s, a, s′) -> T[s, a, s′], R, nothing)
end

struct MDPInitialStateDistribution{MDP}
    mdp::MDP
end

Base.rand(S::MDPInitialStateDistribution) = generate_start_state(S.mdp)

struct DiscreteMDP
    # TODO: Use sparse matrices?
    T::Array{Float64,3} # T(s,a,s′)
    R::Array{Float64,2} # R(s,a) = ∑_s' R(s,a,s')*T(s,a,s′)
    γ::Float64
end

n_states(mdp::DiscreteMDP) = size(mdp.T, 1)
n_actions(mdp::DiscreteMDP) = size(mdp.T, 2)
discount(mdp::DiscreteMDP) = mdp.γ
ordered_states(mdp::DiscreteMDP) = collect(1:n_states(mdp))
ordered_actions(mdp::DiscreteMDP) = collect(1:n_actions(mdp))
state_index(mdp::DiscreteMDP, s::Int) = s

function transition(mdp::DiscreteMDP, s::Int, a::Int)
    return Categorical(mdp.T[s, a, :])
end
function generate_s(mdp::DiscreteMDP, s::Int, a::Int)
    s′ = 1
    r = rand() - mdp.T[s, a, s′]
    while r > 0.0 && s′ < size(mdp.T, 3)
        s′ += 1
        r -= mdp.T[s, a, s′]
    end
    return s′
end

reward(mdp::DiscreteMDP, s::Int, a::Int) = mdp.R[s, a]

function MDP(mdp::DiscreteMDP; γ::Float64=discount(mdp))
    return MDP(
        γ,
        ordered_states(mdp),
        ordered_actions(mdp),
        (s, a, s′=nothing) -> begin
            S′ = transition(mdp, s, a)
            if s′ == nothing
                return S′
            end
            return pdf(S′, s′)
        end,
        (s, a) -> reward(mdp, s, a),
        (s, a) -> begin
            s′ = rand(transition(mdp, s, a))
            r = reward(mdp, s, a)
            return (s′, r)
        end
    )
end

function hex_neighbors(hex::Tuple{Int,Int})
    i, j = hex
    [(i + 1, j), (i, j + 1), (i - 1, j + 1), (i - 1, j), (i, j - 1), (i + 1, j - 1)]
end

struct HexWorldMDP
    # Problem has |hexes| + 1 states, where last state is consuming.
    hexes::Vector{Tuple{Int,Int}}

    # The exact same problem as a DiscreteMDP
    mdp::DiscreteMDP

    # The special hex rewards used to construct the MDP
    special_hex_rewards::Dict{Tuple{Int,Int},Float64}

    function HexWorldMDP(
        hexes::Vector{Tuple{Int,Int}},
        r_bump_border::Float64,
        p_intended::Float64,
        special_hex_rewards::Dict{Tuple{Int,Int},Float64},
        γ::Float64,
    )

        nS = length(hexes) + 1 # Hexes plus one terminal state
        nA = 6 # Six directions. 1 is east, 2 is north east, 3 is north west, etc.
        # As enumerated in hex_neighbors.

        s_absorbing = nS

        T = zeros(Float64, nS, nA, nS)
        R = zeros(Float64, nS, nA)

        p_veer = (1.0 - p_intended) / 2 # Odds of veering left or right.

        for s in 1:length(hexes)
            hex = hexes[s]
            if !haskey(special_hex_rewards, hex)
                # Action taken from a normal tile
                neighbors = hex_neighbors(hex)
                for (a, neigh) in enumerate(neighbors)
                    # Indended transition.
                    s′ = findfirst(h -> h == neigh, hexes)
                    if s′ == nothing
                        # Off the map!
                        s′ = s
                        R[s, a] += r_bump_border * p_intended
                    end
                    T[s, a, s′] += p_intended

                    # Unintended veer left.
                    a_left = mod1(a + 1, nA)
                    neigh_left = neighbors[a_left]
                    s′ = findfirst(h -> h == neigh_left, hexes)
                    if s′ == nothing
                        # Off the map!
                        s′ = s
                        R[s, a] += r_bump_border * p_veer
                    end
                    T[s, a, s′] += p_veer

                    # Unintended veer right.
                    a_right = mod1(a - 1, nA)
                    neigh_right = neighbors[a_right]
                    s′ = findfirst(h -> h == neigh_right, hexes)
                    if s′ == nothing
                        # Off the map!
                        s′ = s
                        R[s, a] += r_bump_border * p_veer
                    end
                    T[s, a, s′] += p_veer
                end
            else
                # Action taken from an absorbing hex
                # In absorbing hex, your action automatically takes you to the absorbing state and you get the reward.
                for a in 1:nA
                    T[s, a, s_absorbing] = 1.0
                    R[s, a] += special_hex_rewards[hex]
                end
            end
        end

        # Absorbing state stays where it is and gets no reward.
        for a in 1:nA
            T[s_absorbing, a, s_absorbing] = 1.0
        end

        mdp = DiscreteMDP(T, R, γ)

        return new(hexes, mdp, special_hex_rewards)
    end
end

const HexWorldRBumpBorder = -1.0 # Reward for falling off hex map
const HexWorldPIntended = 0.7 # Probability of going intended direction
const HexWorldDiscountFactor = 0.9

function HexWorld()
    HexWorld = HexWorldMDP(
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (-1, 2),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
            (8, 2), (4, 1), (5, 0), (6, 0), (7, 0), (7, 1), (8, 1), (9, 0)],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        Dict{Tuple{Int,Int},Float64}(
            (0, 1) => 5.0, # left side reward
            (2, 0) => -10.0, # left side hazard
            (9, 0) => 10.0, # right side reward
        ),
        HexWorldDiscountFactor
    )
    return HexWorld
end

# function StraightLineHexWorld()
#     StraightLineHexWorld = HexWorldMDP(
#         [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (-1, 2),
#             (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
#             (8, 2), (4, 1), (5, 0), (6, 0), (7, 0), (7, 1), (8, 1), (9, 0)],
#         HexWorldRBumpBorder,
#         HexWorldPIntended,
#         Dict{Tuple{Int,Int},Float64}(
#             (0, 1) => 5.0, # left side reward
#             (2, 0) => -10.0, # left side hazard
#             (9, 0) => 10.0, # right side reward
#         ),
#         HexWorldDiscountFactor
#     )
#     return StraightLineHexWorld
# end
function StraightLineHexWorld()
    StraightLineHexWorld = HexWorldMDP(
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        Dict{Tuple{Int,Int},Float64}(
            (7, 0) => 10.0, # right side reward
        ),
        HexWorldDiscountFactor
    )
    return StraightLineHexWorld
end

n_states(mdp::HexWorldMDP) = n_states(mdp.mdp)
n_actions(mdp::HexWorldMDP) = n_actions(mdp.mdp)
discount(mdp::HexWorldMDP) = discount(mdp.mdp)
ordered_states(mdp::HexWorldMDP) = ordered_states(mdp.mdp)
ordered_actions(mdp::HexWorldMDP) = ordered_actions(mdp.mdp)
state_index(mdp::HexWorldMDP, s::Int) = s

transition(mdp::HexWorldMDP, s::Int, a::Int) = transition(mdp.mdp, s, a)
generate_s(mdp::HexWorldMDP, s::Int, a::Int) = generate_s(mdp.mdp, s, a)
reward(mdp::HexWorldMDP, s::Int, a::Int) = reward(mdp.mdp, s, a)
generate_sr(mdp::HexWorldMDP, s::Int, a::Int) = (generate_s(mdp, s, a), reward(mdp, s, a))

generate_start_state(mdp::HexWorldMDP) = rand(1:(n_states(mdp)-1)) # non-terminal state

function hex_distance(a::Tuple{Int,Int}, b::Tuple{Int,Int})
    az = -a[1] - a[2]
    bz = -b[1] - b[2]
    return max(abs(a[1] - b[1]), abs(a[2] - b[2]), abs(az - bz))
end

function DiscreteMDP(mdp::HexWorldMDP)
    return mdp.mdp
end
function MDP(mdp::HexWorldMDP)
    return MDP(mdp.mdp)
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end
function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end

function greedy(𝒫::MDP, U, s)
    u, a = findmax(a -> lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a

struct LinearProgramFormulation end

function tensorform(𝒫::MDP)
    𝒮, 𝒜, R, T = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T
    𝒮′ = eachindex(𝒮)
    𝒜′ = eachindex(𝒜)
    R′ = [R(s, a) for s in 𝒮, a in 𝒜]
    T′ = [T(s, a, s′) for s in 𝒮, a in 𝒜, s′ in 𝒮]
    return 𝒮′, 𝒜′, R′, T′
end

solve(𝒫::MDP) = solve(LinearProgramFormulation(), 𝒫)

function solve(M::LinearProgramFormulation, 𝒫::MDP)
    𝒮, 𝒜, R, T = tensorform(𝒫)
    model = Model(GLPK.Optimizer)
    @variable(model, U[𝒮])
    @objective(model, Min, sum(U))
    @constraint(model, [s = 𝒮, a = 𝒜], U[s] ≥ R[s, a] + 𝒫.γ * T[s, a, :] ⋅ U)
    optimize!(model)
    return U, ValueFunctionPolicy(𝒫, value.(U))
end

slhw = StraightLineHexWorld()
v_star, πolicy = solve(LinearProgramFormulation(), slhw |> MDP)

hexes = reinterpret(reshape, Int, slhw.hexes) |> transpose |> Matrix

scatter(eachcol(hexes)...,
    marker = :hex,
    markersize = 10,
    zcolor = value.(v_star)
)

