using Revise

includet("Miniopoly.jl")

using ..Miniopoly
using Logging, DataFrames, CSV
using ProgressMeter

# Disabling logging for performance reasons
Logging.disable_logging(Logging.Info)

function one_game()
	broke = false
	gm = newgame(2, 1_500)

	while !broke
		broke = turn!(gm)
	end

	return gm
end

# Running many simulations
function run_simulation(n)
	results = DataFrame(
		mine=Vector{Bool}(),
		square=Vector{Int}(),
		reward=Vector{Float64}()
	)

	@showprogress "Simulating" for _ = 1:n
		gm = one_game()

		for player in gm.players
			log = player.rewardslog

			for (sq_info, reward) in log
				sq_is_mine, sq_num = sq_info
				push!(results, [sq_is_mine, sq_num, reward])
			end
		end
	end

	return results
end