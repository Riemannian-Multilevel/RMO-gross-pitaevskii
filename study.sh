#!/bin/bash
set -xe
argv0=./main_coarse

beta_range=(100 1000 10000)
iter_range=(20 40 60)
top_level=11
kappa=0.5
line_search=1

args=(--kappa "$kappa")

if (( line_search )); then
    args+=(--line-search)
fi

for i in "${!beta_range[@]}"; do
    beta=${beta_range[$i]}
    iter=${iter_range[$i]}

    # Reference (gradient descent)
    levels=(--levels "$top_level" --min-level "$top_level")

    $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric none \
	    >"sl_b${beta}_l${top_level}.org"
    
    #for depth in (2 3 4); do
    for depth in 4; do
        levels=(--levels "$top_level" --min-level "$((top_level-depth+1))")

        # Mass metric
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport mass \
            >"ml_mass_proj_b${beta}_l${top_level}_depth${depth}.org"
        
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport differential \
            >"ml_mass_diff_b${beta}_l${top_level}_depth${depth}.org"
        
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport adjoint_restriction \
            >"ml_mass_adj1_b${beta}_l${top_level}_depth${depth}.org"
        
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport adjoint_restriction_scaled \
            >"ml_mass_adj2_b${beta}_l${top_level}_depth${depth}.org"
        
        # Mass metric + Mass galerkin condition
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport mass --interpolate mass \
            >"ml_mass_proj_gal_b${beta}_l${top_level}_depth${depth}.org"
        
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric mass --transport differential --interpolate mass \
            >"ml_mass_diff_gal_b${beta}_l${top_level}_depth${depth}.org"
        
        # Euclidean metric
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric frobenius --transport frobenius \
            >"ml_frob_proj_b${beta}_l${top_level}_depth${depth}.org"
        
        $argv0 "${args[@]}" "${levels[@]}" --max-iter "$iter" --beta "$beta" --metric frobenius --transport differential \
            >"ml_frob_diff_b${beta}_l${top_level}_depth${depth}.org"
    done
done
