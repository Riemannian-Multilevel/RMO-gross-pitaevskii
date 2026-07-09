#!/usr/bin/env bash
set -xe
argv0=./main_coarse

# Problem parameters ----------------------------------------------------------------------------------------------------------
#beta_range=(100 1000 10000)
beta_range=(1000)
#iter_range=(20 40 60)
iter_range=(150)
kappa=0.5
eps=1e-4
#eps=1e-2
tol_res=1e-8
line_search=1
radius=11
boundary=dirichlet
potential=optical_lattice


# Arguments -------------------------------------------------------------------------------------------------------------------
args=(--potential "$potential" --boundary "$boundary" --radius "$radius")

if (( line_search )); then
    args+=(--line-search)
fi

# Coarse condition
args_fas=(--kappa "$kappa" --eps "$eps")

# Define file names
suffix=_optical_lattice.org
#suffix=.org


# Multilevel hierarchy --------------------------------------------------------------------------------------------------------
top_level=11
#top_level=10

declare -A levels
levels[2]="$top_level,$((top_level-1))"
#levels[2_mix]="$top_level,$((top_level-2))"
levels[3]="$top_level,$((top_level-1)),$((top_level-2))"
#levels[3_mix]="$top_level,$((top_level-2)),$((top_level-4))"
levels[4]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3))"
#levels[4_mix]="$top_level,$((top_level-2)),$((top_level-3)),$((top_level-4))"
#levels[5]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3)),$((top_level-4))"
#levels[6]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3)),$((top_level-4)),$((top_level-5))"
#levels[7]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3)),$((top_level-4)),$((top_level-5)),$((top_level-6))"


# Main loop -------------------------------------------------------------------------------------------------------------------
#total=$((${#beta_range[@]} * ${#levels[@]} * 8))
total=$((${#beta_range[@]} * ${#levels[@]} * 6))
#total=$((${#beta_range[@]} * ${#levels[@]} * 4))
count=0

for i in "${!beta_range[@]}"; do
    beta=${beta_range[$i]}
    iter=${iter_range[$i]}

    # Single level reference
    $argv0 "${args[@]}" --levels "$top_level" --metric none --max-iter 1000 --tol-residual 1e-14 --beta "$beta" \
	    >"sl_b${beta}_l${top_level}${suffix}"
    
    
    # Multilevel
    for depth in "${!levels[@]}"; do

        # Mass metric
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport mass \
            >"ml_mass_proj_b${beta}_l${top_level}_depth${depth}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport differential \
            >"ml_mass_diff_b${beta}_l${top_level}_depth${depth}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport adjoint_restriction \
            >"ml_mass_adj1_b${beta}_l${top_level}_depth${depth}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport adjoint_differential \
            >"ml_mass_adj2_b${beta}_l${top_level}_depth${depth}${suffix}"
        

        # Mass metric + Mass galerkin condition
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport mass --interpolate mass \
            >"ml_mass_proj_gal_b${beta}_l${top_level}_depth${depth}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport differential --interpolate mass \
            >"ml_mass_diff_gal_b${beta}_l${top_level}_depth${depth}${suffix}"
        

        # Euclidean metric
        # echo "[$(( ++count ))/$total]"
        # $argv0 "${args[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --beta "$beta" --metric frobenius --transport frobenius \
        #     >"ml_frob_proj_b${beta}_l${top_level}_depth${depth}${suffix}"
        
        # echo "[$(( ++count ))/$total]"
        # $argv0 "${args[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --beta "$beta" --metric frobenius --transport differential \
        #     >"ml_frob_diff_b${beta}_l${top_level}_depth${depth}${suffix}"
    done
done
