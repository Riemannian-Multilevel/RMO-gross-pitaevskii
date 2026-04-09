//
// Created by Ferdinand Vanmaele on 09.04.26.
//
#include <gpe/problem/gpe.h>
#include <gpe/problem/oracle.h>
#include <gpe/option_types.h>

#include "test_gradient.h"

#define NUM_TRIALS 100
#define MIN_LEVEL  5
#define MAX_LEVEL  8
#define DIM        2
#define FE_DEGREE  1
#define RADIUS     10
#define BETA       100
#define MEAN       0.0
#define STDDEV     1.0

using namespace gpe;

int main()
{
    GPE_Options options = {.dimension=DIM, .degree=FE_DEGREE, .radius=RADIUS, .beta=BETA};

    for (unsigned level = MIN_LEVEL; level <= MAX_LEVEL; level++) {
        GrossPitaevskiiSimulator<DIM, EnergyOracle<DIM>> simulator(Square<DIM>(), options, level + 1);
        const auto& problem = simulator.get_problem();
        const unsigned n_dofs = simulator.n_dofs();

        for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
            Vector<double> x(n_dofs);
            gpe::ellipsoid::random_point(x, simulator.get_M(), MEAN, STDDEV);

            problem.assemble_nonlinear_term(x);
            double value = problem.value(x, options.beta);
        }
    }
}
