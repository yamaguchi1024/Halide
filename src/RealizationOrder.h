#ifndef HALIDE_INTERNAL_REALIZATION_ORDER_H
#define HALIDE_INTERNAL_REALIZATION_ORDER_H

/** \file
 *
 * Defines the lowering pass that determines the order in which
 * realizations are injected and groups functions with fused
 * computation loops.
 */

#include <vector>
#include <string>
#include <map>

namespace Halide {
namespace Internal {

class Function;

/** Given a bunch of functions that call each other, determine an
 * order in which to do the scheduling. This in turn influences the
 * order in which stages are computed when there's no strict
 * dependency between them. Currently just some arbitrary depth-first
 * traversal of the call graph. In addition, determine grouping of functions
 * with fused computation loops. The functions within the fused groups
 * are sorted based on realization order. There should not be any dependencies
 * among functions within a fused group. This pass will also populate the
 * 'fused_pairs' list in the function's schedule. Return a pair of
 * the realization order and the fused groups in that order.
 */
std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>> realization_order(
	const std::vector<Function> &output, std::map<std::string, Function> &env);

}
}

#endif
