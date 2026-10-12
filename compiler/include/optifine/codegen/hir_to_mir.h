/* Workload HIR (ir.h) -> generic MIR (mir.h).
 *
 * The one place where workload semantics meet the generic representation.
 * Both workload paths -- the ONNX classifier and the DSP pipeline -- reach
 * the AVR backend only through the module built here:
 *
 *   memory   every HIR op's output tensor is a MIR_MEM_TENSOR object ("t<id>");
 *            the DSP ops' scratch arena is a MIR_MEM_SCRATCH object; the FFT
 *            twiddle table is a MIR_MEM_CONST object whose name is its label
 *            (DSP_TWIDDLE_LABEL), emitted by the backend as program-memory data
 *   initialize  a function storing the embedded input (OP_INPUT) and every
 *            constant (OP_CONST) into their tensors, as MIR stores
 *   infer    a function computing the rest in graph order: OP_OUTPUT as MIR
 *            loads and stores; every other op as a MIR_TARGET region holding
 *            the hand-scheduled AVR kernel from lower_op (or, on the
 *            optimized path, the cheapest of candidates_generate's
 *            candidates), with the tensors, scratch and table it reads and
 *            writes declared
 *
 * Every MIR instruction's `origin` is the HIR op id it came from, which is
 * how the program emitter keeps per-op cost accounting.
 *
 * What stays above MIR: workload semantics (shapes, Q15 and quantization
 * arithmetic) and the kernels' own instruction scheduling, including their
 * counted loops, which exist only inside the target regions and the
 * InstrBuf loop metadata that prices them. docs/ARCHITECTURE.md lists this as
 * the remaining coupling between the workload kernels and the AVR target. */
#ifndef OPTIFINE_CODEGEN_HIR_TO_MIR_H
#define OPTIFINE_CODEGEN_HIR_TO_MIR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "optifine/codegen/avr_mir.h"
#include "optifine/codegen/candidates.h"
#include "optifine/codegen/reuse_analysis.h"
#include "optifine/codegen/sram_layout.h"
#include "optifine/cost_model.h"
#include "optifine/ir.h"
#include "optifine/mir.h"

typedef struct {
    MirModule module;
    uint32_t initialize; /* MIR_NONE unless requested */
    uint32_t infer;      /* MIR_NONE unless requested */
    uint32_t *tensor;    /* per HIR op id: its tensor object */
    size_t num_tensors;
    uint32_t scratch;    /* MIR_NONE for a graph without DSP ops */
    uint32_t twiddle;    /* MIR_NONE for a graph without FFT stages */
} WorkloadMir;

enum { HIR_TO_MIR_INITIALIZE = 1, HIR_TO_MIR_INFER = 2 };

/* Builds the module for a verified graph, with the functions named in
 * `functions` (memory objects are always built). `input` is embedded by
 * OP_INPUT and must match its tensor's byte size. `use_real_candidates`
 * selects the optimized MatMul lowering. Returns 0 (with a verified module)
 * or -1 with a diagnostic. */
int hir_to_mir(const IrGraph *graph, const SramLayout *layout, const ReuseAnalysis *reuse,
               const CostModel *cost_model, const int8_t *input, size_t input_len, int use_real_candidates,
               unsigned functions, WorkloadMir *out, bool dump_mir);
void workload_mir_free(WorkloadMir *mir);

/* The backend layout for a workload module: tensors and scratch at the
 * addresses sram_layout chose. Fails if any MIR value would need SRAM of its
 * own, which the fixed workload layout does not provide. */
int workload_mir_layout(const WorkloadMir *mir, const SramLayout *layout, AvrMirLayout *out);

/* lower_op's OP_INPUT, OP_CONST and OP_OUTPUT: one op through the same MIR
 * translation and AVR selection, as a single priced Candidate. */
int hir_lower_data_movement(const IrGraph *graph, size_t op_id, const SramLayout *layout,
                            const CostModel *cost_model, const int8_t *input, size_t input_len,
                            Candidate *out);

#endif /* OPTIFINE_CODEGEN_HIR_TO_MIR_H */
