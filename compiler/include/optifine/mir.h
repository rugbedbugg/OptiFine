/* OptiFine MIR: the generic, target-independent program representation.
 *
 * The workload graph (ir.h, "HIR") describes ML and DSP operators; it has no
 * control flow and cannot express an arbitrary imperative program. MIR is the
 * layer below it that can:
 *
 *   MirModule    memory objects + functions
 *   MirFunction  typed virtual registers ("values"), parameters, a return
 *                type, and basic blocks; block 0 is the entry
 *   MirBlock     straight-line instructions, then exactly one terminator
 *   MirInst      opcode over typed values, immediates and memory addresses
 *   MirTerminator  br / cbr / ret: the CFG edges
 *
 * Values are virtual registers, NOT SSA: a value may be assigned in more
 * than one place (a loop counter is), and every use must be preceded by an
 * assignment on every path from the entry (mir_verify checks this with a
 * must-be-defined dataflow). Converting to SSA -- phis, dominance -- is left
 * for the frontend work that first needs it; nothing here prevents it.
 *
 * Memory is a set of objects with a kind and a size but no address. Address
 * assignment is the backend's job (codegen/avr_mir.h): the AVR backend places
 * SRAM objects with an allocation strategy and program-memory constants after
 * the code. Loads and stores name an object plus a constant byte offset, or a
 * pointer value plus an offset.
 *
 * Workload operators are deliberately absent. There is no MatMul or FFT
 * opcode: workload HIR is translated to MIR by hir_to_mir (codegen/
 * hir_to_mir.h), where the data-movement ops become ordinary loads and
 * stores, and kernels that only exist as hand-scheduled AVR sequences enter
 * as MIR_TARGET regions -- opaque target code with declared memory effects,
 * the same role inline assembly plays in a C compiler.
 *
 * ---- Frontend contract (for the planned EDG IL adapter) ----
 *
 * A frontend builds a MirModule with this API and nothing else, calls
 * mir_verify for its own diagnostics, and hands the module to the backend
 * through avr_mir_build_program (codegen/avr_mir.h), which verifies it again
 * and refuses invalid MIR -- the backend never relies on the caller having
 * verified. It links optifine_backend only. In particular:
 *   - No frontend type crosses this boundary. EDG IL nodes, types and
 *     headers stay inside the adapter; MIR and everything below it must
 *     build without them.
 *   - Arbitrary C control flow is expressed as blocks and br/cbr/ret, never
 *     by forcing it through the workload graph (ir.h), which has no control
 *     flow and whose operators are workload kernels.
 *   - Globals, locals and string/constant data become MIR memory objects of
 *     the matching kind; the frontend never chooses an address.
 *   - What MIR does not yet model -- calls and a calling convention, address
 *     spaces for pointers into program memory, floating point, aggregates
 *     passed by value, initialized globals -- must be rejected by the
 *     adapter, not approximated. A program is one entry function under the
 *     research entry convention in avr_mir.h (parameters and result at named
 *     SRAM symbols), not a C ABI.
 * docs/ARCHITECTURE.md describes the boundary in full. */
#ifndef OPTIFINE_MIR_H
#define OPTIFINE_MIR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MIR_NONE UINT32_MAX

typedef enum {
    MIR_TYPE_VOID, /* function return type only */
    MIR_TYPE_I8,
    MIR_TYPE_I16,
    MIR_TYPE_I32,
    MIR_TYPE_PTR, /* data pointer; its width is the target's (2 bytes on AVR) */
} MirType;

/* Integer width in bytes (1, 2, 4), or 0 for VOID and PTR, whose size is the
 * target's. */
size_t mir_int_bytes(MirType type);

typedef enum {
    MIR_MEM_GLOBAL,  /* static storage for the whole program */
    MIR_MEM_STACK,   /* storage local to one function */
    MIR_MEM_SCRATCH, /* compiler-generated temporary storage */
    MIR_MEM_CONST,   /* read-only data with an initializer (program memory on AVR) */
    MIR_MEM_TENSOR,  /* a workload tensor from the HIR */
} MirMemKind;

typedef struct {
    MirMemKind kind;
    size_t size;      /* bytes, > 0 */
    char name[32];    /* unique in the module; also its assembly label for CONST */
    uint8_t *init;    /* MIR_MEM_CONST: exactly `size` bytes; NULL otherwise */
    uint32_t function; /* MIR_MEM_STACK: owning function; MIR_NONE otherwise */
} MirMemObject;

typedef enum {
    MIR_OPND_NONE,
    MIR_OPND_VALUE, /* a virtual register */
    MIR_OPND_IMM,   /* an integer constant of the instruction's type */
} MirOperandKind;

typedef struct {
    MirOperandKind kind;
    uint32_t value;
    int64_t imm;
} MirOperand;

/* A memory address: object + offset, or the value of a PTR register +
 * offset. */
typedef struct {
    uint32_t object;  /* MIR_NONE when based on `pointer` */
    uint32_t pointer; /* a PTR value, or MIR_NONE when based on `object` */
    int32_t offset;
} MirAddress;

typedef enum {
    MIR_CONST,   /* dst = a (immediate) */
    MIR_COPY,    /* dst = a */
    MIR_ADD,     /* dst = a + b, wrapping */
    MIR_SUB,     /* dst = a - b, wrapping */
    MIR_MUL,     /* dst = a * b, low bits */
    MIR_AND,
    MIR_OR,
    MIR_XOR,
    MIR_CMP,     /* dst (I8) = a <pred> b ? 1 : 0; a and b have type `type` */
    MIR_ZEXT,    /* dst (type) = a (a narrower integer value), zero-extended */
    MIR_SEXT,    /* dst (type) = a (a narrower integer value), sign-extended */
    MIR_TRUNC,   /* dst (type) = low bytes of a (a wider integer value) */
    MIR_LOAD,    /* dst = *(type *)addr */
    MIR_STORE,   /* *(type *)addr = a */
    MIR_ADDR,    /* dst (PTR) = address of addr.object + addr.offset */
    MIR_PTR_ADD, /* dst (PTR) = a (PTR) + b (I16, a byte offset) */
    MIR_TARGET,  /* opaque target code; see MirTargetCode */
} MirOpcode;

typedef enum { MIR_CMP_EQ, MIR_CMP_NE, MIR_CMP_ULT, MIR_CMP_SLT } MirCmpPred;

/* Target code that MIR does not interpret: a pre-selected instruction
 * sequence for one named target, plus the objects it reads and writes so the
 * module stays verifiable. MIR owns it and frees the payload with
 * `free_payload`. */
typedef struct {
    const char *target; /* e.g. "avr"; a backend refuses any other */
    void *payload;
    void (*free_payload)(void *payload);
    uint32_t *reads;
    size_t num_reads;
    uint32_t *writes;
    size_t num_writes;
} MirTargetCode;

typedef struct {
    MirOpcode op;
    MirType type;     /* the operation's type (for MIR_STORE, the stored type) */
    uint32_t dst;     /* result value, or MIR_NONE */
    MirOperand a, b;
    MirCmpPred pred;  /* MIR_CMP */
    MirAddress addr;  /* MIR_LOAD, MIR_STORE, MIR_ADDR */
    MirTargetCode *target; /* MIR_TARGET */
    uint32_t origin;  /* provenance: what produced it (e.g. an HIR op id); free-form */
} MirInst;

typedef enum { MIR_TERM_NONE, MIR_TERM_BR, MIR_TERM_CBR, MIR_TERM_RET } MirTermKind;

typedef struct {
    MirTermKind kind;      /* MIR_TERM_NONE means "not yet terminated" */
    MirOperand cond;       /* MIR_TERM_CBR: an I8 value; nonzero takes then_block */
    uint32_t then_block;   /* MIR_TERM_BR target, or MIR_TERM_CBR's taken edge */
    uint32_t else_block;   /* MIR_TERM_CBR's other edge */
    MirOperand value;      /* MIR_TERM_RET: the returned value, NONE for void */
} MirTerminator;

typedef struct {
    MirInst *insts;
    size_t count, capacity;
    MirTerminator term;
} MirBlock;

typedef struct {
    char name[32];
    MirType return_type;
    MirType *value_types; /* indexed by value id */
    size_t num_values, values_capacity;
    size_t num_params;    /* values 0 .. num_params-1 are the parameters */
    MirBlock *blocks;
    size_t num_blocks, blocks_capacity;
} MirFunction;

typedef struct {
    MirMemObject *objects;
    size_t num_objects, objects_capacity;
    MirFunction *functions;
    size_t num_functions, functions_capacity;
    int out_of_memory; /* sticky: set by any builder call that could not allocate */
} MirModule;

/* ---- Construction ----
 *
 * Builder calls return the new id, or MIR_NONE if allocation failed (which
 * also sets module->out_of_memory; mir_verify then rejects the module). */
void mir_module_init(MirModule *module);
void mir_module_free(MirModule *module);

uint32_t mir_add_object(MirModule *module, MirMemKind kind, size_t size, const char *name, uint32_t function);
/* Copies `size` bytes of initializer into a MIR_MEM_CONST object. Returns 0
 * or -1. */
int mir_object_set_init(MirModule *module, uint32_t object, const uint8_t *bytes, size_t size);

uint32_t mir_add_function(MirModule *module, const char *name, MirType return_type);
/* Parameters must be added before any other value. */
uint32_t mir_add_param(MirModule *module, uint32_t function, MirType type);
uint32_t mir_new_value(MirModule *module, uint32_t function, MirType type);
uint32_t mir_add_block(MirModule *module, uint32_t function);

/* Appends `inst` to the block. Returns 0 or -1 (out of memory). */
int mir_append(MirModule *module, uint32_t function, uint32_t block, const MirInst *inst);
void mir_set_terminator(MirModule *module, uint32_t function, uint32_t block, const MirTerminator *term);

/* Operand and instruction shorthands. */
MirOperand mir_none(void);
MirOperand mir_value(uint32_t value);
MirOperand mir_imm(int64_t imm);
MirAddress mir_at_object(uint32_t object, int32_t offset);
MirAddress mir_at_pointer(uint32_t pointer, int32_t offset);

int mir_emit_const(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, MirType type, int64_t imm,
                   uint32_t origin);
int mir_emit_binary(MirModule *m, uint32_t fn, uint32_t block, MirOpcode op, uint32_t dst, MirType type,
                    MirOperand a, MirOperand b, uint32_t origin);
/* MIR_ZEXT, MIR_SEXT or MIR_TRUNC of value `a` to `type`. */
int mir_emit_convert(MirModule *m, uint32_t fn, uint32_t block, MirOpcode op, uint32_t dst, MirType type,
                     uint32_t a, uint32_t origin);
int mir_emit_cmp(MirModule *m, uint32_t fn, uint32_t block, MirCmpPred pred, uint32_t dst, MirType type,
                 MirOperand a, MirOperand b, uint32_t origin);
int mir_emit_load(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, MirType type, MirAddress addr,
                  uint32_t origin);
int mir_emit_store(MirModule *m, uint32_t fn, uint32_t block, MirType type, MirAddress addr, MirOperand value,
                   uint32_t origin);
int mir_emit_addr(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, uint32_t object, int32_t offset,
                  uint32_t origin);
/* Takes ownership of `code` (heap-allocated) whether or not it succeeds. */
int mir_emit_target(MirModule *m, uint32_t fn, uint32_t block, MirTargetCode *code, uint32_t origin);

void mir_br(MirModule *m, uint32_t fn, uint32_t block, uint32_t target);
void mir_cbr(MirModule *m, uint32_t fn, uint32_t block, MirOperand cond, uint32_t then_block,
             uint32_t else_block);
void mir_ret(MirModule *m, uint32_t fn, uint32_t block, MirOperand value);

/* Frees a MirTargetCode and its payload. */
void mir_target_code_free(MirTargetCode *code);

/* ---- Fields each opcode uses ----
 *
 *   opcode                     dst  a              b              addr  target
 *   CONST                      yes  immediate      -              -     -
 *   COPY                       yes  value/imm      -              -     -
 *   ADD SUB MUL AND OR XOR CMP yes  value/imm      value/imm      -     -
 *   ZEXT SEXT TRUNC            yes  value          -              -     -
 *   LOAD, ADDR                 yes  -              -              yes   -
 *   STORE                      -    value/imm      -              yes   -
 *   PTR_ADD                    yes  value (PTR)    value/imm      -     -
 *   TARGET                     -    -              -              -     yes
 *
 *   terminator  cond         then_block  else_block  value
 *   br          -            yes         -           -
 *   cbr         value (I8)   yes         yes         -
 *   ret         -            -           -           value/imm, or - for void
 *
 * A field marked "-" must hold its empty form: MIR_NONE for a value or
 * block id, mir_none() for an operand, mir_at_object(MIR_NONE, 0) for an
 * address, NULL for a target. The builder calls above always produce that;
 * mir_verify rejects anything else, so no consumer ever has to wonder
 * whether a stray id in an unused field is meaningful. */
enum { MIR_FIELD_DST = 1, MIR_FIELD_A = 2, MIR_FIELD_B = 4, MIR_FIELD_ADDR = 8, MIR_FIELD_TARGET = 16 };
/* The MIR_FIELD_* set `op` uses; 0 for an unknown opcode. */
unsigned mir_opcode_fields(MirOpcode op);
/* The values `inst` reads, in operand order (at most 3), stored in `used`.
 * Only fields the opcode uses are considered. */
size_t mir_inst_uses(const MirInst *inst, uint32_t used[3]);
/* The value `inst` assigns, or MIR_NONE. */
uint32_t mir_inst_def(const MirInst *inst);
/* The value a terminator reads (cbr's condition, ret's value), or MIR_NONE. */
uint32_t mir_term_use(const MirTerminator *term);

/* ---- Verification ----
 *
 * Returns 0 when the module is well formed; otherwise -1 with a one-line
 * diagnostic in `message` (capacity `message_len`, may be 0). Checks:
 *   - the module was built without running out of memory
 *   - every instruction and terminator is in canonical form: each field its
 *     opcode or kind does not use holds the empty form (the table above), so
 *     a stray id in an unused field is an error, never silently ignored
 *   - objects: nonzero size, unique non-empty names; CONST objects carry an
 *     initializer of exactly their size, other kinds none; STACK objects name
 *     an existing owner function
 *   - functions: at least one block; parameters are the first values; every
 *     value has a legal type (I8/I16/I32/PTR)
 *   - every block ends in exactly one terminator, whose targets are blocks of
 *     the same function; cbr conditions are I8; ret matches the return type
 *   - every instruction: operand kinds, value ids and types agree with the
 *     opcode (binary operands and result share one integer type; CMP yields
 *     I8; ZEXT/SEXT widen and TRUNC narrows an integer value; ADDR and
 *     PTR_ADD yield PTR); immediates fit their type
 *   - every address names an existing object visible to the function (a
 *     STACK object only from its owner) or a PTR value, and an object-based
 *     access lies inside the object; nothing stores to a CONST object
 *   - MIR_TARGET regions name a target and existing objects
 *   - every use of a value is preceded by an assignment on every path from
 *     the entry block */
int mir_verify(const MirModule *module, char *message, size_t message_len);

// Function to dump the MIR for debugging purposes.
void mir_dump(MirModule *module, FILE * file_descriptor);

#endif /* OPTIFINE_MIR_H */
