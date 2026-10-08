#include "optifine/mir.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#define DUMP_MIR false

size_t mir_int_bytes(MirType type) {
    switch (type) {
        case MIR_TYPE_I8:
            return 1;
        case MIR_TYPE_I16:
            return 2;
        case MIR_TYPE_I32:
            return 4;
        default:
            return 0;
    }
}

void mir_module_init(MirModule *module) {
    memset(module, 0, sizeof(*module));
}

void mir_target_code_free(MirTargetCode *code) {
    if (!code) return;
    if (code->free_payload) code->free_payload(code->payload);
    free(code->reads);
    free(code->writes);
    free(code);
}

void mir_module_free(MirModule *module) {
    for (size_t i = 0; i < module->num_objects; i++) free(module->objects[i].init);
    free(module->objects);
    for (size_t f = 0; f < module->num_functions; f++) {
        MirFunction *fn = &module->functions[f];
        for (size_t b = 0; b < fn->num_blocks; b++) {
            MirBlock *block = &fn->blocks[b];
            for (size_t i = 0; i < block->count; i++) {
                if (block->insts[i].op == MIR_TARGET) mir_target_code_free(block->insts[i].target);
            }
            free(block->insts);
        }
        free(fn->blocks);
        free(fn->value_types);
    }
    free(module->functions);
    memset(module, 0, sizeof(*module));
}

/* Grows *items (count used, *capacity allocated, `size` bytes each) to hold
 * one more. Returns 0, or -1 leaving everything unchanged. */
static int grow(void **items, size_t count, size_t *capacity, size_t size) {
    if (count < *capacity) return 0;
    size_t new_capacity = *capacity ? *capacity * 2 : 8;
    if (new_capacity < *capacity || new_capacity > SIZE_MAX / size) return -1;
    void *grown = realloc(*items, new_capacity * size);
    if (!grown) return -1;
    *items = grown;
    *capacity = new_capacity;
    return 0;
}

static void copy_name(char *dst, size_t cap, const char *src) {
    size_t n = src ? strlen(src) : 0;
    if (n >= cap) n = cap - 1; /* mir_verify rejects a name that does not fit */
    memcpy(dst, src ? src : "", n);
    dst[n] = '\0';
}

uint32_t mir_add_object(MirModule *module, MirMemKind kind, size_t size, const char *name, uint32_t function) {
    if (grow((void **)&module->objects, module->num_objects, &module->objects_capacity, sizeof(MirMemObject)) != 0) {
        module->out_of_memory = 1;
        return MIR_NONE;
    }
    MirMemObject *obj = &module->objects[module->num_objects];
    memset(obj, 0, sizeof(*obj));
    obj->kind = kind;
    obj->size = size;
    obj->function = function;
    copy_name(obj->name, sizeof(obj->name), name);
    if (name && strlen(name) >= sizeof(obj->name)) obj->name[0] = '\0';
    return (uint32_t)module->num_objects++;
}

int mir_object_set_init(MirModule *module, uint32_t object, const uint8_t *bytes, size_t size) {
    MirMemObject *obj = &module->objects[object];
    uint8_t *copy = malloc(size > 0 ? size : 1);
    if (!copy) {
        module->out_of_memory = 1;
        return -1;
    }
    memcpy(copy, bytes, size);
    free(obj->init);
    obj->init = copy;
    return 0;
}

uint32_t mir_add_function(MirModule *module, const char *name, MirType return_type) {
    if (grow((void **)&module->functions, module->num_functions, &module->functions_capacity,
             sizeof(MirFunction)) != 0) {
        module->out_of_memory = 1;
        return MIR_NONE;
    }
    MirFunction *fn = &module->functions[module->num_functions];
    memset(fn, 0, sizeof(*fn));
    copy_name(fn->name, sizeof(fn->name), name);
    fn->return_type = return_type;
    return (uint32_t)module->num_functions++;
}

uint32_t mir_new_value(MirModule *module, uint32_t function, MirType type) {
    MirFunction *fn = &module->functions[function];
    if (grow((void **)&fn->value_types, fn->num_values, &fn->values_capacity, sizeof(MirType)) != 0) {
        module->out_of_memory = 1;
        return MIR_NONE;
    }
    fn->value_types[fn->num_values] = type;
    return (uint32_t)fn->num_values++;
}

uint32_t mir_add_param(MirModule *module, uint32_t function, MirType type) {
    MirFunction *fn = &module->functions[function];
    if (fn->num_values != fn->num_params) {
        /* A parameter after an ordinary value: rejected by mir_verify, since
         * parameters must be values 0 .. num_params-1. */
        fn->num_params = SIZE_MAX;
        return MIR_NONE;
    }
    uint32_t v = mir_new_value(module, function, type);
    if (v != MIR_NONE) fn->num_params++;
    return v;
}

uint32_t mir_add_block(MirModule *module, uint32_t function) {
    MirFunction *fn = &module->functions[function];
    if (grow((void **)&fn->blocks, fn->num_blocks, &fn->blocks_capacity, sizeof(MirBlock)) != 0) {
        module->out_of_memory = 1;
        return MIR_NONE;
    }
    memset(&fn->blocks[fn->num_blocks], 0, sizeof(MirBlock));
    return (uint32_t)fn->num_blocks++;
}

int mir_append(MirModule *module, uint32_t function, uint32_t block, const MirInst *inst) {
    MirBlock *b = &module->functions[function].blocks[block];
    if (grow((void **)&b->insts, b->count, &b->capacity, sizeof(MirInst)) != 0) {
        module->out_of_memory = 1;
        return -1;
    }
    b->insts[b->count++] = *inst;
    return 0;
}

void mir_set_terminator(MirModule *module, uint32_t function, uint32_t block, const MirTerminator *term) {
    module->functions[function].blocks[block].term = *term;
}

MirOperand mir_value(uint32_t value) {
    MirOperand o = {MIR_OPND_VALUE, value, 0};
    return o;
}

MirOperand mir_imm(int64_t imm) {
    MirOperand o = {MIR_OPND_IMM, MIR_NONE, imm};
    return o;
}

MirOperand mir_none(void) {
    MirOperand o = {MIR_OPND_NONE, MIR_NONE, 0};
    return o;
}

MirAddress mir_at_object(uint32_t object, int32_t offset) {
    MirAddress a = {object, MIR_NONE, offset};
    return a;
}

MirAddress mir_at_pointer(uint32_t pointer, int32_t offset) {
    MirAddress a = {MIR_NONE, pointer, offset};
    return a;
}

static MirInst blank(MirOpcode op, MirType type, uint32_t dst, uint32_t origin) {
    MirInst inst;
    memset(&inst, 0, sizeof(inst));
    inst.op = op;
    inst.type = type;
    inst.dst = dst;
    inst.a = mir_none();
    inst.b = mir_none();
    inst.addr = mir_at_object(MIR_NONE, 0);
    inst.origin = origin;
    return inst;
}

int mir_emit_const(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, MirType type, int64_t imm,
                   uint32_t origin) {
    MirInst inst = blank(MIR_CONST, type, dst, origin);
    inst.a = mir_imm(imm);
    return mir_append(m, fn, block, &inst);
}

int mir_emit_binary(MirModule *m, uint32_t fn, uint32_t block, MirOpcode op, uint32_t dst, MirType type,
                    MirOperand a, MirOperand b, uint32_t origin) {
    MirInst inst = blank(op, type, dst, origin);
    inst.a = a;
    inst.b = b;
    return mir_append(m, fn, block, &inst);
}

int mir_emit_convert(MirModule *m, uint32_t fn, uint32_t block, MirOpcode op, uint32_t dst, MirType type,
                     uint32_t a, uint32_t origin) {
    MirInst inst = blank(op, type, dst, origin);
    inst.a = mir_value(a);
    return mir_append(m, fn, block, &inst);
}

int mir_emit_cmp(MirModule *m, uint32_t fn, uint32_t block, MirCmpPred pred, uint32_t dst, MirType type,
                 MirOperand a, MirOperand b, uint32_t origin) {
    MirInst inst = blank(MIR_CMP, type, dst, origin);
    inst.pred = pred;
    inst.a = a;
    inst.b = b;
    return mir_append(m, fn, block, &inst);
}

int mir_emit_load(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, MirType type, MirAddress addr,
                  uint32_t origin) {
    MirInst inst = blank(MIR_LOAD, type, dst, origin);
    inst.addr = addr;
    return mir_append(m, fn, block, &inst);
}

int mir_emit_store(MirModule *m, uint32_t fn, uint32_t block, MirType type, MirAddress addr, MirOperand value,
                   uint32_t origin) {
    MirInst inst = blank(MIR_STORE, type, MIR_NONE, origin);
    inst.addr = addr;
    inst.a = value;
    return mir_append(m, fn, block, &inst);
}

int mir_emit_addr(MirModule *m, uint32_t fn, uint32_t block, uint32_t dst, uint32_t object, int32_t offset,
                  uint32_t origin) {
    MirInst inst = blank(MIR_ADDR, MIR_TYPE_PTR, dst, origin);
    inst.addr = mir_at_object(object, offset);
    return mir_append(m, fn, block, &inst);
}

int mir_emit_target(MirModule *m, uint32_t fn, uint32_t block, MirTargetCode *code, uint32_t origin) {
    MirInst inst = blank(MIR_TARGET, MIR_TYPE_VOID, MIR_NONE, origin);
    inst.target = code;
    if (mir_append(m, fn, block, &inst) != 0) {
        mir_target_code_free(code);
        return -1;
    }
    return 0;
}

unsigned mir_opcode_fields(MirOpcode op) {
    switch (op) {
        case MIR_CONST:
        case MIR_COPY:
        case MIR_ZEXT:
        case MIR_SEXT:
        case MIR_TRUNC:
            return MIR_FIELD_DST | MIR_FIELD_A;
        case MIR_ADD:
        case MIR_SUB:
        case MIR_MUL:
        case MIR_AND:
        case MIR_OR:
        case MIR_XOR:
        case MIR_CMP:
        case MIR_PTR_ADD:
            return MIR_FIELD_DST | MIR_FIELD_A | MIR_FIELD_B;
        case MIR_LOAD:
        case MIR_ADDR:
            return MIR_FIELD_DST | MIR_FIELD_ADDR;
        case MIR_STORE:
            return MIR_FIELD_A | MIR_FIELD_ADDR;
        case MIR_TARGET:
            return MIR_FIELD_TARGET;
    }
    return 0;
}

size_t mir_inst_uses(const MirInst *inst, uint32_t used[3]) {
    unsigned fields = mir_opcode_fields(inst->op);
    size_t n = 0;
    if ((fields & MIR_FIELD_A) && inst->a.kind == MIR_OPND_VALUE) used[n++] = inst->a.value;
    if ((fields & MIR_FIELD_B) && inst->b.kind == MIR_OPND_VALUE) used[n++] = inst->b.value;
    if ((fields & MIR_FIELD_ADDR) && inst->op != MIR_ADDR && inst->addr.pointer != MIR_NONE) {
        used[n++] = inst->addr.pointer;
    }
    return n;
}

uint32_t mir_inst_def(const MirInst *inst) {
    return (mir_opcode_fields(inst->op) & MIR_FIELD_DST) ? inst->dst : MIR_NONE;
}

uint32_t mir_term_use(const MirTerminator *term) {
    if (term->kind == MIR_TERM_CBR && term->cond.kind == MIR_OPND_VALUE) return term->cond.value;
    if (term->kind == MIR_TERM_RET && term->value.kind == MIR_OPND_VALUE) return term->value.value;
    return MIR_NONE;
}

void mir_br(MirModule *m, uint32_t fn, uint32_t block, uint32_t target) {
    MirTerminator t = {MIR_TERM_BR, mir_none(), target, MIR_NONE, mir_none()};
    mir_set_terminator(m, fn, block, &t);
}

void mir_cbr(MirModule *m, uint32_t fn, uint32_t block, MirOperand cond, uint32_t then_block,
             uint32_t else_block) {
    MirTerminator t = {MIR_TERM_CBR, cond, then_block, else_block, mir_none()};
    mir_set_terminator(m, fn, block, &t);
}

void mir_ret(MirModule *m, uint32_t fn, uint32_t block, MirOperand value) {
    MirTerminator t = {MIR_TERM_RET, mir_none(), MIR_NONE, MIR_NONE, value};
    mir_set_terminator(m, fn, block, &t);
}

char* mir_dump_mem_kind(MirMemKind mem_kind) {
    switch (mem_kind) {
    case MIR_MEM_GLOBAL:
        return "Global";
    case MIR_MEM_STACK:
        return "Stack";
    case MIR_MEM_SCRATCH:
        return "Scratch";
    case MIR_MEM_CONST:
        return "Constant";
    case MIR_MEM_TENSOR:
        return "Tensor";
    }
}

char* mir_dump_type(MirType type) {
    switch (type) {
    case MIR_TYPE_VOID:
        return "Void";
    case MIR_TYPE_I8:
        return "I8";
    case MIR_TYPE_I16:
        return "I16";
    case MIR_TYPE_I32:
        return "I32";
    case MIR_TYPE_PTR:
        return "Ptr";
    }
}

char* mir_dump_opcode(MirOpcode op) {
    switch (op) {
    case MIR_CONST  : return "CONST";
    case MIR_COPY   : return "COPY";
    case MIR_ADD    : return "ADD";
    case MIR_SUB    : return "SUB";
    case MIR_MUL    : return "MUL";
    case MIR_AND    : return "AND";
    case MIR_OR     : return "OR";
    case MIR_XOR    : return "XOR";
    case MIR_CMP    : return "CMP";
    case MIR_ZEXT   : return "ZEXT";
    case MIR_SEXT   : return "SEXT";
    case MIR_TRUNC  : return "TRUNC";
    case MIR_LOAD   : return "LOAD";
    case MIR_STORE  : return "STORE";
    case MIR_ADDR   : return "ADDR";
    case MIR_TARGET : return "TARGET";
    case MIR_PTR_ADD : return "PTR_ADD";
    }
}

char* mir_dump_operand_kind(MirOperandKind opk) {
    switch(opk) {
    case MIR_OPND_NONE: return "OPND_NONE";
    case MIR_OPND_VALUE: return "OPND_VALUE";
    case MIR_OPND_IMM: return "OPND_IMM";
    }
}

char* mir_dump_pred(MirCmpPred cmp) {
    switch (cmp) {
    case MIR_CMP_EQ: return "CMP_EQ";
    case MIR_CMP_NE: return "CMP_NE";
    case MIR_CMP_ULT: return "CMP_ULT";
    case MIR_CMP_SLT: return "CMP_SLT";
    }
}

char* mir_dump_term_kind(MirTermKind k){
    switch(k) {
    case MIR_TERM_NONE: return "TERM_NONE";
    case MIR_TERM_BR: return "TERM_BR";
    case MIR_TERM_CBR: return "TERM_CBR";
    case MIR_TERM_RET: return "TERM_RET";
    }
}

void mir_dump_operand(MirOperand oper, FILE * fd) {
    if (oper.kind == MIR_OPND_NONE) {
        return;   
    }
    fprintf(fd, "        Kind:  %s\n", mir_dump_operand_kind(oper.kind));
    fprintf(fd, "        Value: %d\n", oper.value);
    fprintf(fd, "        Imm:   %lld\n", oper.imm);
}

void mir_dump_block(MirBlock *block, FILE * fd) {
    for(size_t i=0; i<block->count; i++) {
        fprintf(fd, "    Inst:\n");
        MirInst inst = block->insts[i];
        fprintf(fd, "      Op: %s\n", mir_dump_opcode(inst.op));
        fprintf(fd, "      Type: %s\n", mir_dump_type(inst.type));
        fprintf(fd, "      Result Value: %u\n", inst.dst);
        fprintf(fd, "      Operand a: \n");
        mir_dump_operand(inst.a, fd);
        fprintf(fd, "      Operand b: \n");
        mir_dump_operand(inst.b, fd);
        fprintf(fd, "      Pred: %s\n", mir_dump_pred(inst.pred));
        fprintf(fd, "      Addr: Obj - %d, Ptr - %d, Offset - %d \n", inst.addr.object, inst.addr.pointer, inst.addr.offset);
    }

    fprintf(fd, "    Terminator:\n");
    fprintf(fd, "      Kind: %s\n", mir_dump_term_kind(block->term.kind));
    fprintf(fd, "      Cond: \n");
    mir_dump_operand(block->term.cond, fd);
    fprintf(fd, "      Then Block: %d\n", block->term.then_block);
    fprintf(fd, "      Else Block: %d\n", block->term.else_block);
    fprintf(fd, "      Return Op:\n");
    mir_dump_operand(block->term.value, fd);
}

void mir_dump(MirModule *module, FILE * fd) {
    if(!DUMP_MIR) return;
    fprintf(fd, "%zu memory objects:\n", module->num_objects);
    for(size_t i=0; i<module->num_objects; i++) {
        MirMemObject *mem_obj = &module->objects[i];
        fprintf(fd, "  Name:            %s\n", mem_obj->name);
        fprintf(fd, "  Kind:            %s\n", mir_dump_mem_kind(mem_obj->kind));
        fprintf(fd, "  Size:            %zu\n", mem_obj->size);
        fprintf(fd, "  Owning Function: %s\n", mem_obj->kind == MIR_MEM_STACK ? module->functions[mem_obj->function].name : "None");
        fprintf(fd, "\n");
    }

    fprintf(fd, "%zu functions:\n", module->num_functions);
    for(size_t i=0; i<module->num_functions; i++) {
        MirFunction *func = &module->functions[i];
        fprintf(fd, "  Name:        %s\n", func->name);
        fprintf(fd, "  Return Type: %s\n", mir_dump_type(func->return_type));
        fprintf(fd, "  Value Type\n");
        for(size_t i=0; i<func->num_values; i++) {
            fprintf(fd, "    for %zu: %s\n", i, mir_dump_type(func->value_types[i]));
        }
        fprintf(fd, "  Number of Params: %zu\n", func->num_params);
        for(size_t i=0; i<func->num_blocks; i++) {
            fprintf(fd, "   Block %zu:\n", i);
            mir_dump_block(&func->blocks[i], fd);
        }
    }
    fprintf(fd, "\n");
}
