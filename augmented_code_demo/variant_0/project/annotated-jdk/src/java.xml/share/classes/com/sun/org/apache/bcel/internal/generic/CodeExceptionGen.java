/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.bcel.internal.generic;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.CodeException;

    @Positive
public final class CodeExceptionGen implements InstructionTargeter, Cloneable {

    @Positive
    public CodeExceptionGen(final InstructionHandle startPc, final InstructionHandle endPc, final InstructionHandle handlerPc, final ObjectType catchType) {
    @Positive
    }

    @Positive
    public CodeException getCodeException(final ConstantPoolGen cp);

    @Positive
    public void setStartPC(final InstructionHandle start_pc);

    @Positive
    public void setEndPC(final InstructionHandle end_pc);

    @Positive
    public void setHandlerPC(final InstructionHandle handler_pc);

    @Positive
    @Override
    @Positive
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsTarget(final InstructionHandle ih);

    @Positive
    public void setCatchType(final ObjectType catchType);

    @Positive
    public ObjectType getCatchType();

    @Positive
    public InstructionHandle getStartPC();

    @Positive
    public InstructionHandle getEndPC();

    @Positive
    public InstructionHandle getHandlerPC();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 0
