/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.bcel.internal.classfile.CodeException;

public final class CodeExceptionGen implements InstructionTargeter, Cloneable {

    public CodeExceptionGen(final InstructionHandle startPc, final InstructionHandle endPc, final InstructionHandle handlerPc, final ObjectType catchType) {
    }

    public CodeException getCodeException(final ConstantPoolGen cp);

    public void setStartPC(final InstructionHandle start_pc);

    public void setEndPC(final InstructionHandle end_pc);

    public void setHandlerPC(final InstructionHandle handler_pc);

    @Override
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Override
    @Pure
    public boolean containsTarget(final InstructionHandle ih);

    public void setCatchType(final ObjectType catchType);

    public ObjectType getCatchType();

    public InstructionHandle getStartPC();

    public InstructionHandle getEndPC();

    public InstructionHandle getHandlerPC();

    @Override
    public String toString();

    @Override
    public Object clone();
}
