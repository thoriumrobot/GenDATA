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
import com.sun.org.apache.bcel.internal.Const;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.LocalVariable;

    @Positive
public class LocalVariableGen implements InstructionTargeter, NamedAndTyped, Cloneable {

    @Positive
    public LocalVariableGen(final int index, final String name, final Type type, final InstructionHandle start, final InstructionHandle end) {
    @Positive
    }

    @Positive
    public LocalVariableGen(final int index, final String name, final Type type, final InstructionHandle start, final InstructionHandle end, final int origIndex) {
    @Positive
    }

    @Positive
    public LocalVariable getLocalVariable(final ConstantPoolGen cp);

    @Positive
    public void setIndex(final int index);

    @Positive
    public int getIndex();

    @Positive
    public int getOrigIndex();

    @Positive
    public void setLiveToEnd(final boolean live_to_end);

    @Positive
    public boolean getLiveToEnd();

    @Positive
    @Override
    @Positive
    public void setName(final String name);

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    @Override
    @Positive
    public void setType(final Type type);

    @Positive
    @Override
    @Positive
    public Type getType();

    @Positive
    public InstructionHandle getStart();

    @Positive
    public InstructionHandle getEnd();

    @Positive
    public void setStart(final InstructionHandle start);

    @Positive
    public void setEnd(final InstructionHandle end);

    @Positive
    @Override
    @Positive
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Positive
    void dispose();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsTarget(final InstructionHandle ih);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public boolean equals(final Object o);

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
