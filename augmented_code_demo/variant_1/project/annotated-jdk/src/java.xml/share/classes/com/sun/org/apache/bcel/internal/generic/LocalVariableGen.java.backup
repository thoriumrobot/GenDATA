/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.bcel.internal.Const;
import com.sun.org.apache.bcel.internal.classfile.LocalVariable;

public class LocalVariableGen implements InstructionTargeter, NamedAndTyped, Cloneable {

    public LocalVariableGen(final int index, final String name, final Type type, final InstructionHandle start, final InstructionHandle end) {
    }

    public LocalVariableGen(final int index, final String name, final Type type, final InstructionHandle start, final InstructionHandle end, final int origIndex) {
    }

    public LocalVariable getLocalVariable(final ConstantPoolGen cp);

    public void setIndex(final int index);

    public int getIndex();

    public int getOrigIndex();

    public void setLiveToEnd(final boolean live_to_end);

    public boolean getLiveToEnd();

    @Override
    public void setName(final String name);

    @Override
    public String getName();

    @Override
    public void setType(final Type type);

    @Override
    public Type getType();

    public InstructionHandle getStart();

    public InstructionHandle getEnd();

    public void setStart(final InstructionHandle start);

    public void setEnd(final InstructionHandle end);

    @Override
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    void dispose();

    @Override
    @Pure
    public boolean containsTarget(final InstructionHandle ih);

    @Override
    public int hashCode();

    @Override
    public boolean equals(final Object o);

    @Override
    public String toString();

    @Override
    public Object clone();
}
