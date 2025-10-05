/*
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import java.io.DataOutputStream;
import java.io.IOException;
import com.sun.org.apache.bcel.internal.util.ByteSequence;

public abstract class Select extends BranchInstruction implements VariableLengthInstruction, StackConsumer, StackProducer {

    @Override
    protected int updatePosition(final int offset, final int max_offset);

    @Override
    public void dump(final DataOutputStream out) throws IOException;

    @Override
    protected void initFromFile(final ByteSequence bytes, final boolean wide) throws IOException;

    @Override
    public String toString(final boolean verbose);

    public void setTarget(final int i, final InstructionHandle target);

    @Override
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Override
    @Pure
    public boolean containsTarget(final InstructionHandle ih);

    @Override
    protected Object clone() throws CloneNotSupportedException;

    @Override
    void dispose();

    public int[] getMatchs();

    public int[] getIndices();

    public InstructionHandle[] getTargets();

    final int getMatch(final int index);

    final int getIndices(final int index);

    final InstructionHandle getTarget(final int index);

    final int getFixed_length();

    final void setFixed_length(final int fixed_length);

    final int getMatch_length();

    final int setMatch_length(final int match_length);

    final void setMatch(final int index, final int value);

    final void setIndices(final int[] array);

    final void setMatches(final int[] array);

    final void setTargets(final InstructionHandle[] array);

    final int getPadding();

    final int setIndices(final int i, final int value);
}
