/*
    @Positive
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.bcel.internal.generic;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.DataOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import com.sun.org.apache.bcel.internal.util.ByteSequence;

    @Positive
public abstract class Select extends BranchInstruction implements VariableLengthInstruction, StackConsumer, StackProducer {

    @Positive
    @Override
    @Positive
    protected int updatePosition(final int offset, final int max_offset);

    @Positive
    @Override
    @Positive
    public void dump(final DataOutputStream out) throws IOException;

    @Positive
    @Override
    @Positive
    protected void initFromFile(final ByteSequence bytes, final boolean wide) throws IOException;

    @Positive
    @Override
    @Positive
    public String toString(final boolean verbose);

    @Positive
    public void setTarget(final int i, final InstructionHandle target);

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
    @Override
    @Positive
    protected Object clone() throws CloneNotSupportedException;

    @Positive
    @Override
    @Positive
    void dispose();

    @Positive
    public int[] getMatchs();

    @Positive
    public int[] getIndices();

    @Positive
    public InstructionHandle[] getTargets();

    @Positive
    final int getMatch(final int index);

    @Positive
    final int getIndices(final int index);

    @Positive
    final InstructionHandle getTarget(final int index);

    @Positive
    final int getFixed_length();

    @Positive
    final void setFixed_length(final int fixed_length);

    @Positive
    final int getMatch_length();

    @Positive
    final int setMatch_length(final int match_length);

    @Positive
    final void setMatch(final int index, final int value);

    @Positive
    final void setIndices(final int[] array);

    @Positive
    final void setMatches(final int[] array);

    @Positive
    final void setTargets(final InstructionHandle[] array);

    @Positive
    final int getPadding();

    @Positive
    final int setIndices(final int i, final int value);
    @Positive
}

// CFWR semantic augmentation - variant 1
