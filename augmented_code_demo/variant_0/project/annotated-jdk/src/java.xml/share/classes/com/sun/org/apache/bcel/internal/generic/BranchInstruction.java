/*
    @Positive
 * Copyright (c) 2017, 2020, Oracle and/or its affiliates. All rights reserved.
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
public abstract class BranchInstruction extends Instruction implements InstructionTargeter {

    @Positive
    protected BranchInstruction(final short opcode, final InstructionHandle target) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void dump(final DataOutputStream out) throws IOException;

    @Positive
    protected int getTargetOffset(final InstructionHandle _target);

    @Positive
    protected int getTargetOffset();

    @Positive
    protected int updatePosition(final int offset, final int max_offset);

    @Positive
    @Override
    @Positive
    public String toString(final boolean verbose);

    @Positive
    @Override
    @Positive
    protected void initFromFile(final ByteSequence bytes, final boolean wide) throws IOException;

    @Positive
    public final int getIndex();

    @Positive
    public InstructionHandle getTarget();

    @Positive
    public void setTarget(final InstructionHandle target);

    @Positive
    static void notifyTarget(final InstructionHandle old_ih, final InstructionHandle new_ih, final InstructionTargeter t);

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
    void setOpcode(final short opcode);

    @Positive
    @Override
    @Positive
    void dispose();

    @Positive
    protected int getPosition();

    @Positive
    protected void setPosition(final int position);

    @Positive
    protected void setIndex(final int index);
    @Positive
}

// CFWR semantic augmentation - variant 0
