/*
 * Copyright (c) 2017, 2020, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import java.io.DataOutputStream;
import java.io.IOException;
import com.sun.org.apache.bcel.internal.util.ByteSequence;

public abstract class BranchInstruction extends Instruction implements InstructionTargeter {

    protected BranchInstruction(final short opcode, final InstructionHandle target) {
    }

    @Override
    public void dump(final DataOutputStream out) throws IOException;

    protected int getTargetOffset(final InstructionHandle _target);

    protected int getTargetOffset();

    protected int updatePosition(final int offset, final int max_offset);

    @Override
    public String toString(final boolean verbose);

    @Override
    protected void initFromFile(final ByteSequence bytes, final boolean wide) throws IOException;

    public final int getIndex();

    public InstructionHandle getTarget();

    public void setTarget(final InstructionHandle target);

    static void notifyTarget(final InstructionHandle old_ih, final InstructionHandle new_ih, final InstructionTargeter t);

    @Override
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Override
    @Pure
    public boolean containsTarget(final InstructionHandle ih);

    @Override
    void setOpcode(final short opcode);

    @Override
    void dispose();

    protected int getPosition();

    protected void setPosition(final int position);

    protected void setIndex(final int index);
}
