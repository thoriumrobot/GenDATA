/*
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.bcel.internal.Const;
import com.sun.org.apache.bcel.internal.classfile.Constant;
import com.sun.org.apache.bcel.internal.util.ByteSequence;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public class InstructionList implements Iterable<InstructionHandle> {

    public InstructionList() {
    }

    public InstructionList(final Instruction i) {
    }

    public InstructionList(final BranchInstruction i) {
    }

    public InstructionList(final CompoundInstruction c) {
    }

    public boolean isEmpty();

    public static InstructionHandle findHandle(final InstructionHandle[] ihs, final int[] pos, final int count, final int target);

    public InstructionHandle findHandle(final int pos);

    public InstructionList(final byte[] code) {
    }

    public InstructionHandle append(final InstructionHandle ih, final InstructionList il);

    public InstructionHandle append(final Instruction i, final InstructionList il);

    public InstructionHandle append(final InstructionList il);

    public InstructionHandle append(final Instruction i);

    public BranchHandle append(final BranchInstruction i);

    public InstructionHandle append(final Instruction i, final Instruction j);

    public InstructionHandle append(final Instruction i, final CompoundInstruction c);

    public InstructionHandle append(final CompoundInstruction c);

    public InstructionHandle append(final InstructionHandle ih, final CompoundInstruction c);

    public InstructionHandle append(final InstructionHandle ih, final Instruction i);

    public BranchHandle append(final InstructionHandle ih, final BranchInstruction i);

    public InstructionHandle insert(final InstructionHandle ih, final InstructionList il);

    public InstructionHandle insert(final InstructionList il);

    public InstructionHandle insert(final Instruction i, final InstructionList il);

    public InstructionHandle insert(final Instruction i);

    public BranchHandle insert(final BranchInstruction i);

    public InstructionHandle insert(final Instruction i, final Instruction j);

    public InstructionHandle insert(final Instruction i, final CompoundInstruction c);

    public InstructionHandle insert(final CompoundInstruction c);

    public InstructionHandle insert(final InstructionHandle ih, final Instruction i);

    public InstructionHandle insert(final InstructionHandle ih, final CompoundInstruction c);

    public BranchHandle insert(final InstructionHandle ih, final BranchInstruction i);

    public void move(final InstructionHandle start, final InstructionHandle end, final InstructionHandle target);

    public void move(final InstructionHandle ih, final InstructionHandle target);

    public void delete(final InstructionHandle ih) throws TargetLostException;

    public void delete(final Instruction i) throws TargetLostException;

    public void delete(final InstructionHandle from, final InstructionHandle to) throws TargetLostException;

    public void delete(final Instruction from, final Instruction to) throws TargetLostException;

    @Pure
    public boolean contains(final InstructionHandle i);

    @Pure
    public boolean contains(final Instruction i);

    public void setPositions();

    public void setPositions(final boolean check);

    public byte[] getByteCode();

    public Instruction[] getInstructions();

    @Override
    public String toString();

    public String toString(final boolean verbose);

    @Override
    public Iterator<InstructionHandle> iterator();

    public InstructionHandle[] getInstructionHandles();

    public int[] getInstructionPositions();

    public InstructionList copy();

    public void replaceConstantPool(final ConstantPoolGen old_cp, final ConstantPoolGen new_cp);

    public void dispose();

    public InstructionHandle getStart();

    public InstructionHandle getEnd();

    public int getLength();

    public int size();

    public void redirectBranches(final InstructionHandle old_target, final InstructionHandle new_target);

    public void redirectLocalVariables(final LocalVariableGen[] lg, final InstructionHandle old_target, final InstructionHandle new_target);

    public void redirectExceptionHandlers(final CodeExceptionGen[] exceptions, final InstructionHandle old_target, final InstructionHandle new_target);

    public void addObserver(final InstructionListObserver o);

    public void removeObserver(final InstructionListObserver o);

    public void update();
}
