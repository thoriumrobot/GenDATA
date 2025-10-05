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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.org.apache.bcel.internal.Const;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.Constant;
    @Positive
import com.sun.org.apache.bcel.internal.util.ByteSequence;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.DataOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;

    @Positive
public class InstructionList implements Iterable<InstructionHandle> {

    @Positive
    public InstructionList() {
    @Positive
    }

    @Positive
    public InstructionList(final Instruction i) {
    @Positive
    }

    @Positive
    public InstructionList(final BranchInstruction i) {
    @Positive
    }

    @Positive
    public InstructionList(final CompoundInstruction c) {
    @Positive
    }

    @Positive
    public boolean isEmpty();

    @Positive
    public static InstructionHandle findHandle(final InstructionHandle[] ihs, final int[] pos, final int count, final int target);

    @Positive
    public InstructionHandle findHandle(final int pos);

    @Positive
    public InstructionList(final byte[] code) {
    @Positive
    }

    @Positive
    public InstructionHandle append(final InstructionHandle ih, final InstructionList il);

    @Positive
    public InstructionHandle append(final Instruction i, final InstructionList il);

    @Positive
    public InstructionHandle append(final InstructionList il);

    @Positive
    public InstructionHandle append(final Instruction i);

    @Positive
    public BranchHandle append(final BranchInstruction i);

    @Positive
    public InstructionHandle append(final Instruction i, final Instruction j);

    @Positive
    public InstructionHandle append(final Instruction i, final CompoundInstruction c);

    @Positive
    public InstructionHandle append(final CompoundInstruction c);

    @Positive
    public InstructionHandle append(final InstructionHandle ih, final CompoundInstruction c);

    @Positive
    public InstructionHandle append(final InstructionHandle ih, final Instruction i);

    @Positive
    public BranchHandle append(final InstructionHandle ih, final BranchInstruction i);

    @Positive
    public InstructionHandle insert(final InstructionHandle ih, final InstructionList il);

    @Positive
    public InstructionHandle insert(final InstructionList il);

    @Positive
    public InstructionHandle insert(final Instruction i, final InstructionList il);

    @Positive
    public InstructionHandle insert(final Instruction i);

    @Positive
    public BranchHandle insert(final BranchInstruction i);

    @Positive
    public InstructionHandle insert(final Instruction i, final Instruction j);

    @Positive
    public InstructionHandle insert(final Instruction i, final CompoundInstruction c);

    @Positive
    public InstructionHandle insert(final CompoundInstruction c);

    @Positive
    public InstructionHandle insert(final InstructionHandle ih, final Instruction i);

    @Positive
    public InstructionHandle insert(final InstructionHandle ih, final CompoundInstruction c);

    @Positive
    public BranchHandle insert(final InstructionHandle ih, final BranchInstruction i);

    @Positive
    public void move(final InstructionHandle start, final InstructionHandle end, final InstructionHandle target);

    @Positive
    public void move(final InstructionHandle ih, final InstructionHandle target);

    @Positive
    public void delete(final InstructionHandle ih) throws TargetLostException;

    @Positive
    public void delete(final Instruction i) throws TargetLostException;

    @Positive
    public void delete(final InstructionHandle from, final InstructionHandle to) throws TargetLostException;

    @Positive
    public void delete(final Instruction from, final Instruction to) throws TargetLostException;

    @Positive
    @Pure
    @Positive
    public boolean contains(final InstructionHandle i);

    @Positive
    @Pure
    @Positive
    public boolean contains(final Instruction i);

    @Positive
    public void setPositions();

    @Positive
    public void setPositions(final boolean check);

    @Positive
    public byte[] getByteCode();

    @Positive
    public Instruction[] getInstructions();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public String toString(final boolean verbose);

    @Positive
    @Override
    @Positive
    public Iterator<InstructionHandle> iterator();

    @Positive
    public InstructionHandle[] getInstructionHandles();

    @Positive
    public int[] getInstructionPositions();

    @Positive
    public InstructionList copy();

    @Positive
    public void replaceConstantPool(final ConstantPoolGen old_cp, final ConstantPoolGen new_cp);

    @Positive
    public void dispose();

    @Positive
    public InstructionHandle getStart();

    @Positive
    public InstructionHandle getEnd();

    @Positive
    public int getLength();

    @Positive
    public int size();

    @Positive
    public void redirectBranches(final InstructionHandle old_target, final InstructionHandle new_target);

    @Positive
    public void redirectLocalVariables(final LocalVariableGen[] lg, final InstructionHandle old_target, final InstructionHandle new_target);

    @Positive
    public void redirectExceptionHandlers(final CodeExceptionGen[] exceptions, final InstructionHandle old_target, final InstructionHandle new_target);

    @Positive
    public void addObserver(final InstructionListObserver o);

    @Positive
    public void removeObserver(final InstructionListObserver o);

    @Positive
    public void update();
    @Positive
}

// CFWR semantic augmentation - variant 0
