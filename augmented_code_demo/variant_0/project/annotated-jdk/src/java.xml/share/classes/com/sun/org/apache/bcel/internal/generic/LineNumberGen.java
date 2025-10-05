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
import java.util.Objects;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.LineNumber;

    @Positive
public class LineNumberGen implements InstructionTargeter, Cloneable {

    @Positive
    public LineNumberGen(final InstructionHandle ih, final int src_line) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsTarget(final InstructionHandle ih);

    @Positive
    @Override
    @Positive
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    @Positive
    public LineNumber getLineNumber();

    @Positive
    public void setInstruction(final InstructionHandle instructionHandle);

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    public InstructionHandle getInstruction();

    @Positive
    public void setSourceLine(final int src_line);

    @Positive
    public int getSourceLine();
    @Positive
}

// CFWR semantic augmentation - variant 0
