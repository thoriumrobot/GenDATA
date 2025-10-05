/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import java.util.Objects;
import com.sun.org.apache.bcel.internal.classfile.LineNumber;

public class LineNumberGen implements InstructionTargeter, Cloneable {

    public LineNumberGen(final InstructionHandle ih, final int src_line) {
    }

    @Override
    @Pure
    public boolean containsTarget(final InstructionHandle ih);

    @Override
    public void updateTarget(final InstructionHandle old_ih, final InstructionHandle new_ih);

    public LineNumber getLineNumber();

    public void setInstruction(final InstructionHandle instructionHandle);

    @Override
    public Object clone();

    public InstructionHandle getInstruction();

    public void setSourceLine(final int src_line);

    public int getSourceLine();
}
